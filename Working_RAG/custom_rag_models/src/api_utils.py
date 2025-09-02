import requests
import json
import base64
from io import BytesIO
from PIL import Image, ImageOps
import os
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.config import *

# Create a session with retry capabilities
def create_retry_session(retries=3, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504)):
    """
    Create a requests Session with retry capabilities
    
    Args:
        retries (int): Number of retries
        backoff_factor (float): Backoff factor for retry delay
        status_forcelist (tuple): Status codes to retry on
        
    Returns:
        requests.Session: Session with retry capabilities
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def extract_from_pdf(pdf_path, max_attempts=3, timeout=(30, 4000)):
    """
    Extract structured content from a scientific PDF using the extraction API
    
    Args:
        pdf_path (str): Path to the PDF file
        max_attempts (int): Maximum number of attempts to try extraction
        timeout (tuple): Connection and read timeout in seconds
        
    Returns:
        dict: Extracted content including text, tables, and figures
    """
    session = create_retry_session(retries=3)
    
    for attempt in range(max_attempts):
        try:
            print(f"PDF extraction attempt {attempt+1}/{max_attempts}...")
            
            # Check file exists and is accessible
            if not os.path.exists(pdf_path):
                print(f"Error: PDF file not found: {pdf_path}")
                return None
                
            file_size = os.path.getsize(pdf_path)
            if file_size == 0:
                print(f"Error: PDF file is empty: {pdf_path}")
                return None
                
            print(f"PDF file size: {file_size/1024:.2f} KB")
            
            with open(pdf_path, 'rb') as file:
                files = {'file': (os.path.basename(pdf_path), file, 'application/pdf')}
                response = session.post(
                    PDF_EXTRACTOR_API_ENDPOINT,
                    files=files,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    # Handle the new response format
                    response_json = response.json()
                    
                    # Log the response structure for debugging
                    log_response_structure(response_json)
                      # Check if the response has the new format
                    if response_json.get("status") == "success" and "data" in response_json:
                        print("Processing response using new format structure")
                        # Extract data from the new format
                        data = response_json["data"]
                        
                        # Convert the new format to the expected format
                        result = {
                            "title": data.get("title", "Untitled Paper"),
                            "authors": data.get("authors", []),
                            "abstract": data.get("abstract", ""),
                            "full_text": data.get("full_text", ""),
                            "sections": [],
                            "tables": [],
                            "figures": []
                        }
                        
                        # Process the full text into sections
                        full_text = data.get("full_text", "")
                        if full_text:
                            # Simple text parsing - split by potential section headers
                            # This is a basic approach - in production, you'd want more sophisticated section parsing
                            sections = []
                            
                            # First try to identify sections based on common section headers
                            section_keywords = ["abstract", "introduction", "method", "methodology", 
                                              "experiment", "result", "discussion", "conclusion", 
                                              "reference", "background", "related work"]
                            
                            # Simple section extraction - can be improved
                            lines = full_text.split('\n')
                            current_section = {"heading": "Introduction", "text": ""}
                            
                            for line in lines:
                                line_lower = line.lower()
                                is_heading = False
                                
                                # Check if this line might be a section heading
                                for keyword in section_keywords:
                                    if keyword in line_lower and len(line.strip()) < 100:
                                        # This looks like a section heading
                                        if current_section["text"].strip():
                                            # Save the previous section if it has content
                                            sections.append(current_section.copy())
                                        
                                        # Start a new section
                                        current_section = {"heading": line.strip(), "text": ""}
                                        is_heading = True
                                        break
                                        
                                if not is_heading:
                                    # Add line to the current section
                                    current_section["text"] += line + "\n"
                            
                            # Add the last section
                            if current_section["text"].strip():
                                sections.append(current_section)
                                
                            result["sections"] = sections
                          # Process tables
                        for idx, table in enumerate(data.get("tables", [])):
                            try:
                                image_data = None
                                if isinstance(table, dict) and table.get("image"):
                                    # Try to decode the base64 image
                                    try:
                                        image_data = base64.b64decode(table.get("image", ""))
                                    except Exception as e:
                                        print(f"Error decoding table image: {str(e)}")
                                
                                table_data = {
                                    "caption": table.get("caption", f"Table {idx+1}"),
                                    "image_data": image_data
                                }
                                result["tables"].append(table_data)
                            except Exception as e:
                                print(f"Error processing table {idx}: {str(e)}")
                        
                        # Process figures
                        for idx, figure in enumerate(data.get("figures", [])):
                            try:
                                image_data = None
                                if isinstance(figure, dict) and figure.get("image"):
                                    # Try to decode the base64 image
                                    try:
                                        image_data = base64.b64decode(figure.get("image", ""))
                                    except Exception as e:
                                        print(f"Error decoding figure image: {str(e)}")
                                
                                figure_data = {
                                    "caption": figure.get("caption", f"Figure {idx+1}"),
                                    "image_data": image_data
                                }
                                result["figures"].append(figure_data)
                            except Exception as e:
                                print(f"Error processing figure {idx}: {str(e)}")
                            
                        return result
                    else:
                        # Return the original format if it's not in the new format
                        print("Using original response format")
                        return response_json
                else:
                    print(f"Error extracting PDF content: {response.status_code}")
                    print(f"Response: {response.text}")
                    
                    # If this is not the last attempt, wait before retrying
                    if attempt < max_attempts - 1:
                        wait_time = (attempt + 1) * 2  # Progressive waiting: 2, 4, 6... seconds
                        print(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
        
        except Exception as e:
            print(f"Exception during PDF extraction (attempt {attempt+1}): {str(e)}")
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 2
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)            # Just return None when all attempts fail
            return None
    
    print("All PDF extraction attempts failed")
    return None

def classify_paper(text, max_attempts=3, timeout=(10, 4000)):
    """
    Classify the scientific paper into categories using the classification API
    
    Args:
        text (str): Text content of the paper
        max_attempts (int): Maximum number of attempts
        timeout (tuple): Connection and read timeout in seconds
        
    Returns:
        dict: Classification results with categories and confidence scores
    """
    session = create_retry_session(retries=2)
    
    for attempt in range(max_attempts):
        try:
            print(f"Paper classification attempt {attempt+1}/{max_attempts}...")
            
            # Check if text is valid
            if not text or len(text.strip()) < 50:
                print("Error: Text too short for classification")
                return None
                
            response = session.post(
                PAPER_CLASSIFICATION_API_ENDPOINT,
                json={"text": text},
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error classifying paper: {response.status_code}")
                print(f"Response: {response.text}")
                
                # If this is not the last attempt, wait before retrying
                if attempt < max_attempts - 1:
                    wait_time = (attempt + 1) * 1.5
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error during paper classification (attempt {attempt+1}): {str(e)}")
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 1.5
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        except requests.exceptions.Timeout as e:
            print(f"Timeout error during paper classification (attempt {attempt+1}): {str(e)}")
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 1.5
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        except Exception as e:
            print(f"Exception during paper classification (attempt {attempt+1}): {str(e)}")
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 1.5
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    
    print("All paper classification attempts failed")
    
    # Provide a default classification if all attempts fail
    return {"categories": ["unclassified"]}

def summarize_text(text, max_length=500, min_length=50, max_attempts=3, timeout=(10, 4000)):
    """
    Summarize text content using PEGASUS API
    
    Args:
        text (str): Text content to summarize
        max_length (int): Maximum summary length
        min_length (int): Minimum summary length
        max_attempts (int): Maximum number of attempts
        timeout (tuple): Connection and read timeout in seconds
        
    Returns:
        str: Generated summary
    """
    session = create_retry_session(retries=2)
    
    # Check if text is valid
    if not text or len(text.strip()) < min_length:
        print("Warning: Text too short for summarization")
        return text  # Return the original text if it's too short
    
    # Truncate excessively large text to avoid timeout issues
    max_input_chars = 20000  # PEGASUS typically has input limits
    if len(text) > max_input_chars:
        print(f"Warning: Truncating input text from {len(text)} to {max_input_chars} characters")
        text = text[:max_input_chars]
    
    for attempt in range(max_attempts):
        try:
            print(f"Text summarization attempt {attempt+1}/{max_attempts}...")
            
            response = session.post(
                TEXT_SUMMARIZATION_API_ENDPOINT,
                json={
                    "text": text,
                    "max_length": max_length,
                    "min_length": min_length
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                summary = response.json().get("summary", "")
                if summary:
                    return summary
                else:
                    print("Warning: Empty summary returned")
                    # If this is the last attempt, return a fallback
                    if attempt == max_attempts - 1:
                        # Try extracting the first few sentences as a fallback summary
                        sentences = text.split('.')
                        if len(sentences) >= 3:
                            return '. '.join(sentences[:3]) + '.'
                        else:
                            return text
            else:
                print(f"Error summarizing text: {response.status_code}")
                print(f"Response: {response.text}")
                
                # If this is not the last attempt, wait before retrying
                if attempt < max_attempts - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
        
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error during text summarization (attempt {attempt+1}): {str(e)}")
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 2
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        except requests.exceptions.Timeout as e:
            print(f"Timeout error during text summarization (attempt {attempt+1}): {str(e)}")
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 2
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        except Exception as e:
            print(f"Exception during text summarization (attempt {attempt+1}): {str(e)}")
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 2
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    
    print("All text summarization attempts failed")
    
    # Return a simplified extract as fallback
    sentences = text.split('.')
    if len(sentences) >= 3:
        return '. '.join(sentences[:3]) + '.'
    else:
        return text

def summarize_table(table_image_path, max_attempts=3, timeout=(15, 4000)):
    """
    Summarize table content using Qwen Table Summarizer API (multipart/form-data, 'file' field)
    Args:
        table_image_path (str): Path to the table image
        max_attempts (int): Maximum number of attempts
        timeout (tuple): Connection and read timeout in seconds
    Returns:
        str: Generated table summary
    """
    session = create_retry_session(retries=2)
    if not os.path.exists(table_image_path):
        print(f"Error: Table image file not found: {table_image_path}")
        return "Table description unavailable"
    try:
        file_size = os.path.getsize(table_image_path)
        if file_size == 0:
            print(f"Error: Table image file is empty: {table_image_path}")
            return "Table description unavailable"
        png_image_path = ensure_png_format(table_image_path)
        if png_image_path != table_image_path:
            print(f"Table image converted to PNG: {png_image_path}")
            table_image_path = png_image_path
        file_size = os.path.getsize(table_image_path)
        if file_size > 5 * 1024 * 1024:
            try:
                img = Image.open(table_image_path)
                max_dimension = 1024
                width, height = img.size
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                img = img.resize((new_width, new_height), Image.LANCZOS)
                resized_path = os.path.splitext(table_image_path)[0] + '_resized.png'
                img.save(resized_path, format='PNG')
                print(f"Resized image saved to {resized_path}")
                table_image_path = resized_path
            except Exception as e:
                print(f"Error resizing image: {str(e)}")
    except Exception as e:
        print(f"Error checking image file: {str(e)}")
    for attempt in range(max_attempts):
        try:
            print(f"Table summarization attempt {attempt+1}/{max_attempts}...")
            with open(table_image_path, 'rb') as img_file:
                files = {'file': (os.path.basename(table_image_path), img_file, 'image/png')}
                response = session.post(
                    TABLE_SUMMARIZATION_API_ENDPOINT,
                    files=files,
                    timeout=timeout
                )
            if response.status_code == 200:
                response_json = response.json()
                summary = response_json.get("summary", "")
                if summary:
                    print(f"Table summary: {summary}")
                    return summary
                else:
                    print("Warning: Empty table summary returned")
                    print(f"Full API response: {response_json}")
            else:
                print(f"Error summarizing table (status code: {response.status_code})")
                print(f"Response: {response.text}")
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 3
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        except Exception as e:
            print(f"Exception during table summarization (attempt {attempt+1}): {str(e)}")
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 3
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    print("All table summarization attempts failed")
    return "Table description unavailable"

def summarize_figure(figure_image_path, max_attempts=3, timeout=(15, 4000)):
    """
    Summarize figure content using Azure OpenAI API
    
    Args:
        figure_image_path (str): Path to the figure image
        max_attempts (int): Maximum number of attempts
        timeout (tuple): Connection and read timeout in seconds
        
    Returns:
        str: Generated figure description
    """
    # First import the OpenAI client
    try:
        from openai import AzureOpenAI
    except ImportError:
        print("Error: OpenAI library not installed. Please run 'pip install openai'")
        return "Figure description unavailable (OpenAI library not installed)"
    
    # Check if image exists and is accessible
    if not os.path.exists(figure_image_path):
        print(f"Error: Figure image file not found: {figure_image_path}")
        return "Figure description unavailable"
        
    # Check file size
    try:
        file_size = os.path.getsize(figure_image_path)
        if file_size == 0:
            print(f"Error: Figure image file is empty: {figure_image_path}")
            return "Figure description unavailable"
            
        # If the image is very large, attempt to resize it
        if file_size > 2 * 1024 * 1024:  # 2MB (Azure has stricter limits than the table API)
            try:
                print(f"Large figure image detected ({file_size/1024/1024:.2f} MB), attempting to resize")
                img = Image.open(figure_image_path)
                
                # Resize while maintaining aspect ratio
                max_dimension = 800  # Smaller than tables since Azure has limits
                width, height = img.size
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Save resized image to a new file
                resized_path = figure_image_path.replace('.png', '_resized.png')
                img.save(resized_path)
                print(f"Resized image saved to {resized_path}")
                figure_image_path = resized_path
            except Exception as e:
                print(f"Error resizing image: {str(e)}")
    except Exception as e:
        print(f"Error checking image file: {str(e)}")
    
    # Extract Azure endpoint and key from configuration
    endpoint = "https://azureopenairag281.openai.azure.com/"
    model_name = "gpt-4o-mini"
    deployment = "gpt-4o-mini"
    api_key = AZURE_OPENAI_API_KEY
    api_version = "2024-12-01-preview"  # This may need to be updated as Azure updates their API
    
    for attempt in range(max_attempts):
        try:
            print(f"Figure summarization attempt {attempt+1}/{max_attempts}...")
            
            # Create Azure OpenAI client
            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=api_key,
            )
            
            # Read and encode image
            with open(figure_image_path, 'rb') as img_file:
                image_data = img_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create the messages array with system and user messages
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a scientific figure analysis expert. Describe the figure in detail, explaining its purpose, key findings, and relevance to scientific research."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please analyze and summarize this scientific figure:"
                            },
                            {
                                "type": "image_url", 
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.7,
                model=deployment
            )
            
            # Extract response content
            figure_description = response.choices[0].message.content
            if figure_description:
                print(f"Figure description: {figure_description}")
                return figure_description
            else:
                print("Warning: Empty figure description returned")
                
        except Exception as e:
            print(f"Exception during figure summarization (attempt {attempt+1}): {str(e)}")
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 3
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    
    print("All figure summarization attempts failed")
    return "Figure description unavailable"

def batch_summarize_figures(figure_image_paths, max_attempts=3, timeout=(15, 4000)):
    """
    Summarize multiple figures at once using the caption service
    
    Args:
        figure_image_paths (dict): Dictionary mapping figure IDs to image paths
        max_attempts (int): Maximum number of attempts
        timeout (tuple): Connection and read timeout in seconds
        
    Returns:
        dict: Dictionary mapping figure IDs to their generated captions
    """
    print(f"Starting batch figure summarization for {len(figure_image_paths)} images...")
    session = create_retry_session(retries=2)
    
    # Check if we have any images to process
    if not figure_image_paths:
        print("No figure images to summarize")
        return {}
    
    # Initialize result dictionary with default values
    result_captions = {fig_id: "Figure description unavailable" for fig_id in figure_image_paths}
    
    for attempt in range(max_attempts):
        try:
            print(f"Batch figure summarization attempt {attempt+1}/{max_attempts}...")
            
            # Prepare the files for multipart/form-data request
            files = []
            file_id_map = {}  # Map filenames to our internal figure IDs
            
            for fig_id, img_path in figure_image_paths.items():
                if not os.path.exists(img_path):
                    print(f"Warning: Figure image file not found: {img_path}")
                    continue
                    
                try:                    # Ensure image is in a supported format
                    img = Image.open(img_path)
                    filename = os.path.basename(img_path)
                    file_id_map[filename] = fig_id
                    
                    # Open file as binary
                    file_content = open(img_path, 'rb')
                    files.append(('files', (filename, file_content, f'image/{img.format.lower()}')))
                except Exception as e:
                    print(f"Error preparing figure {fig_id}: {str(e)}")
            
            if not files:
                print("No valid figure files to process")
                return result_captions
                  # Send the request with multiple images
            try:
                response = session.post(
                    FIGURE_CAPTION_BATCH_API_ENDPOINT,
                    files=files,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    response_json = response.json()
                    captions = response_json.get("captions", {})
                    
                    # Map the captions back to our figure IDs
                    for filename, caption in captions.items():
                        fig_id = file_id_map.get(filename)
                        if fig_id:
                            result_captions[fig_id] = caption
                            print(f"Figure {fig_id} caption: {caption}")
                    
                    return result_captions
                else:
                    print(f"Error summarizing figures (status code: {response.status_code})")
                    print(f"Response: {response.text}")
                    
                    # If this is not the last attempt, wait before retrying
                    if attempt < max_attempts - 1:
                        wait_time = (attempt + 1) * 3
                        print(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
            finally:
                # Clean up file handles
                for _, file_tuple in files:
                    try:
                        file_tuple[1].close()
                    except:
                        pass
        
        except Exception as e:
            print(f"Exception during batch figure summarization (attempt {attempt+1}): {str(e)}")
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 3
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    
    print("All batch figure summarization attempts failed")
    return result_captions

def check_api_health():
    """
    Check the health status of all API services
    
    Returns:
        dict: Health status of each API service
    """
    print("Checking API health status...")
    session = create_retry_session(retries=1)
    health_status = {
        "pdf_extractor": False,
        "classification": False,
        "text_summarization": False,
        "table_summarization": False,
        "figure_summarization": False
    }
    
    # Check PDF Extractor API
    try:
        # Check root endpoint instead of health
        response = session.get("/".join(PDF_EXTRACTOR_API_ENDPOINT.split("/")[:-2]) + "/", timeout=5)
        health_status["pdf_extractor"] = response.status_code < 500  # Any response that's not a server error
        print(f"PDF Extractor API: {'Available' if health_status['pdf_extractor'] else 'Unavailable'}")
    except Exception as e:
        print(f"PDF Extractor API check failed: {str(e)}")
    
    # Check Classification API
    try:
        # Check root endpoint instead of health
        response = session.get("/".join(PAPER_CLASSIFICATION_API_ENDPOINT.split("/")[:-1]), timeout=5)
        health_status["classification"] = response.status_code < 500  # Any response that's not a server error
        print(f"Classification API: {'Available' if health_status['classification'] else 'Unavailable'}")
    except Exception as e:
        print(f"Classification API check failed: {str(e)}")
    
    # Check Text Summarization API
    try:
        response = session.get(TEXT_SUMMARIZATION_API_ENDPOINT.replace("summarize", "health"), timeout=5)
        health_status["text_summarization"] = response.status_code == 200
        print(f"Text Summarization API: {'Available' if health_status['text_summarization'] else 'Unavailable'}")
    except Exception as e:
        print(f"Text Summarization API check failed: {str(e)}")
    
    # Check Table Summarization API
    try:
        response = session.get(TABLE_SUMMARIZATION_API_ENDPOINT.replace("summarize_table/", "health"), timeout=5)
        health_status["table_summarization"] = response.status_code == 200
        print(f"Table Summarization API: {'Available' if health_status['table_summarization'] else 'Unavailable'}")
    except Exception as e:
        print(f"Table Summarization API check failed: {str(e)}")
    
    # Check Figure Summarization API (Azure OpenAI)
    try:
        # Try to import OpenAI library
        try:
            from openai import AzureOpenAI
            
            # Check if we have credentials
            if AZURE_OPENAI_API_KEY and "azureopenairag281.openai.azure.com" in str(AZURE_OPENAI_API_ENDPOINT):
                # Try to create a client - this will fail if library is missing or credentials are invalid
                client = AzureOpenAI(
                    api_version="2024-12-01-preview",
                    azure_endpoint="https://azureopenairag281.openai.azure.com/",
                    api_key=AZURE_OPENAI_API_KEY,
                )
                
                # Just assume service is up if we can create client
                health_status["figure_summarization"] = True
                print(f"Figure Summarization API: Credentials available (assuming service is up)")
            else:
                health_status["figure_summarization"] = False
                print(f"Figure Summarization API: Missing or invalid credentials")
        except ImportError:
            print("OpenAI library not installed. Please run 'pip install openai'")
            health_status["figure_summarization"] = False
    except Exception as e:
        print(f"Error checking Figure Summarization API: {str(e)}")
        health_status["figure_summarization"] = False
    except Exception as e:
        print(f"Figure Summarization API check failed: {str(e)}")
    
    return health_status

def log_response_structure(response_data, log_file="data/api_responses.log"):
    """
    Log the structure of an API response for debugging
    
    Args:
        response_data (dict): The API response data
        log_file (str): Path to the log file
    """
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, "a") as f:
            f.write(f"\n\n--- API Response Log: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            
            # Print basic info about the response structure
            f.write(f"Response Type: {type(response_data)}\n")
            
            if isinstance(response_data, dict):
                f.write("Top-level keys:\n")
                for key in response_data.keys():
                    value = response_data[key]
                    value_type = type(value).__name__
                    
                    # Summarize the value based on type
                    if isinstance(value, dict):
                        summary = f"dict with {len(value)} keys"
                    elif isinstance(value, list):
                        summary = f"list with {len(value)} items"
                        if value and isinstance(value[0], dict):
                            sample_keys = list(value[0].keys())
                            if len(sample_keys) <= 5:
                                summary += f", sample keys: {sample_keys}"
                            else:
                                summary += f", sample keys: {sample_keys[:5]}..."
                    elif isinstance(value, str):
                        summary = f"string ({min(50, len(value))} chars): {value[:50]}"
                        if len(value) > 50:
                            summary += "..."
                    else:
                        summary = str(value)
                        
                    f.write(f"  - {key} ({value_type}): {summary}\n")
                    
                # More detailed logging for nested structures
                if "data" in response_data and isinstance(response_data["data"], dict):
                    f.write("\nData structure:\n")
                    data = response_data["data"]
                    for key in data.keys():
                        value = data[key]
                        value_type = type(value).__name__
                        
                        if isinstance(value, list):
                            f.write(f"  - {key} ({value_type}): list with {len(value)} items\n")
                            if value and isinstance(value[0], dict):
                                sample_keys = list(value[0].keys())
                                f.write(f"    Sample item keys: {sample_keys}\n")
                        else:
                            summary = str(value)
                            if isinstance(value, str) and len(summary) > 50:
                                summary = summary[:50] + "..."
                            f.write(f"  - {key} ({value_type}): {summary}\n")
    except Exception as e:
        print(f"Error logging response structure: {str(e)}")

def ensure_png_format(image_path):
    """
    Ensures that an image is in PNG format. If not, converts it to PNG.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Path to a PNG version of the image
    """
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return image_path
            
        # Check if the image already has PNG extension
        if image_path.lower().endswith('.png'):
            try:
                # Verify it's actually a valid PNG by opening it
                img = Image.open(image_path)
                if img.format == 'PNG':
                    print(f"Image is already in PNG format: {image_path}")
                    return image_path
            except Exception as e:
                print(f"Warning: File has PNG extension but could not be verified: {str(e)}")
        
        # Convert the image to PNG format
        try:
            img = Image.open(image_path)
            # Check image mode and convert if necessary
            if img.mode not in ['RGB', 'RGBA']:
                print(f"Converting image mode from {img.mode} to RGB")
                img = img.convert('RGB')
                
            png_path = os.path.splitext(image_path)[0] + '_converted.png'
            img.save(png_path, format='PNG')
            
            # Verify the saved PNG is valid
            try:
                verify_img = Image.open(png_path)
                verify_img.verify()
                print(f"Successfully converted image to PNG: {png_path}")
                return png_path
            except Exception as verify_err:
                print(f"Warning: Converted PNG verification failed: {str(verify_err)}")
                # If verification fails, try one more conversion with different parameters
                try:
                    img = Image.open(image_path).convert('RGB')
                    final_png_path = os.path.splitext(image_path)[0] + '_final.png'
                    img.save(final_png_path, format='PNG', optimize=True)
                    print(f"Second conversion attempt successful: {final_png_path}")
                    return final_png_path
                except Exception as final_err:
                    print(f"Error in final conversion attempt: {str(final_err)}")
                    return image_path
        except Exception as e:
            print(f"Error converting image to PNG: {str(e)}")
            return image_path  # Return original if conversion fails
    except Exception as e:
        print(f"Error in ensure_png_format: {str(e)}")
        return image_path  # Return original path on any error

def generate_response_from_chunks(query, chunks, max_attempts=3, timeout=(30, 4000)):
    """
    Generate a response from GPT API using the relevant chunks
    
    Args:
        query (str): The user query
        chunks (list): List of relevant chunks (dicts with content and metadata)
        max_attempts (int): Maximum number of attempts
        timeout (tuple): Connection and read timeout in seconds
        
    Returns:
        str: Generated response from GPT
    """
    try:
        from openai import AzureOpenAI
    except ImportError:
        print("Error: OpenAI library not installed. Please run 'pip install openai'")
        return "Unable to generate response: OpenAI library not installed."
    
    # Extract Azure endpoint and key from configuration
    endpoint = "https://azureopenairag281.openai.azure.com/"
    model_name = "gpt-4o-mini"
    deployment = "gpt-4o-mini"
    api_key = AZURE_OPENAI_API_KEY
    api_version = "2024-12-01-preview"
    
    for attempt in range(max_attempts):
        try:
            print(f"GPT response generation attempt {attempt+1}/{max_attempts}...")
            
            # Create Azure OpenAI client
            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=api_key,
            )
            
            # Format the context from chunks
            context = ""
            for chunk in chunks:
                source_type = chunk["metadata"].get("source", "unknown")
                if source_type == "text":
                    section_title = chunk["metadata"].get("section_title", "Section")
                    context += f"[TEXT - {section_title}]\n{chunk['content']}\n\n"
                elif source_type == "table":
                    table_caption = chunk["metadata"].get("table_caption", "Table")
                    context += f"[TABLE - {table_caption}]\n{chunk['content']}\n\n"
                elif source_type == "figure":
                    figure_caption = chunk["metadata"].get("figure_caption", "Figure")
                    context += f"[FIGURE - {figure_caption}]\n{chunk['content']}\n\n"
                else:
                    context += f"{chunk['content']}\n\n"
            
            # Create the messages array with system and user messages
            messages = [
                {
                    "role": "system",
                    "content": "You are a scientific research assistant. Answer the user's question based on the provided context from scientific papers. If the context doesn't contain enough information to answer the question, say so clearly. Do not make up information that isn't supported by the context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nPlease provide a comprehensive answer based on the context."
                }
            ]
            
            # Call the OpenAI API
            response = client.chat.completions.create(
                messages=messages,
                max_tokens=800,
                temperature=0.3,
                model=deployment
            )
            
            # Extract and return the response
            ai_response = response.choices[0].message.content
            if ai_response:
                return ai_response
            else:
                print("Warning: Empty response from GPT API")
                if attempt == max_attempts - 1:
                    return "Unable to generate a response. Please try again or refine your query."
                
        except Exception as e:
            print(f"Error generating GPT response (attempt {attempt+1}): {str(e)}")
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 2
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    
    return "Failed to generate a response after multiple attempts. Please try again later."

def generate_suggested_questions(query, chunks, generated_response, max_attempts=3, timeout=(30, 4000)):
    """
    Generate suggested follow-up questions based on the user's query, retrieved chunks, and generated response
    
    Args:
        query (str): The original user query
        chunks (list): List of relevant chunks (dicts with content and metadata)
        generated_response (str): The generated response from the previous query
        max_attempts (int): Maximum number of attempts
        timeout (tuple): Connection and read timeout in seconds
        
    Returns:
        list: List of suggested follow-up questions
    """
    try:
        from openai import AzureOpenAI
    except ImportError:
        print("Error: OpenAI library not installed. Please run 'pip install openai'")
        return ["Unable to generate suggested questions: OpenAI library not installed."]
    
    # Extract Azure endpoint and key from configuration
    endpoint = "https://azureopenairag281.openai.azure.com/"
    model_name = "gpt-4o-mini"
    deployment = "gpt-4o-mini"
    api_key = AZURE_OPENAI_API_KEY
    api_version = "2024-12-01-preview"
    
    for attempt in range(max_attempts):
        try:
            print(f"Generating suggested questions attempt {attempt+1}/{max_attempts}...")
            
            # Create Azure OpenAI client
            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=api_key,
            )
            
            # Format the context from chunks (keeping it brief for this task)
            context_summary = ""
            for i, chunk in enumerate(chunks[:3]):  # Use just the top 3 chunks to keep it concise
                source_type = chunk["metadata"].get("source", "unknown")
                if source_type == "text":
                    section_title = chunk["metadata"].get("section_title", "Section")
                    context_summary += f"[TEXT - {section_title}]\n{chunk['content'][:200]}...\n\n"
                elif source_type == "table":
                    table_caption = chunk["metadata"].get("table_caption", "Table")
                    context_summary += f"[TABLE - {table_caption}]\n{chunk['content'][:200]}...\n\n"
                elif source_type == "figure":
                    figure_caption = chunk["metadata"].get("figure_caption", "Figure")
                    context_summary += f"[FIGURE - {figure_caption}]\n{chunk['content'][:200]}...\n\n"
                else:
                    context_summary += f"{chunk['content'][:200]}...\n\n"
                
                if i >= 2:  # Limit to first 3 chunks
                    break
            
            # Create the messages array with system and user messages
            messages = [
                {
                    "role": "system",
                    "content": "You are a scientific research assistant. Generate 3-5 relevant follow-up questions that a user might want to ask next based on their original query, the context from scientific papers, and the response they received. Make the questions specific, insightful, and designed to deepen understanding of the topic. Format your response as a numbered list of questions only, without any explanations or additional text."
                },
                {
                    "role": "user",
                    "content": f"Original query: {query}\n\nContext summary:\n{context_summary}\n\nGenerated response: {generated_response}\n\nBased on this interaction, generate 3-5 specific follow-up questions the user might want to ask next."
                }
            ]
            
            # Call the OpenAI API
            response = client.chat.completions.create(
                messages=messages,
                max_tokens=400,
                temperature=0.7,  # Higher temperature for more creative questions
                model=deployment
            )
            
            # Extract and process the response
            ai_response = response.choices[0].message.content
            if ai_response:
                # Parse the numbered list into a Python list
                questions = []
                for line in ai_response.strip().split('\n'):
                    # Remove leading numbers, dots, parentheses, etc.
                    cleaned_line = line.strip()
                    # Check if line starts with a number (potentially followed by a separator)
                    if cleaned_line and (cleaned_line[0].isdigit() or cleaned_line.startswith('- ')):
                        # Remove the number/bullet and any separators
                        for i, char in enumerate(cleaned_line):
                            if char.isalpha():
                                question = cleaned_line[i:].strip()
                                questions.append(question)
                                break
                    else:
                        # If it doesn't start with a number but looks like a question
                        if '?' in cleaned_line:
                            questions.append(cleaned_line)
                
                # If we didn't successfully parse any questions, return the raw response
                if not questions and ai_response:
                    # Just split by newlines as a fallback
                    questions = [q for q in ai_response.strip().split('\n') if q.strip()]
                
                return questions
            else:
                print("Warning: Empty response from GPT API")
                if attempt == max_attempts - 1:
                    return ["No suggested questions available. Please try again."]
                
        except Exception as e:
            print(f"Error generating suggested questions (attempt {attempt+1}): {str(e)}")
            if attempt < max_attempts - 1:
                wait_time = (attempt + 1) * 2
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    
    return ["Failed to generate suggested questions. Please try again later."]
