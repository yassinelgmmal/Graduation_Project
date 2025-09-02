import os
import uuid
import json
import base64
from PIL import Image
from src.api_utils import (
    extract_from_pdf, 
    classify_paper, 
    summarize_text,
    summarize_table,
    summarize_figure
)
from src.rag_manager import RAGManager

class ScientificPaperProcessor:
    """
    Processor for scientific papers that extracts, analyzes and summarizes paper content
    using multimodal approaches and a RAG framework
    """
    
    def __init__(self, fault_tolerant=True):
        """
        Initialize the paper processor
        
        Args:
            fault_tolerant (bool): If True, continue processing even if some services fail
        """
        self.rag_manager = RAGManager()
        self.fault_tolerant = fault_tolerant
        os.makedirs("data/papers", exist_ok=True)
        os.makedirs("data/images/tables", exist_ok=True)
        os.makedirs("data/images/figures", exist_ok=True)
    
    def process_pdf(self, pdf_path):
        """
        Process a scientific PDF paper
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: Processed paper data with summaries
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Generate a unique ID for this paper
        paper_id = str(uuid.uuid4())
          # 1. Extract content from PDF
        print("Extracting content from PDF...")
        try:
            # Verify PDF file exists and can be read
            if not os.path.exists(pdf_path):
                print(f"Error: PDF file not found at path: {pdf_path}")
                return None
                
            file_size = os.path.getsize(pdf_path)
            print(f"PDF file size: {file_size/1024:.2f} KB")
            
            if file_size == 0:
                print("Error: PDF file is empty")
                return None
                
            # Attempt PDF extraction with increased timeout and retries
            extracted_data = extract_from_pdf(pdf_path, max_attempts=3, timeout=(60, 300))
            
            if not extracted_data:
                print("Failed to extract content from PDF")
                return None
                  # Basic validation of extracted data
            if (not extracted_data.get("sections") or len(extracted_data.get("sections", [])) == 0) and not extracted_data.get("abstract"):
                print("Warning: Extracted data has no sections or abstract")
                
                # Create a minimal section if none exist and we have full text
                if not extracted_data.get("sections"):
                    extracted_data["sections"] = []
                if extracted_data.get("full_text") and not extracted_data.get("sections"):
                    # If we have full text but no sections, create a single section with all the text
                    extracted_data["sections"].append({"heading": "Full Text", "text": extracted_data["full_text"]})
                
        except Exception as e:
            print(f"Critical error during PDF extraction: {str(e)}")
            return None
              # Initialize paper data
        paper_data = {
            "paper_id": paper_id,
            "title": extracted_data.get("title", "Untitled Paper"),
            "authors": extracted_data.get("authors", []),
            "abstract": extracted_data.get("abstract", ""),
            "full_text": extracted_data.get("full_text", ""),
            "text_sections": [],
            "tables": [],
            "figures": [],
            "classifications": [],
            "summary": ""
        }
        
        # Extract full text from the PDF
        full_text = ""
        if extracted_data.get("full_text"):
            # If we have the full_text field directly
            full_text = extracted_data["full_text"]
            print(f"Extracted full text: {len(full_text)} characters")
        elif extracted_data.get("sections"):
            # Concatenate all section texts if no full_text field
            full_text = " ".join([section.get("text", "") for section in extracted_data.get("sections", [])])
            print(f"Combined text from {len(extracted_data['sections'])} sections: {len(full_text)} characters")
        else:
            print("Warning: No full text or sections found in extracted data")
        
        # Store the full text in paper data
        paper_data["full_text"] = full_text
          # 2. Classify paper
        print("Classifying paper...")
        try:
            # Handle both old and new response formats
            text_for_classification = ""
            if extracted_data.get("abstract"):
                text_for_classification += extracted_data["abstract"] + " "
                
            if extracted_data.get("full_text"):
                # If we have the full_text field from new format
                text_for_classification += extracted_data["full_text"]
            else:
                # Otherwise, concatenate all section texts
                text_for_classification += " ".join([section.get("text", "") for section in extracted_data.get("sections", [])])
                
            classification_result = classify_paper(text_for_classification[:5000])  # Using first 5000 chars for classification
            
            if classification_result:
                paper_data["classifications"] = classification_result.get("predicted_label", [])
            else:
                print("Warning: Paper classification failed")
                paper_data["classifications"] = ["unclassified"]
        except Exception as e:
            print(f"Error during paper classification: {str(e)}")
            paper_data["classifications"] = ["unclassified"]
            if not self.fault_tolerant:
                raise
          # 3. Process text content
        print("Processing text content...")
        sections_processed = 0
        sections_failed = 0
        
        # Use the full text directly for chunking and storage in vector store
        if paper_data["full_text"]:
            print(f"Processing full text: {len(paper_data['full_text'])} characters")
            try:
                # Create a single section with the full text for summarization
                section_title = "Full Text"
                section_text = paper_data["full_text"]
                
                # Generate a summary for the full text
                section_summary = summarize_text(section_text[:5000])  # Use first 5000 chars for summary
                
                paper_data["text_sections"].append({
                    "title": section_title,
                    "text": section_text,
                    "summary": section_summary
                })
                
                sections_processed += 1
                print(f"Full text processed as a single section: {len(section_text)} characters")
            except Exception as e:
                sections_failed += 1
                print(f"Error processing full text: {str(e)}")
                
                # Add a minimal section despite the error
                paper_data["text_sections"].append({
                    "title": "Full Text",
                    "text": paper_data["full_text"],
                    "summary": "Summary unavailable"
                })
                
                if not self.fault_tolerant:
                    raise
        # Process sections only if we don't have full text or as a fallback
        elif extracted_data.get("sections") and len(extracted_data.get("sections", [])) > 0:
            # Process each section individually
            print(f"Processing {len(extracted_data['sections'])} text sections...")
            
            for section in extracted_data.get("sections", []):
                try:
                    section_text = section.get("text", "")
                    section_title = section.get("heading", "Untitled Section")
                    
                    if section_text:
                        summary = summarize_text(section_text)
                        paper_data["text_sections"].append({
                            "title": section_title,
                            "text": section_text,
                            "summary": summary
                        })
                        sections_processed += 1
                except Exception as e:
                    sections_failed += 1
                    print(f"Error processing text section '{section.get('heading', 'unknown')}': {str(e)}")
                    # Add the section with empty summary
                    if section.get("text"):
                        paper_data["text_sections"].append({
                            "title": section.get("heading", "Untitled Section"),
                            "text": section.get("text", ""),
                            "summary": "Summary unavailable"
                        })
                        
                    if not self.fault_tolerant:
                        raise
        else:
            print("Warning: No text content available to process")
            
        print(f"Text sections processed: {sections_processed}, failed: {sections_failed}")
          # 4. Process tables
        print("Processing tables...")
        tables_processed = 0
        tables_failed = 0
        
        for idx, table in enumerate(extracted_data.get("tables", [])):
            try:
                table_image_path = f"data/images/tables/{paper_id}_table_{idx}.png"
                  # Check if table has image data
                if table.get("image_data"):
                    # Old format with image_data already decoded
                    image_data = table.get("image_data")
                elif table.get("image"):
                    # New format might have image as base64 string
                    try:
                        # Handle both base64 string or already decoded bytes
                        if isinstance(table["image"], str):
                            try:
                                image_data = base64.b64decode(table["image"])
                            except Exception as e:
                                print(f"Error decoding table image base64: {str(e)}")
                                image_data = None
                        else:
                            # Already decoded bytes
                            image_data = table["image"]
                    except Exception as e:
                        print(f"Error processing table image: {str(e)}")
                        image_data = None
                else:
                    image_data = None
                  # Save table image if we have data
                if image_data:
                    try:
                        # Make sure the table_image_path ends with .png
                        if not table_image_path.lower().endswith('.png'):
                            table_image_path = os.path.splitext(table_image_path)[0] + '.png'
                        
                        with open(table_image_path, "wb") as f:
                            f.write(image_data)
                        
                        # Verify the image was saved properly and is valid
                        try:
                            img = Image.open(table_image_path)
                            img.verify()  # Verify it's a valid image file
                        except Exception as img_err:
                            print(f"Warning: Saved table image may be invalid: {str(img_err)}")
                            # Try to convert the image data to PNG using PIL
                            try:
                                from io import BytesIO
                                img = Image.open(BytesIO(image_data))
                                img.save(table_image_path, format='PNG')
                                print(f"Successfully converted and saved table image to {table_image_path}")
                            except Exception as conv_err:
                                print(f"Error converting table image: {str(conv_err)}")
                        
                        table_caption = table.get("caption", f"Table {idx+1}")
                        table_summary = summarize_table(table_image_path)
                    except Exception as save_err:
                        print(f"Error saving table image: {str(save_err)}")
                        table_caption = table.get("caption", f"Table {idx+1}")
                        table_summary = "Table summary unavailable due to image processing error"
                    
                    paper_data["tables"].append({
                        "id": f"table_{idx}",
                        "caption": table_caption,
                        "image_path": table_image_path,
                        "summary": table_summary
                    })
                    tables_processed += 1
                else:
                    print(f"Warning: Table {idx} has no image data")
                    tables_failed += 1
            except Exception as e:
                tables_failed += 1
                print(f"Error processing table {idx}: {str(e)}")
                
                # Add basic table info despite failure
                if table.get("caption"):
                    paper_data["tables"].append({
                        "id": f"table_{idx}",
                        "caption": table.get("caption", f"Table {idx+1}"),
                        "image_path": "",
                        "summary": "Table summary unavailable"
                    })
                
                if not self.fault_tolerant:
                    raise
        
        print(f"Tables processed: {tables_processed}, failed: {tables_failed}")        # 5. Process figures
        print("Processing figures...")
        figures_processed = 0
        figures_failed = 0
        figure_images = {}  # Dictionary to collect figure paths for batch processing
        
        # First pass: Extract and save all figure images
        for idx, figure in enumerate(extracted_data.get("figures", [])):
            try:
                figure_image_path = f"data/images/figures/{paper_id}_figure_{idx}.png"
                figure_id = f"figure_{idx}"
                
                # Check if figure has image data
                if figure.get("image_data"):
                    # Old format with image_data already decoded
                    image_data = figure.get("image_data")
                elif figure.get("image"):
                    # New format might have image as base64 string
                    try:
                        # Handle both base64 string or already decoded bytes
                        if isinstance(figure["image"], str):
                            try:
                                image_data = base64.b64decode(figure["image"])
                            except Exception as e:
                                print(f"Error decoding figure image base64: {str(e)}")
                                image_data = None
                        else:
                            # Already decoded bytes
                            image_data = figure["image"]
                    except Exception as e:
                        print(f"Error processing figure image: {str(e)}")
                        image_data = None
                else:
                    image_data = None
                
                # Save figure image if we have data
                if image_data:
                    try:
                        # Make sure the figure_image_path ends with .png
                        if not figure_image_path.lower().endswith('.png'):
                            figure_image_path = os.path.splitext(figure_image_path)[0] + '.png'
                        
                        with open(figure_image_path, "wb") as f:
                            f.write(image_data)
                        
                        # Verify the image was saved properly and is valid
                        try:
                            img = Image.open(figure_image_path)
                            img.verify()  # Verify it's a valid image file
                        except Exception as img_err:
                            print(f"Warning: Saved figure image may be invalid: {str(img_err)}")
                            # Try to convert the image data to PNG using PIL
                            try:
                                from io import BytesIO
                                img = Image.open(BytesIO(image_data))
                                img.save(figure_image_path, format='PNG')
                                print(f"Successfully converted and saved figure image to {figure_image_path}")
                            except Exception as conv_err:
                                print(f"Error converting figure image: {str(conv_err)}")
                        
                        figure_caption = figure.get("caption", f"Figure {idx+1}")
                        
                        # Add to the batch processing dictionary
                        figure_images[figure_id] = figure_image_path
                        
                        # Store basic info in paper_data without the summary for now
                        paper_data["figures"].append({
                            "id": figure_id,
                            "caption": figure_caption,
                            "image_path": figure_image_path,
                            "summary": ""  # Will be filled in after batch processing
                        })
                        figures_processed += 1
                    except Exception as save_err:
                        print(f"Error saving figure image: {str(save_err)}")
                        figure_caption = figure.get("caption", f"Figure {idx+1}")
                        
                        paper_data["figures"].append({
                            "id": figure_id,
                            "caption": figure_caption,
                            "image_path": "",
                            "summary": "Figure description unavailable due to image processing error"
                        })
                else:
                    print(f"Warning: Figure {idx} has no image data")
                    figures_failed += 1
            except Exception as e:
                figures_failed += 1
                print(f"Error processing figure {idx}: {str(e)}")
                
                # Add basic figure info despite failure
                if figure.get("caption"):
                    paper_data["figures"].append({
                        "id": f"figure_{idx}",
                        "caption": figure.get("caption", f"Figure {idx+1}"),
                        "image_path": "",
                        "summary": "Figure description unavailable"
                    })
                
                if not self.fault_tolerant:
                    raise
        
        # Batch process all the figures together
        if figure_images:
            print(f"Batch processing {len(figure_images)} figures...")
            from src.api_utils import batch_summarize_figures
            figure_summaries = batch_summarize_figures(figure_images)
            
            # Update the figures in paper_data with their summaries
            for figure in paper_data["figures"]:
                figure_id = figure["id"]
                if figure_id in figure_summaries:
                    figure["summary"] = figure_summaries[figure_id]
        
        print(f"Figures processed: {figures_processed}, failed: {figures_failed}")
          # 6. Generate overall paper summary using PEGASUS        print("Generating overall summary...")
        try:
            abstract = paper_data.get("abstract", "")
            full_text = ""
            
            # Collect all text sections to create a comprehensive summary
            for section in paper_data["text_sections"]:
                full_text += section["text"] + " "
            
            # Use the abstract if available, otherwise the full text
            if abstract.strip():
                # Combine abstract with full text if both are available
                summary_text = abstract
                if full_text.strip():
                    # Add a selection of the full text if the abstract is too short
                    if len(abstract) < 500 and len(full_text) > 0:
                        summary_text += " " + full_text[:5000]  # Use first 5000 chars of full text
            else:
                # If no abstract, use the full text
                summary_text = full_text[:8000]  # Limit to 8000 chars to avoid overwhelming the summarization service
                
            # Generate summary
            if summary_text.strip():
                print(f"Sending {len(summary_text)} characters to summarization service...")
                paper_data["summary"] = summarize_text(summary_text, max_length=800, min_length=150)
            else:
                # If no text is available, create a basic summary
                print("Warning: No text available for overall summary")
                paper_data["summary"] = f"This paper titled '{paper_data['title']}' was processed with limited content available."
        except Exception as e:
            print(f"Error generating overall summary: {str(e)}")
            # Create a fallback summary
            paper_data["summary"] = f"This paper titled '{paper_data['title']}' contains {len(paper_data['text_sections'])} text sections, {len(paper_data['tables'])} tables, and {len(paper_data['figures'])} figures."
            
            if not self.fault_tolerant:
                raise
        
        # 7. Save processed data
        print("Saving paper data...")
        try:
            self.rag_manager.save_paper_data(paper_data, paper_id)
        except Exception as e:
            print(f"Error saving paper data: {str(e)}")
            if not self.fault_tolerant:
                raise
        
        # 8. Ingest into RAG
        print("Ingesting into RAG system...")
        try:
            self.rag_manager.ingest_paper(paper_data)
        except Exception as e:
            print(f"Error ingesting into RAG system: {str(e)}")
            if not self.fault_tolerant:
                raise
        
        return paper_data
    
    def generate_multimodal_summary(self, paper_data, query=None):
        """
        Generate a multimodal summary from paper data
        
        Args:
            paper_data (dict): Processed paper data
            query (str, optional): Specific aspect to focus on
            
        Returns:
            dict: Multimodal summary
        """
        # If there's a specific query, retrieve relevant chunks
        relevant_chunks = []
        if query:
            print(f"Retrieving relevant chunks for query: {query}")
            relevant_chunks = self.rag_manager.retrieve_relevant_chunks(query, k=5)
        
        # Create multimodal summary
        summary = {
            "paper_id": paper_data["paper_id"],
            "title": paper_data["title"],
            "authors": paper_data["authors"],
            "overall_summary": paper_data["summary"],
            "classifications": paper_data["classifications"],
            "text_sections": [],
            "tables": [],
            "figures": [],
            "relevant_chunks": []
        }
        
        # Include text section summaries
        for section in paper_data["text_sections"]:
            summary["text_sections"].append({
                "title": section["title"],
                "summary": section["summary"]
            })
        
        # Include table summaries
        for table in paper_data["tables"]:
            summary["tables"].append({
                "caption": table["caption"],
                "summary": table["summary"],
                "image_path": table["image_path"]
            })
        
        # Include figure summaries
        for figure in paper_data["figures"]:
            summary["figures"].append({
                "caption": figure["caption"],
                "description": figure["summary"],
                "image_path": figure["image_path"]
            })
        
        # Add relevant chunks if query was provided
        if query:
            for doc in relevant_chunks:
                summary["relevant_chunks"].append({
                    "content": doc.page_content,
                    "metadata": doc.metadata                })
        
        return summary
        
    def query_paper(self, paper_id, query):
        """
        Query a specific paper using the RAG system
        
        Args:
            paper_id (str): Paper identifier
            query (str): Query text
            
        Returns:
            dict: Query results with relevant chunks and generated response
        """
        # Load paper data
        try:
            with open(f"data/papers/{paper_id}.json", 'r') as f:
                paper_data = json.load(f)
        except Exception as e:
            print(f"Error loading paper data: {str(e)}")
            return None
        
        # Retrieve relevant chunks
        relevant_chunks = self.rag_manager.retrieve_relevant_chunks(query, k=10)
        
        # Prepare response
        results = {
            "paper_id": paper_id,
            "title": paper_data["title"],
            "summary": paper_data.get("summary", "No summary available"),
            "query": query,
            "relevant_chunks": []
        }
        
        # Add relevant chunks
        for doc in relevant_chunks:
            results["relevant_chunks"].append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        # Generate response using GPT API
        try:
            from src.api_utils import generate_response_from_chunks
            gpt_response = generate_response_from_chunks(query, results["relevant_chunks"])
            results["generated_response"] = gpt_response
        except Exception as e:
            print(f"Error generating response from chunks: {str(e)}")
            results["generated_response"] = "Failed to generate response from the retrieved content."
        
        return results
