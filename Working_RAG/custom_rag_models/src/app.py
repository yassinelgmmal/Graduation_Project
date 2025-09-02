from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
import shutil
import datetime
from tempfile import NamedTemporaryFile
from src.paper_processor import ScientificPaperProcessor
from typing import Optional

# Initialize the FastAPI application
app = FastAPI(title="Scientific Paper Multimodal Summarization API",
              description="API for extracting and summarizing multimodal content from scientific papers")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the paper processor
paper_processor = ScientificPaperProcessor()

# Mount static files directory
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/process-paper/")
async def process_paper(file: UploadFile = File(...)):
    """
    Process a scientific paper PDF and generate multimodal summaries
    
    Args:
        file (UploadFile): The PDF file to process
        
    Returns:
        dict: Paper processing results
    """
    temp_file_path = None
    
    try:
        # Check file type
        filename = file.filename
        if not filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
            
        # Check file size
        contents = await file.read()
        size_mb = len(contents) / (1024 * 1024)
        
        if size_mb > 50:  # Limit to 50MB
            raise HTTPException(status_code=400, detail=f"File size ({size_mb:.1f} MB) exceeds the limit of 50 MB")
            
        if size_mb < 0.01:  # 10KB
            raise HTTPException(status_code=400, detail="File is too small or empty")
            
        # Create temporary file
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name
            
        print(f"Saved PDF to temporary file: {temp_file_path}")
        print(f"File size: {size_mb:.2f} MB")
        
        # Check if external APIs are available
        from src.api_utils import check_api_health
        api_status = check_api_health()
        
        # Proceed only if PDF extraction API is available
        if not api_status["pdf_extractor"]:
            print("PDF extraction service is unavailable")
            raise HTTPException(
                status_code=503, 
                detail="PDF extraction service is currently unavailable. Please try again later."
            )
        
        # Process the paper
        paper_data = paper_processor.process_pdf(temp_file_path)
        
        # Cleanup temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"Removed temporary file: {temp_file_path}")
            except Exception as e:
                print(f"Warning: Could not remove temporary file: {str(e)}")
        
        if not paper_data:
            raise HTTPException(
                status_code=500, 
                detail="Failed to process paper. The extraction service may be experiencing issues."
            )
        
        # Generate multimodal summary
        try:
            summary = paper_processor.generate_multimodal_summary(paper_data)
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            # Return partial results if summary generation fails
            return {
                "paper_id": paper_data["paper_id"],
                "title": paper_data["title"],
                "status": "partial",
                "error": f"Summary generation incomplete: {str(e)}",
                "paper_data": paper_data
            }
        
        # Return paper ID and summary
        return {
            "paper_id": paper_data["paper_id"],
            "title": paper_data["title"],
            "status": "complete",
            "summary": summary
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error processing paper: {str(e)}")
        
        # Ensure temporary file is cleaned up
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"Removed temporary file after error: {temp_file_path}")
            except:
                pass
                
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing paper: {str(e)}. Please check if the PDF is valid and try again."
        )

@app.get("/papers/{paper_id}")
async def get_paper(paper_id: str):
    """
    Get processed paper data by ID
    
    Args:
        paper_id (str): The paper identifier
        
    Returns:
        dict: Paper data
    """
    try:
        paper_file_path = f"data/papers/{paper_id}.json"
        
        if not os.path.exists(paper_file_path):
            raise HTTPException(status_code=404, detail="Paper not found")
        
        with open(paper_file_path, 'r') as f:
            paper_data = json.load(f)
            
        return paper_data
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving paper: {str(e)}")

@app.post("/query/")
async def query_rag(paper_id: Optional[str] = Form(None), query: str = Form(...)):
    """
    Query the RAG system about a specific paper or across all papers and get a generated response
    
    Args:
        paper_id (str, optional): Paper ID to query specifically
        query (str): The query text
        
    Returns:
        dict: Query results with relevant chunks and a generated response
    """
    try:
        # First make sure the vector DB is loaded with allow_dangerous_deserialization=True
        paper_processor.rag_manager.reload_vector_db()
        
        if paper_id:
            # Query specific paper
            results = paper_processor.query_paper(paper_id, query)
            if not results:
                raise HTTPException(status_code=404, detail=f"Paper with ID {paper_id} not found or could not be loaded")
        else:            # Query across all papers in the RAG system
            relevant_chunks = paper_processor.rag_manager.retrieve_relevant_chunks(query, k=10)
            results = {
                "query": query,
                "relevant_chunks": [],
                "papers": {}
            }
            
            # Process chunks and group by paper_id to include paper summaries
            for doc in relevant_chunks:
                doc_paper_id = doc.metadata.get("paper_id", "unknown")
                
                # Get paper summary if we haven't already
                if doc_paper_id not in results["papers"] and doc_paper_id != "unknown":
                    try:
                        with open(f"data/papers/{doc_paper_id}.json", 'r') as f:
                            paper_data = json.load(f)
                            results["papers"][doc_paper_id] = {
                                "title": paper_data.get("title", "Untitled"),
                                "summary": paper_data.get("summary", "No summary available")
                            }
                    except Exception:
                        results["papers"][doc_paper_id] = {
                            "title": "Unknown",
                            "summary": "Paper data not available"
                        }
                
                # Add chunk to results
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
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error querying RAG system: {str(e)}")
        error_message = str(e)
        # Make a more user-friendly error message for the deserialization issue
        if "allow_dangerous_deserialization" in error_message:
            error_message = "Database load error. The system is being updated to handle this issue."
        raise HTTPException(status_code=500, detail=f"Error querying RAG system: {error_message}")

@app.get("/papers/")
async def list_papers():
    """
    List all processed papers
    
    Returns:
        list: List of paper summaries
    """
    try:
        papers = []
        papers_dir = "data/papers"
        
        if os.path.exists(papers_dir):
            for filename in os.listdir(papers_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(papers_dir, filename), 'r') as f:
                        paper_data = json.load(f)
                        papers.append({
                            "paper_id": paper_data["paper_id"],
                            "title": paper_data["title"],
                            "authors": paper_data["authors"],
                            "summary": paper_data["summary"][:200] + "..." if len(paper_data["summary"]) > 200 else paper_data["summary"]
                        })
                        
        return papers
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing papers: {str(e)}")

@app.get("/health/")
async def health_check():
    """
    API health check endpoint
    
    Returns:
        dict: Health status of all services
    """
    from src.api_utils import check_api_health
    
    api_status = check_api_health()
    
    # Check vector database
    vector_db_status = "not_initialized"
    try:
        if os.path.exists("data/vector_db"):
            vector_db_status = "available"
    except:
        vector_db_status = "error"
    
    # Get disk space information
    disk_space = {}
    try:
        total, used, free = shutil.disk_usage("/")
        disk_space = {
            "total": f"{total // (2**30)} GB",
            "used": f"{used // (2**30)} GB",
            "free": f"{free // (2**30)} GB",
            "percent_used": f"{used * 100 / total:.1f}%"
        }
    except:
        disk_space = {"error": "Could not get disk information"}
    
    return {
        "status": "healthy",
        "api_services": api_status,
        "vector_db": vector_db_status,
        "disk_space": disk_space,
        "timestamp": str(datetime.datetime.now())
    }

@app.post("/suggest-questions/")
async def suggest_questions(paper_id: str = Form(...), query: str = Form(...)):
    """
    Generate suggested follow-up questions based on a previous query and response
    
    Args:
        paper_id (str): The paper identifier
        query (str): The original query text
        
    Returns:
        dict: List of suggested follow-up questions
    """
    try:
        # First make sure the vector DB is loaded
        paper_processor.rag_manager.reload_vector_db()
        
            
        # Retrieve relevant chunks for the query
        relevant_chunks = paper_processor.rag_manager.retrieve_relevant_chunks(query, k=5)
        chunks_data = []
        
        for doc in relevant_chunks:
            chunks_data.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
        # Generate response from chunks
        from src.api_utils import generate_response_from_chunks
        generated_response = generate_response_from_chunks(query, chunks_data)
        
        # Generate suggested questions
        from src.api_utils import generate_suggested_questions
        suggested_questions = generate_suggested_questions(query, chunks_data, generated_response)
        
        return {
            "paper_id": paper_id,
            "original_query": query,
            "suggested_questions": suggested_questions
        }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error generating suggested questions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating suggested questions: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8020, reload=True)
