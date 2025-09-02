from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import uuid
from pathlib import Path
from typing import Optional, List

from app.services.ingest import ingest_document
from app.services.answering import answer_query

router = APIRouter(prefix="/qa", tags=["QuestionAnswering"])

class QARequest(BaseModel):
    query: str
    top_k: int = 5
    document_id: Optional[str] = None  # Optional filter by document

class IngestResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks_processed: int

class QAResponse(BaseModel):
    answer: str
    sources: List[dict]
    query: str

@router.post("/ingest", response_model=IngestResponse)
async def ingest_paper(file: UploadFile = File(...)):
    """
    Upload a PDF scientific paper and ingest into vector store.
    """
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
        # Validate file size (limit to 50MB)
        if file.size and file.size > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size too large. Maximum 50MB allowed.")
        
        # Create unique filename
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        
        # Save to temp directory
        temp_dir = Path("./uploads")
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / unique_filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest document
        doc_id, chunks_count = await ingest_document(str(file_path))
        
        # Clean up temp file
        file_path.unlink()
        
        return IngestResponse(
            document_id=doc_id,
            filename=file.filename,
            status="success",
            chunks_processed=chunks_count
        )
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.post("/ask", response_model=QAResponse)
async def ask_question(request: QARequest):
    """
    Ask a question over ingested documents.
    """
    try:
        answer, sources = await answer_query(
            request.query, 
            top_k=request.top_k,
            document_id=request.document_id
        )
        
        return QAResponse(
            answer=answer,
            sources=sources,
            query=request.query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

@router.post("/follow-up", response_model=dict)
async def get_follow_up_questions(request: QARequest):
    """
    Get follow-up questions based on a query and its answer.
    """
    try:
        from app.services.answering import generate_follow_up_questions
        
        # First get the answer
        answer, sources = await answer_query(
            request.query, 
            top_k=request.top_k,
            document_id=request.document_id
        )
        
        # Generate follow-up questions
        follow_ups = await generate_follow_up_questions(request.query, answer, sources)
        
        return {
            "original_query": request.query,
            "answer": answer,
            "follow_up_questions": follow_ups,
            "sources_count": len(sources)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating follow-up questions: {str(e)}")

@router.get("/methodology/{document_id}")
async def explain_methodology(document_id: str, aspect: str = "general"):
    """
    Explain the methodology used in a specific document.
    """
    try:
        from app.services.answering import explain_methodology
        
        methodology_explanation = await explain_methodology(document_id, aspect)
        
        return {
            "document_id": document_id,
            "methodology_aspect": aspect,
            "explanation": methodology_explanation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error explaining methodology: {str(e)}")

@router.get("/documents")
async def list_documents():
    """
    List all ingested documents.
    """
    try:
        from app.services.retrieval import get_all_documents
        documents = await get_all_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a specific document and all its chunks from the vector store.
    """
    try:
        from app.services.retrieval import delete_document
        
        success = await delete_document(document_id)
        
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found or could not be deleted")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")