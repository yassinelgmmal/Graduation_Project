from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.retrieval import retrieve_documents, retrieve_documents_by_id, get_document_by_id
from app.models.text_model import TextModel
from app.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/summarize", tags=["Summarization"])

class SummaryRequest(BaseModel):
    doc_id: str
    summary_type: str = "comprehensive"  # "comprehensive", "executive", "technical"
    max_length: Optional[int] = None

class SummaryResponse(BaseModel):
    document_id: str
    summary: str
    summary_type: str
    word_count: int
    source_chunks: int

@router.post("", response_model=SummaryResponse)
async def summarize_document_post(request: SummaryRequest):
    logger.info(f"Summarizing document {request.doc_id} with type {request.summary_type}")
    try:
        return await _generate_summary(
            request.doc_id, 
            request.summary_type, 
            request.max_length
        )
    except Exception as e:
        logger.error(f"Error summarizing document {request.doc_id}: {str(e)}", exc_info=True)
        raise

@router.get("/{doc_id}", response_model=SummaryResponse)
async def summarize_document(
    doc_id: str, 
    summary_type: str = "comprehensive",
    max_length: Optional[int] = None
):
    """
    Generate a summary of an ingested document by its ID.
    """
    return await _generate_summary(doc_id, summary_type, max_length)

async def _generate_summary(doc_id: str, summary_type: str, max_length: Optional[int] = None):
    """
    Internal function to generate document summary.
    """
    try:
        # Retrieve all chunks for this document using direct ID lookup
        docs = await retrieve_documents_by_id(doc_id, top_k=100)
        
        # Fallback to similarity search if direct lookup fails
        if not docs:
            logger.warning(f"Direct lookup failed for document ID: {doc_id}, trying similarity search")
            docs = await retrieve_documents(
                "", 
                top_k=100, 
                filter_kwargs={"document_id": doc_id},
                score_threshold=0.0  # No score threshold for document retrieval by ID
            )
        
        if not docs:
            logger.warning(f"No documents found for document ID: {doc_id}")
            raise HTTPException(status_code=404, detail="Document not found or no content.")
        
        # Log the number of chunks found
        logger.info(f"Found {len(docs)} chunks for document ID: {doc_id}")
        
        # Separate content by type
        text_chunks = [d.page_content for d in docs if d.metadata.get("chunk_type") == "text"]
        table_chunks = [d.page_content for d in docs if d.metadata.get("chunk_type") == "table"]
        figure_chunks = [d.page_content for d in docs if d.metadata.get("chunk_type") == "figure"]
        
        # Log content types found
        logger.info(f"Content breakdown - Text: {len(text_chunks)}, Tables: {len(table_chunks)}, Figures: {len(figure_chunks)}")
        
        # Combine content with structure
        combined_content = ""
        
        if text_chunks:
            combined_content += "TEXT CONTENT:\n" + "\n\n".join(text_chunks) + "\n\n"
        
        if table_chunks:
            combined_content += "TABLES:\n" + "\n\n".join(table_chunks) + "\n\n"
        
        if figure_chunks:
            combined_content += "FIGURES:\n" + "\n\n".join(figure_chunks) + "\n\n"
        
        if not combined_content:
            raise HTTPException(status_code=404, detail="No content found for document.")
        
        # Initialize text model
        text_model = TextModel()
        
        # Generate summary based on type
        summary_prompts = {
            "comprehensive": "Provide a detailed comprehensive summary of this scientific document, covering all key points, methods, results, and conclusions. Include important technical details and findings:",
            "executive": "Provide a concise executive summary highlighting the main purpose, findings, and significance of this scientific document:",
            "technical": "Provide a technical summary focusing on research methodology, technical implementation details, and quantitative results from this scientific document:"
        }
        
        prompt = summary_prompts.get(summary_type, summary_prompts["comprehensive"])
        
        # Set appropriate max length
        if not max_length:
            length_by_type = {
                "comprehensive": 1500,  # Increased from 1000
                "executive": 500,       # Increased from 300
                "technical": 1200       # Increased from 800
            }
            max_length = length_by_type.get(summary_type, 1500)
        
        # Generate the summary
        logger.info(f"Generating {summary_type} summary with max length {max_length}")
        summary = await text_model.summarize(
            combined_content,
            prompt=prompt,
            max_length=max_length
        )
        
        return SummaryResponse(
            document_id=doc_id,
            summary=summary,
            summary_type=summary_type,
            word_count=len(summary.split()),
            source_chunks=len(docs)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@router.get("/document/{doc_id}/info")
async def get_document_info(doc_id: str):
    """
    Get information about a document including chunk counts by type.
    """
    try:
        docs = await retrieve_documents_by_id(doc_id, top_k=1000)
        
        # Fallback to similarity search if direct lookup fails
        if not docs:
            logger.warning(f"Direct lookup failed for document info: {doc_id}, trying similarity search")
            docs = await retrieve_documents(
                "", 
                top_k=1000, 
                filter_kwargs={"document_id": doc_id},
                score_threshold=0.0
            )
        
        if not docs:
            raise HTTPException(status_code=404, detail="Document not found.")
        
        # Count chunks by type
        chunk_counts = {}
        for doc in docs:
            chunk_type = doc.metadata.get("chunk_type", "unknown")
            chunk_counts[chunk_type] = chunk_counts.get(chunk_type, 0) + 1
        
        return {
            "document_id": doc_id,
            "total_chunks": len(docs),
            "chunk_counts": chunk_counts,
            "document_title": docs[0].metadata.get("title", "Unknown"),
            "source_file": docs[0].metadata.get("source", "Unknown")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document info: {str(e)}")