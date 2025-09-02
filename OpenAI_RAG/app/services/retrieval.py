import os
from typing import List, Dict, Optional, Tuple
import logging

from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

from app.config import settings

logger = logging.getLogger(__name__)

async def _get_vectorstore() -> Chroma:
    """
    Initialize and return the Chroma vector store.
    """
    if not os.path.exists(settings.chroma_directory):
        raise ValueError(f"Chroma directory {settings.chroma_directory} does not exist. No documents have been ingested yet.")
    
    embedding_fn = AzureOpenAIEmbeddings(
        azure_deployment=settings.embedding_deployment_name,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version
    )

    return Chroma(
        persist_directory=settings.chroma_directory,
        collection_name=settings.chroma_collection,
        embedding_function=embedding_fn,
    )

async def retrieve_documents(
    query: str,
    top_k: int = 5,
    filter_kwargs: Optional[Dict[str, str]] = None,
    score_threshold: float = 0.7
) -> List[Document]:
    """
    Retrieve the most relevant document chunks for a given query from Chroma.

    Args:
        query: the user's question or search string.
        top_k: number of top results to return.
        filter_kwargs: optional metadata filters, e.g., {"document_id": "uuid"}.
        score_threshold: minimum similarity score for results.

    Returns:
        A list of LangChain Document objects.
    """
    try:
        vectordb = await _get_vectorstore()

        # Perform similarity search with scores
        if filter_kwargs:
            results = vectordb.similarity_search_with_score(
                query,
                k=top_k,
                filter=filter_kwargs
            )
        else:
            results = vectordb.similarity_search_with_score(
                query,
                k=top_k
            )

        # Filter by score threshold
        filtered_results = [
            doc for doc, score in results 
            if score >= score_threshold
        ]

        logger.info(f"Retrieved {len(filtered_results)} documents for query: {query[:50]}...")
        return filtered_results

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []

async def retrieve_documents_with_scores(
    query: str,
    top_k: int = 5,
    filter_kwargs: Optional[Dict[str, str]] = None,
    score_threshold: float = 0.0
) -> List[Dict]:
    """
    Retrieve documents with their similarity scores.
    
    Args:
        query: The user's query
        top_k: Number of documents to retrieve
        filter_kwargs: Optional metadata filters
        score_threshold: Minimum similarity score for results
        
    Returns:
        List of dictionaries with document content, metadata, and score
    """
    try:
        vectordb = await _get_vectorstore()

        if filter_kwargs:
            logger.info(f"Searching with filter: {filter_kwargs}")
            results = vectordb.similarity_search_with_score(
                query,
                k=top_k,
                filter=filter_kwargs
            )
        else:
            logger.info(f"Searching without filter for query: {query}")
            results = vectordb.similarity_search_with_score(
                query,
                k=top_k
            )
        
        # Log raw results
        logger.info(f"Raw search results: {len(results)} documents found")
        
        # Log the first few results with their scores for debugging
        for i, (doc, score) in enumerate(results[:3]):
            doc_id = doc.metadata.get('document_id', 'unknown')
            chunk_id = doc.metadata.get('chunk_id', 'unknown')
            logger.info(f"Result #{i+1}: doc_id={doc_id}, chunk_id={chunk_id}, score={score:.4f}")
        
        # Convert to standardized format and filter by score threshold
        processed_results = []
        for doc, score in results:
            if score >= score_threshold:
                processed_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            else:
                logger.info(f"Document filtered out due to low score: {score} < {score_threshold}")
        
        logger.info(f"Retrieved {len(processed_results)} documents with scores for query: {query[:50]}...")
        return processed_results

    except Exception as e:
        logger.error(f"Error retrieving documents with scores: {e}")
        return []

async def get_all_documents() -> List[Dict]:
    """
    Get information about all ingested documents.
    """
    try:
        vectordb = await _get_vectorstore()
        all_docs = vectordb.get()
        
        # Group by document_id
        document_info = {}
        for i, metadata in enumerate(all_docs.get('metadatas', [])):
            doc_id = metadata.get('document_id', 'unknown')
            filename = metadata.get('filename', 'unknown')
            
            if doc_id not in document_info:
                document_info[doc_id] = {
                    'document_id': doc_id,
                    'filename': filename,
                    'chunk_count': 0,
                    'chunk_types': set()
                }
            
            document_info[doc_id]['chunk_count'] += 1
            document_info[doc_id]['chunk_types'].add(metadata.get('chunk_type', 'unknown'))
        
        # Convert sets to lists for JSON serialization
        for doc_info in document_info.values():
            doc_info['chunk_types'] = list(doc_info['chunk_types'])
        
        return list(document_info.values())

    except Exception as e:
        logger.error(f"Error getting all documents: {e}")
        return []


async def retrieve_documents_by_id(
    doc_id: str,
    top_k: int = 100
) -> List[Document]:
    """
    Retrieve document chunks by document ID directly without using similarity search.
    This is more reliable for getting all chunks belonging to a document.

    Args:
        doc_id: The document ID to retrieve
        top_k: Maximum number of chunks to retrieve (default 100)

    Returns:
        A list of LangChain Document objects for the document
    """
    try:
        vectordb = await _get_vectorstore()
        
        # Use Chroma's get method instead of similarity search
        results = vectordb.get(
            where={"document_id": doc_id},
            limit=top_k
        )
        
        if not results['documents']:
            logger.warning(f"No documents found with ID: {doc_id}")
            return []
            
        # Convert to LangChain Document objects
        documents = []
        for i, (doc_text, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            documents.append(
                Document(
                    page_content=doc_text,
                    metadata=metadata
                )
            )
            
        logger.info(f"Retrieved {len(documents)} documents for ID: {doc_id}")
        return documents
        
    except Exception as e:
        logger.error(f"Error retrieving documents by ID: {e}")
        return []
async def get_document_by_id(doc_id: str) -> Optional[Dict]:
    """
    Get information about a specific document.
    """
    logger.info(f"Retrieving document: {doc_id}")
    try:

        vectordb = await _get_vectorstore()
        
        results = vectordb.get(
            where={"document_id": doc_id}
        )        
        
        if not results.get('metadatas'):
            return None
        
        metadatas = results['metadatas']
        
        # Aggregate information
        doc_info = {
            'document_id': doc_id,
            'filename': metadatas[0].get('filename', 'unknown'),
            'chunk_count': len(metadatas),
            'chunk_types': list(set(meta.get('chunk_type', 'unknown') for meta in metadatas))
        }
        
        return doc_info

    except Exception as e:
        logger.error(f"Error getting document {doc_id}: {e}")
        return None

async def delete_document(doc_id: str) -> bool:
    """
    Delete all chunks for a specific document.
    """
    try:
        vectordb = await _get_vectorstore()
        
        # Delete by document_id filter
        vectordb.delete(where={"document_id": doc_id})
        
        logger.info(f"Deleted document {doc_id}")
        return True

    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
        return False

async def retrieve_document_by_id(doc_id: str) -> Optional[Dict]:
    """
    Retrieve the content and metadata of a specific document by its ID.
    
    Args:
        doc_id: The document ID to retrieve
        
    Returns:
        Dictionary with document content and metadata, or None if not found
    """
    try:
        vectordb = await _get_vectorstore()
        
        # Get all chunks for this document ID
        results = vectordb.get(
            where={"document_id": doc_id}
        )
        
        if not results or not results.get('metadatas') or not results.get('documents'):
            logger.warning(f"Document not found: {doc_id}")
            return None
        
        # Combine all chunks into a single document
        full_content = "\n\n".join(results.get('documents', []))
        
        # Use metadata from first chunk
        metadata = results.get('metadatas', [{}])[0].copy()
        metadata.update({
            'document_id': doc_id,
            'chunk_count': len(results.get('metadatas', [])),
            'full_document': True
        })
        
        return {
            "content": full_content,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error retrieving document by ID {doc_id}: {e}")
        return None