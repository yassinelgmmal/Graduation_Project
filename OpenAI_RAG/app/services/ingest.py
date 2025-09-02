import os
import uuid
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

from app.utils.pdf_parser import parse_pdf
from app.utils.chunker import chunk_text, chunk_table, chunk_figure
from app.models.text_model import TextModel
from app.models.table_model import TableModel
from app.models.figure_model import FigureModel
from app.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def ingest_document(pdf_path: str) -> Tuple[str, int]:
    """
    Main pipeline: parse PDF -> chunk by modality -> summarize -> embed and store.

    Returns:
        Tuple[str, int]: (document_id, number_of_chunks_processed)
    """
    try:
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Ensure chroma directory
        os.makedirs(settings.chroma_directory, exist_ok=True)        # Initialize vectorstore with Azure OpenAI embeddings
        embedding_fn = AzureOpenAIEmbeddings(
            azure_deployment=settings.embedding_deployment_name,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version="2023-05-15"
        )
        
        vectordb = Chroma(
            persist_directory=settings.chroma_directory,
            collection_name=settings.chroma_collection,
            embedding_function=embedding_fn,
        )

        # Parse PDF into elements
        logger.info(f"Parsing PDF: {pdf_path}")
        elements = await parse_pdf(Path(pdf_path))
        
        if not elements:
            raise ValueError("No content extracted from PDF")

        all_docs: List[Document] = []
        
        # Initialize models
        text_model = TextModel()
        table_model = TableModel()
        figure_model = FigureModel()

        logger.info(f"Processing {len(elements)} elements")
        logger.info(f"Element types extracted: {[elem['type'] for elem in elements]}")
        # Process each element based on its type
        for idx, elem in enumerate(elements):
            try:
                meta: Dict[str, Any] = elem.get("metadata", {})
                meta.update({
                    "source": pdf_path,
                    "document_id": doc_id,
                    "element_index": idx,
                    "chunk_type": elem["type"],
                    "filename": Path(pdf_path).name
                })

                # Process based on content type
                if elem["type"] == "text":
                    chunks = chunk_text(
                        elem["content"],
                        chunk_size=settings.text_chunk_size,
                        chunk_overlap=settings.text_chunk_overlap,
                    )
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        try:
                            logger.info(f"Processing text chunk {chunk_idx} with content: {chunk}")
                            summary = await text_model.summarize(chunk)
                            logger.info(f"Text chunk {chunk_idx} summary: {summary}")
                            chunk_meta = meta.copy()
                            chunk_meta["chunk_index"] = chunk_idx
                            all_docs.append(Document(page_content=summary, metadata=chunk_meta))
                        except Exception as e:
                            logger.error(f"Error processing text chunk {chunk_idx}: {e}")
                            # Store original content if summarization fails
                            chunk_meta = meta.copy()
                            chunk_meta["chunk_index"] = chunk_idx
                            chunk_meta["processing_error"] = str(e)
                            all_docs.append(Document(page_content=chunk, metadata=chunk_meta))

                elif elem["type"] == "table":
                    chunks = chunk_table(
                        elem["content"],
                        row_chunk_size=settings.table_chunk_size,
                        row_chunk_overlap=settings.table_chunk_overlap,
                    )
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        try:
                            logger.info(f"Processing table chunk {chunk_idx} with content: {chunk}")
                            summary = await table_model.summarize(chunk)
                            logger.info(f"Table chunk {chunk_idx} summary: {summary}")
                            chunk_meta = meta.copy()
                            chunk_meta["chunk_index"] = chunk_idx
                            all_docs.append(Document(page_content=summary, metadata=chunk_meta))
                        except Exception as e:
                            logger.error(f"Error processing table chunk {chunk_idx}: {e}")
                            chunk_meta = meta.copy()
                            chunk_meta["chunk_index"] = chunk_idx
                            chunk_meta["processing_error"] = str(e)
                            all_docs.append(Document(page_content=chunk, metadata=chunk_meta))

                elif elem["type"] == "figure":
                    try:
                        logger.info(f"Processing figure {idx} with content: {elem['content']}")
                        # For figures, we'll store the description/caption
                        caption = await figure_model.caption(elem["content"])
                        logger.info(f"Figure {idx} caption: {caption}")
                        figure_meta = meta.copy()
                        figure_meta["chunk_index"] = 0
                        all_docs.append(Document(page_content=caption, metadata=figure_meta))
                    except Exception as e:
                        logger.error(f"Error processing figure {idx}: {e}")
                        figure_meta = meta.copy()
                        figure_meta["chunk_index"] = 0
                        figure_meta["processing_error"] = str(e)
                        # Store basic figure information
                        all_docs.append(Document(
                            page_content=f"Figure {idx}: {elem.get('content', 'Figure content unavailable')}", 
                            metadata=figure_meta
                        ))

            except Exception as e:
                logger.error(f"Error processing element {idx}: {e}")
                continue

        if not all_docs:
            raise ValueError("No documents were successfully processed")

        # Add documents to vector store
        logger.info(f"Adding {len(all_docs)} chunks to vector store")
        vectordb.add_documents(all_docs)
        
        logger.info(f"Successfully ingested document with ID: {doc_id}")
        return doc_id, len(all_docs)

    except Exception as e:
        logger.error(f"Error ingesting document {pdf_path}: {e}")
        raise

def ingest_document_sync(pdf_path: str) -> Tuple[str, int]:
    """
    Synchronous wrapper for the async ingest_document function.
    """
    return asyncio.run(ingest_document(pdf_path))