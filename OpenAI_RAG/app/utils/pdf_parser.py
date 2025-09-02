import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import os

logger = logging.getLogger(__name__)

async def parse_pdf(path: Path) -> List[Dict[str, Any]]:
    """
    Parse a PDF into a list of elements, each tagged by modality.
    Returns a list of dicts:
    [
      {"type": "text", "content": str, "metadata": {...}},
      {"type": "table", "content": str, "metadata": {...}},
      {"type": "figure", "content": str, "metadata": {...}},
    ]
    """
    try:
        # Check if unstructured is available without actually importing it
        import importlib.util
        unstructured_spec = importlib.util.find_spec("unstructured")
        
        if unstructured_spec is None:
            logger.warning("Unstructured package not found, using fallback parser")
            return await _fallback_pdf_parse(path)
            
        # Check NumPy compatibility before attempting to import unstructured
        try:
            import numpy as np
            logger.info(f"Using NumPy version: {np.__version__}")
        except (ImportError, AttributeError) as np_err:
            logger.warning(f"NumPy issue detected: {np_err}")
            return await _fallback_pdf_parse(path)
          # Now try to import the unstructured components
        try:
            from unstructured.partition.pdf import partition_pdf
            from unstructured.documents.elements import Table, Text, Image, FigureCaption
            # Try to import Figure, but handle if it's not available
            try:
                from unstructured.documents.elements import Figure
            except ImportError:
                logger.warning("Figure class not found in unstructured library, using Image class as fallback")
                # Create a Figure class that inherits from Image
                class Figure(Image):
                    pass  # Fallback Figure class for compatibility
        except (ImportError, AttributeError) as e:
            logger.warning(f"Error importing unstructured components: {e}")
            return await _fallback_pdf_parse(path)
        
        logger.info(f"Starting PDF parsing for: {path}")
          # Run the CPU-intensive parsing in a thread pool
        elements = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: partition_pdf(
                filename=str(path),
                strategy="hi_res",  # High resolution for better table/figure extraction
                infer_table_structure=True,
                chunking_strategy="by_title",
                include_page_breaks=False,
                extract_images_in_pdf=True,
                include_metadata=False,
                max_characters=5000,
                new_after_n_chars=4800,
                combine_text_under_n_chars=3000,
                extract_image_block_types=["Image"]  # Explicitly extract figures
            )
        )
        
        parsed = []
        
        for i, elem in enumerate(elements):
            try:
                # Get basic metadata
                metadata = {
                    "element_id": i,
                    "page_number": getattr(elem.metadata, 'page_number', None) if hasattr(elem, 'metadata') else None,
                    "element_type": type(elem).__name__,
                }
                
                # Add additional metadata if available
                if hasattr(elem, 'metadata') and elem.metadata:
                    if hasattr(elem.metadata, 'coordinates'):
                        metadata["coordinates"] = elem.metadata.coordinates
                    if hasattr(elem.metadata, 'filename'):
                        metadata["source_file"] = elem.metadata.filename
                
                # Process based on element type
                if isinstance(elem, Table):
                    # Extract table as HTML or text
                    table_content = elem.metadata.text_as_html if hasattr(elem.metadata, 'text_as_html') else str(elem)
                    parsed.append({
                        "type": "table",
                        "content": table_content,
                        "metadata": metadata,
                    })
                    logger.debug(f"Extracted table from page {metadata.get('page_number', 'unknown')}")
                elif isinstance(elem, Image) or isinstance(elem, Figure) or "figure" in str(elem).lower() or "fig" in str(elem).lower():
                    # For images and figures, store the description or path
                    image_content = str(elem) if elem else f"Image on page {metadata.get('page_number', 'unknown')}"
                    
                    # Add additional metadata for figures/images
                    if hasattr(elem, 'metadata') and elem.metadata:
                        if hasattr(elem.metadata, 'image_path'):
                            metadata["image_path"] = elem.metadata.image_path
                        if hasattr(elem.metadata, 'image_width'):
                            metadata["image_width"] = elem.metadata.image_width
                        if hasattr(elem.metadata, 'image_height'):
                            metadata["image_height"] = elem.metadata.image_height
                    
                    parsed.append({
                        "type": "figure",
                        "content": image_content,
                        "metadata": metadata,
                    })
                    logger.info(f"Extracted figure from page {metadata.get('page_number', 'unknown')}")
                
                elif isinstance(elem, FigureCaption):
                    # Handle figure captions separately
                    caption_content = elem.text if hasattr(elem, 'text') else str(elem)
                    parsed.append({
                        "type": "figure",
                        "content": f"Figure caption: {caption_content}",
                        "metadata": {**metadata, "is_caption": True},
                    })
                    logger.info(f"Extracted figure caption from page {metadata.get('page_number', 'unknown')}")
                    
                elif isinstance(elem, Text) or hasattr(elem, 'text'):
                    # Text elements
                    text_content = elem.text if hasattr(elem, 'text') else str(elem)
                    if text_content.strip():  # Only add non-empty text
                        parsed.append({
                            "type": "text",
                            "content": text_content.strip(),
                            "metadata": metadata,
                        })
                        
                else:
                    # Fallback: treat as text
                    content = str(elem).strip()
                    if content:
                        parsed.append({
                            "type": "text",
                            "content": content,
                            "metadata": metadata,
                        })
                        
            except Exception as e:
                logger.warning(f"Error processing element {i}: {e}")
                # Try to salvage what we can
                try:
                    content = str(elem).strip()
                    if content:
                        parsed.append({
                            "type": "text",
                            "content": content,
                            "metadata": {"element_id": i, "processing_error": str(e)},
                        })
                except:
                    logger.error(f"Failed to salvage element {i}")
                    continue

        logger.info(f"Successfully parsed PDF: {len(parsed)} elements extracted")
        return parsed
        
    except ImportError as e:
        logger.error(f"Missing required dependencies for PDF parsing: {e}")
        # Fallback to basic text extraction
        return await _fallback_pdf_parse(path)
        
    except Exception as e:
        logger.error(f"Error parsing PDF {path}: {e}")
        # Try fallback parsing
        return await _fallback_pdf_parse(path)

async def _fallback_pdf_parse(path: Path) -> List[Dict[str, Any]]:
    """
    Fallback PDF parsing using PyPDF2 or PyMuPDF if unstructured fails.
    """
    try:
        import PyPDF2
        
        logger.info(f"Using fallback PDF parsing for: {path}")
        
        def extract_text():
            with open(path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_elements = []
                
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        text_elements.append({
                            "type": "text",
                            "content": text.strip(),
                            "metadata": {
                                "page_number": page_num + 1,
                                "extraction_method": "PyPDF2_fallback"
                            }
                        })
                
                return text_elements
        
        # Run in executor to avoid blocking
        elements = await asyncio.get_event_loop().run_in_executor(None, extract_text)
        
        logger.info(f"Fallback parsing extracted {len(elements)} text elements")
        return elements
        
    except ImportError:
        logger.error("PyPDF2 not available for fallback parsing")
        return []
    except Exception as e:
        logger.error(f"Fallback PDF parsing failed: {e}")
        return []

def validate_pdf_file(path: Path) -> bool:
    """
    Validate that the file is a readable PDF.
    """
    try:
        if not path.exists():
            return False
        
        if not path.suffix.lower() == '.pdf':
            return False
        
        # Basic PDF header check
        with open(path, 'rb') as file:
            header = file.read(4)
            return header == b'%PDF'
            
    except Exception as e:
        logger.error(f"Error validating PDF file {path}: {e}")
        return False

async def get_pdf_metadata(path: Path) -> Dict[str, Any]:
    """
    Extract metadata from PDF file.
    """
    try:
        import PyPDF2
        
        def extract_metadata():
            with open(path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata = reader.metadata or {}
                
                return {
                    "title": metadata.get("/Title", ""),
                    "author": metadata.get("/Author", ""),
                    "subject": metadata.get("/Subject", ""),
                    "creator": metadata.get("/Creator", ""),
                    "producer": metadata.get("/Producer", ""),
                    "creation_date": metadata.get("/CreationDate", ""),
                    "modification_date": metadata.get("/ModDate", ""),
                    "page_count": len(reader.pages),
                    "file_size": path.stat().st_size,
                }
        
        metadata = await asyncio.get_event_loop().run_in_executor(None, extract_metadata)
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting PDF metadata: {e}")
        return {
            "page_count": 0,
            "file_size": path.stat().st_size if path.exists() else 0,
            "error": str(e)
        }