import os
import uuid
import logging
from typing import List, Union, Any, Dict
from pathlib import Path
import pandas as pd
from io import StringIO

# Import LangChain text splitters
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: List[str] = None
) -> List[str]:
    """
    Splits text into chunks using LangChain's RecursiveCharacterTextSplitter for natural boundaries.

    Args:
        text: the full text to split.
        chunk_size: maximum characters per chunk.
        chunk_overlap: characters to overlap between chunks.
        separators: list of separators for splitting priority.

    Returns:
        A list of text chunks.
    """
    try:
        if not text or not text.strip():
            return []
        
        if separators is None:
            # Use default separators optimized for scientific text
            separators = [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "! ",    # Exclamations
                "? ",    # Questions
                "; ",    # Semicolons
                ", ",    # Commas
                " ",     # Spaces
                ""       # Character level
            ]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
        
        chunks = splitter.split_text(text)
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        # Fallback: simple character-based chunking
        return _fallback_text_chunk(text, chunk_size, chunk_overlap)

def _fallback_text_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Fallback text chunking if LangChain fails.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        
        # Try to break at word boundary
        if end < text_length and not text[end].isspace():
            last_space = chunk.rfind(' ')
            if last_space > chunk_size // 2:  # Only break if we don't lose too much
                chunk = chunk[:last_space]
                end = start + last_space
        
        chunks.append(chunk.strip())
        start = end - chunk_overlap
        
        if start >= end:  # Prevent infinite loop
            start = end
    
    return [chunk for chunk in chunks if chunk.strip()]

def chunk_table(
    table_data: Union[str, Any],
    row_chunk_size: int = 10,
    row_chunk_overlap: int = 2
) -> List[str]:
    """
    Processes table data into manageable chunks.

    Args:
        table_data: Table data as string (HTML, CSV) or structured object.
        row_chunk_size: number of rows per chunk.
        row_chunk_overlap: overlapping rows between chunks.

    Returns:
        A list of serialized table chunks as strings.
    """
    try:
        # Handle different table input types
        if isinstance(table_data, str):
            return _chunk_table_string(table_data, row_chunk_size, row_chunk_overlap)
        else:
            # Try to extract table from object
            return _chunk_table_object(table_data, row_chunk_size, row_chunk_overlap)
            
    except Exception as e:
        logger.error(f"Error chunking table: {e}")
        # Fallback: treat as single text chunk
        return [str(table_data)]

def _chunk_table_string(table_str: str, row_chunk_size: int, row_chunk_overlap: int) -> List[str]:
    """
    Chunk table from string representation (CSV, HTML, etc.).
    """
    try:
        # Try to parse as CSV
        if ',' in table_str or '\t' in table_str:
            # Detect delimiter
            delimiter = ',' if ',' in table_str else '\t'
            df = pd.read_csv(StringIO(table_str), delimiter=delimiter)
            return _chunk_dataframe(df, row_chunk_size, row_chunk_overlap)
        
        # Try HTML table parsing
        elif '<table' in table_str.lower():
            tables = pd.read_html(StringIO(table_str))
            if tables:
                return _chunk_dataframe(tables[0], row_chunk_size, row_chunk_overlap)
        
        # Fallback: split by lines and treat as rows
        lines = [line.strip() for line in table_str.split('\n') if line.strip()]
        return _chunk_table_lines(lines, row_chunk_size, row_chunk_overlap)
        
    except Exception as e:
        logger.warning(f"Could not parse table string: {e}")
        return [table_str]

def _chunk_table_object(table_obj: Any, row_chunk_size: int, row_chunk_overlap: int) -> List[str]:
    """
    Chunk table from object (Unstructured table, etc.).
    """
    try:
        # Try to extract rows from Unstructured table
        if hasattr(table_obj, 'rows'):
            rows = []
            for row in table_obj.rows:
                if hasattr(row, 'cells'):
                    row_data = [cell.text.strip() if hasattr(cell, 'text') else str(cell) 
                              for cell in row.cells]
                    rows.append(row_data)
                else:
                    rows.append([str(cell) for cell in row])
            
            return _chunk_table_rows(rows, row_chunk_size, row_chunk_overlap)
        
        # Try to convert to DataFrame
        elif hasattr(table_obj, 'to_pandas'):
            df = table_obj.to_pandas()
            return _chunk_dataframe(df, row_chunk_size, row_chunk_overlap)
        
        else:
            # Fallback: convert to string
            return [str(table_obj)]
            
    except Exception as e:
        logger.warning(f"Could not process table object: {e}")
        return [str(table_obj)]

def _chunk_dataframe(df: pd.DataFrame, row_chunk_size: int, row_chunk_overlap: int) -> List[str]:
    """
    Chunk a pandas DataFrame.
    """
    chunks = []
    total_rows = len(df)
    
    if total_rows == 0:
        return []
    
    # Get column headers
    headers = df.columns.tolist()
    
    for start in range(0, total_rows, row_chunk_size - row_chunk_overlap):
        end = min(start + row_chunk_size, total_rows)
        chunk_df = df.iloc[start:end]
        
        # Convert to CSV string with headers
        csv_string = chunk_df.to_csv(index=False)
        chunks.append(csv_string)
        
        if end >= total_rows:
            break
    
    return chunks

def _chunk_table_rows(rows: List[List[str]], row_chunk_size: int, row_chunk_overlap: int) -> List[str]:
    """
    Chunk a list of table rows.
    """
    chunks = []
    total_rows = len(rows)
    
    if total_rows == 0:
        return []
    
    # Assume first row is header
    header = rows[0] if rows else []
    data_rows = rows[1:] if len(rows) > 1 else []
    
    for start in range(0, len(data_rows), row_chunk_size - row_chunk_overlap):
        end = min(start + row_chunk_size, len(data_rows))
        chunk_rows = [header] + data_rows[start:end] if header else data_rows[start:end]
        
        # Convert to CSV format
        csv_lines = [','.join(f'"{cell}"' if ',' in cell else str(cell) for cell in row) 
                    for row in chunk_rows]
        chunks.append('\n'.join(csv_lines))
        
        if end >= len(data_rows):
            break
    
    return chunks

def _chunk_table_lines(lines: List[str], row_chunk_size: int, row_chunk_overlap: int) -> List[str]:
    """
    Chunk table from list of text lines.
    """
    chunks = []
    total_lines = len(lines)
    
    for start in range(0, total_lines, row_chunk_size - row_chunk_overlap):
        end = min(start + row_chunk_size, total_lines)
        chunk_lines = lines[start:end]
        chunks.append('\n'.join(chunk_lines))
        
        if end >= total_lines:
            break
    
    return chunks

def chunk_figure(
    figure_data: Union[str, Any], 
    output_dir: str = './figures'
) -> str:
    """
    Process figure data and return a description or path.

    Args:
        figure_data: Figure data (could be image, description, path).
        output_dir: directory to store extracted figure images.

    Returns:
        String description or path to the figure.
    """
    try:
        # If it's already a string description, return it
        if isinstance(figure_data, str):
            return figure_data
        
        # Try to save actual image if available
        if hasattr(figure_data, 'image') or hasattr(figure_data, 'element'):
            return _save_figure_image(figure_data, output_dir)
        
        # Fallback: return string representation
        return str(figure_data)
        
    except Exception as e:
        logger.error(f"Error processing figure: {e}")
        return f"Figure processing failed: {str(e)}"

def _save_figure_image(fig_obj: Any, output_dir: str) -> str:
    """
    Save figure image to disk and return path.
    """
    try:
        from PIL import Image
        from io import BytesIO
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Try to extract PIL Image
        pil_image = None
        
        if hasattr(fig_obj, 'element') and hasattr(fig_obj.element, 'save'):
            pil_image = fig_obj.element
        elif hasattr(fig_obj, 'image') and hasattr(fig_obj.image, 'save'):
            pil_image = fig_obj.image
        elif hasattr(fig_obj, 'image_data'):
            # Try to load from bytes
            try:
                pil_image = Image.open(BytesIO(fig_obj.image_data))
            except Exception:
                pass
        
        if pil_image:
            # Generate unique filename
            fname = f"fig_{uuid.uuid4().hex}.png"
            path = os.path.join(output_dir, fname)
            pil_image.save(path)
            logger.debug(f"Saved figure to {path}")
            return path
        else:
            # Return description instead
            return f"Figure: {str(fig_obj)}"
            
    except ImportError:
        logger.warning("PIL not available for image processing")
        return f"Figure: {str(fig_obj)}"
    except Exception as e:
        logger.error(f"Error saving figure image: {e}")
        return f"Figure: {str(fig_obj)}"