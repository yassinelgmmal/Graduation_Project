from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import os
import json
from src.config import EMBEDDING_MODEL, VECTOR_DB_PATH

class RAGManager:
    """Manager for the Retrieval-Augmented Generation system"""
    
    def __init__(self):
        """Initialize the RAG Manager with embeddings model"""
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_db = None
        
    def create_documents_from_paper(self, paper_data):
        """
        Create document chunks from extracted paper data
        
        Args:
            paper_data (dict): Extracted data from the paper
            
        Returns:
            list: List of Document objects
        """
        documents = []
          # Add the overall summary as a separate chunk
        if "summary" in paper_data and paper_data["summary"]:
            print(f"Adding overall summary as a complete chunk (length: {len(paper_data['summary'])} chars)")
            doc = Document(
                page_content=paper_data["summary"],
                metadata={
                    "source": "summary",
                    "type": "overall_summary",
                    "paper_id": paper_data.get("paper_id", ""),
                    "paper_title": paper_data.get("title", ""),
                }
            )
            documents.append(doc)
        
        # Process full text if available
        if "full_text" in paper_data and paper_data["full_text"] and len(paper_data["full_text"]) > 0:
            full_text = paper_data["full_text"]
            print(f"Processing full text ({len(full_text)} chars) into chunks...")
            
            # Split full text into chunks with overlap
            chunks = self.split_text_into_chunks(full_text, max_chunk_size=1000, overlap=200)
            print(f"Split full text into {len(chunks)} chunks")
            
            # Add each chunk as a document
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": "full_text",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "paper_id": paper_data.get("paper_id", ""),
                        "paper_title": paper_data.get("title", ""),
                    }
                )
                documents.append(doc)
        
        # Add text sections as documents
        if "text_sections" in paper_data:
            for section in paper_data["text_sections"]:
                section_title = section.get("title", "Untitled Section")
                section_text = section.get("text", "")
                
                if section_text:
                    # Split section text into chunks with 200 character overlap
                    chunks = self.split_text_into_chunks(section_text, max_chunk_size=1000, overlap=200)
                    print(f"Adding {len(chunks)} chunks for section '{section_title}'")
                    
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": "text",
                                "section_title": section_title,
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "paper_id": paper_data.get("paper_id", ""),
                                "paper_title": paper_data.get("title", ""),
                            }
                        )
                        documents.append(doc)
          
        # Add table summaries as documents
        if "tables" in paper_data:
            for idx, table in enumerate(paper_data["tables"]):
                table_summary = table.get("summary", "")
                table_caption = table.get("caption", f"Table {idx+1}")
                
                if table_summary:
                    # Ensure table summaries are kept as whole chunks without splitting
                    print(f"Adding table summary as a complete chunk (id: {table.get('id', f'table_{idx}')}, length: {len(table_summary)} chars)")
                    doc = Document(
                        page_content=table_summary,
                        metadata={
                            "source": "table",
                            "table_caption": table_caption,
                            "table_id": table.get("id", f"table_{idx}"),
                            "paper_id": paper_data.get("paper_id", ""),
                            "paper_title": paper_data.get("title", ""),
                        }
                    )
                    documents.append(doc)
          
        # Add figure summaries as documents
        if "figures" in paper_data:
            for idx, figure in enumerate(paper_data["figures"]):
                figure_summary = figure.get("summary", "")
                figure_caption = figure.get("caption", f"Figure {idx+1}")
                
                if figure_summary:
                    # Ensure figure summaries are kept as whole chunks without splitting
                    print(f"Adding figure summary as a complete chunk (id: {figure.get('id', f'figure_{idx}')}, length: {len(figure_summary)} chars)")
                    doc = Document(
                        page_content=figure_summary,
                        metadata={
                            "source": "figure",
                            "figure_caption": figure_caption,
                            "figure_id": figure.get("id", f"figure_{idx}"),
                            "paper_id": paper_data.get("paper_id", ""),
                            "paper_title": paper_data.get("title", ""),
                        }
                    )
                    documents.append(doc)
        
        return documents
    
    def ingest_paper(self, paper_data):
        """
        Ingest a scientific paper into the vector database
        
        Args:
            paper_data (dict): Processed paper data with summaries
            
        Returns:
            bool: Success status
        """
        try:
            # Create document chunks
            documents = self.create_documents_from_paper(paper_data)
            
            if not documents:
                print("No valid documents created from paper data")
                return False
                
            # Initialize vector store if needed
            if self.vector_db is None:                
                if os.path.exists(VECTOR_DB_PATH):
                    try:
                        print("Loading existing vector database")
                        self.vector_db = FAISS.load_local(VECTOR_DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
                    except Exception as e:
                        print(f"Error loading vector database: {str(e)}")
                        print("Creating new vector database")
                        self.vector_db = FAISS.from_documents(documents, self.embeddings)
                else:
                    print("Creating new vector database")
                    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
                    self.vector_db = FAISS.from_documents(documents, self.embeddings)
            else:
                # Add documents to existing vector store
                self.vector_db.add_documents(documents)
                
            # Save the updated vector store
            self.vector_db.save_local(VECTOR_DB_PATH)
            return True
            
        except Exception as e:
            print(f"Error ingesting paper: {str(e)}")
            return False
    
    def retrieve_relevant_chunks(self, query, k=5):
        """
        Retrieve relevant chunks from the vector database
        
        Args:
            query (str): Query text
            k (int): Number of chunks to retrieve
            
        Returns:
            list: List of retrieved documents
        """ 
        if self.vector_db is None:
            if not self.reload_vector_db():
                print("Failed to load vector database")
                return []
        
        # Perform similarity search
        retrieved_docs = self.vector_db.similarity_search(query, k=k)
        return retrieved_docs
    
    def save_paper_data(self, paper_data, paper_id):
        """
        Save processed paper data to disk
        
        Args:
            paper_data (dict): Processed paper data
            paper_id (str): Unique paper identifier
            
        Returns:
            str: Path to saved file
        """
        os.makedirs("data/papers", exist_ok=True)
        file_path = f"data/papers/{paper_id}.json"
        
        with open(file_path, 'w') as f:
            json.dump(paper_data, f, indent=2)
            
        return file_path
    
    def reload_vector_db(self):
        """
        Reload the vector database with the allow_dangerous_deserialization parameter
        to fix the pickle loading issue.
        
        Returns:
            bool: Success status
        """
        try:
            if os.path.exists(VECTOR_DB_PATH):
                print("Reloading vector database with allow_dangerous_deserialization=True")
                self.vector_db = FAISS.load_local(
                    VECTOR_DB_PATH, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return True
            else:
                print("Vector database does not exist")
                return False
        except Exception as e:
            print(f"Error reloading vector database: {str(e)}")
            return False
    
    def split_text_into_chunks(self, text, max_chunk_size=1000, overlap=200):
        """
        Split text into chunks of up to max_chunk_size characters with specified overlap,
        preserving sentence boundaries if possible.
        
        Args:
            text (str): The text to split into chunks
            max_chunk_size (int): Maximum size of each chunk
            overlap (int): Number of characters to overlap between chunks
            
        Returns:
            list: List of text chunks
        """
        import re
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        
        chunks = []
        current_chunk = ''
        current_length = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed the max chunk size
            if current_length + len(sentence) + 1 > max_chunk_size:
                if current_chunk:  # Save the current chunk if it's not empty
                    chunks.append(current_chunk.strip())
                    
                    # Start a new chunk with overlap
                    overlap_point = max(0, len(current_chunk) - overlap)
                    current_chunk = current_chunk[overlap_point:] + (' ' if current_chunk else '') + sentence
                    current_length = len(current_chunk)
                else:
                    # If the sentence itself is longer than max_chunk_size, we need to split it
                    if len(sentence) > max_chunk_size:
                        # Just take the first max_chunk_size characters
                        chunks.append(sentence[:max_chunk_size])
                        current_chunk = sentence[max(0, max_chunk_size - overlap):]
                        current_length = len(current_chunk)
                    else:
                        current_chunk = sentence
                        current_length = len(sentence)
            else:
                # Add the sentence to the current chunk
                current_chunk += (' ' if current_chunk else '') + sentence
                current_length = len(current_chunk)
                
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
