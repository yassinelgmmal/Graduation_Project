from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from typing import List, Any
from app.config import settings

class VectorStore:
    def __init__(
        self,
        persist_directory: str,
        client_settings: dict = None,
    ):
        """
        Wrapper around Chroma DB with Azure OpenAI embeddings.

        Args:
            persist_directory: Path to persist the Chroma database.
            client_settings: Optional dict for Chroma client settings.
        """
        self.embedding = AzureOpenAIEmbeddings(
            azure_deployment=settings.embedding_deployment_name,
            openai_api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
            openai_api_key=settings.azure_openai_api_key
        )
        self.persist_directory = persist_directory
        self.client_settings = client_settings or {}
        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding,
            client_settings=self.client_settings,
        )

    def add_documents(self, texts: List[str], metadatas: List[dict]) -> None:
        """
        Add documents to the vector store.

        Args:
            texts: List of text chunks to embed and store.
            metadatas: List of metadata dicts corresponding to each text.
        """
        self.db.add_texts(texts=texts, metadatas=metadatas)
        self.db.persist()

    def retrieve(
        self,
        query: str,
        k: int = 5,
    ) -> List[Any]:
        """
        Retrieve the top-k most similar documents to the query.

        Args:
            query: The query string.
            k: Number of results to retrieve.

        Returns:
            List of documents with 'text' and 'metadata'.
        """
        results = self.db.similarity_search(query, k=k)
        return results