from pydantic_settings import BaseSettings
from pydantic import Field
import os

class Settings(BaseSettings):
    # Azure OpenAI API settings
    azure_openai_api_key: str = Field(..., env="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field("2024-02-15-preview", env="AZURE_OPENAI_API_VERSION")

    azure_openai_api_key_credits_account: str = Field(..., env="AZURE_OPENAI_API_KEY_CREDITS_ACCOUNT")
    azure_openai_endpoint_credits_account: str = Field(..., env="AZURE_OPENAI_ENDPOINT_CREDITS_ACCOUNT")
    azure_openai_api_version_credits_account: str = Field("2024-12-01-preview", env="AZURE_OPENAI_API_VERSION_CREDITS_ACCOUNT")

    text_deployment_name: str = Field("gpt-35-turbo", env="TEXT_DEPLOYMENT_NAME")
    table_deployment_name: str = Field("gpt-35-turbo", env="TABLE_DEPLOYMENT_NAME")
    figure_deployment_name: str = Field("gpt-4o-mini", env="FIGURE_DEPLOYMENT_NAME")
    embedding_deployment_name: str = Field("text-embedding-ada-002", env="EMBEDDING_DEPLOYMENT_NAME")
    
    # Chroma DB settings
    chroma_directory: str = Field("./chroma_db", env="CHROMA_DIRECTORY")
    chroma_collection: str = Field("multimodal_papers", env="CHROMA_COLLECTION")

    # Chunking parameters
    text_chunk_size: int = Field(4000, env="TEXT_CHUNK_SIZE")
    text_chunk_overlap: int = Field(800, env="TEXT_CHUNK_OVERLAP")
    table_chunk_size: int = Field(1500, env="TABLE_CHUNK_SIZE")
    table_chunk_overlap: int = Field(150, env="TABLE_CHUNK_OVERLAP")

    # Legacy model names (kept for compatibility, but deployments are used instead)
    text_model_name: str = Field("gpt-35-turbo", env="TEXT_MODEL_NAME")
    table_model_name: str = Field("gpt-35-turbo", env="TABLE_MODEL_NAME")
    figure_model_name: str = Field("gpt-4-vision", env="FIGURE_MODEL_NAME")
    embedding_model: str = Field("text-embedding-ada-002", env="EMBEDDING_MODEL")

    class Config:
        env_file = ".env"

settings = Settings()