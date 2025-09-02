# Configuration file for API endpoints and keys

# PDF Processing API
PDF_EXTRACTOR_API_ENDPOINT = "http://paper-extractor:8001/process-pdf/"

# Classification API
PAPER_CLASSIFICATION_API_ENDPOINT = "http://paper-classification:8002/predict"

# Text Summarization API (PEGASUS)
TEXT_SUMMARIZATION_API_ENDPOINT = "http://pegasus-api:8003/summarize"

# Table Summarization API (Qwen)
TABLE_SUMMARIZATION_API_ENDPOINT = "http://table-summarizer:8004/summarize_table/"

# Figure Summarization API endpoints
FIGURE_CAPTION_BATCH_API_ENDPOINT = "http://20.245.224.246:8020/captions"
AZURE_OPENAI_API_ENDPOINT = "https://azureopenairag281.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview"
AZURE_OPENAI_API_KEY = "6AQ3ve5mIP9xC392HeDB9gCxAcIcVeM8v070U3SMuPuS1i8Js690JQQJ99BFACHYHv6XJ3w3AAABACOGFTN6"

# RAG Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_PATH = "data/vector_db"
