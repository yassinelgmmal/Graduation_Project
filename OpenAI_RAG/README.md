# Multimodal RAG System for Scientific Papers

A sophisticated Retrieval-Augmented Generation (RAG) system designed specifically for scientific papers. This system can ingest PDF documents, extract and process text, tables, and figures separately, and provide intelligent Q&A capabilities.

## 🌟 Features

- **Multimodal PDF Processing**: Extracts and processes text, tables, and figures from scientific papers
- **Smart Chunking**: Uses different chunking strategies optimized for each content type
- **Azure OpenAI Integration**: Leverages Azure OpenAI GPT models for summarization and question answering
- **Vector Storage**: ChromaDB for efficient similarity search and retrieval
- **REST API**: FastAPI-based API with automatic documentation
- **Comprehensive Q&A**: Context-aware question answering with source attribution
- **Document Summarization**: Generate summaries for entire documents or specific sections
- **Methodology Analysis**: Extract and explain research methodologies

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Input     │    │   Processing    │    │   Storage       │
│                 │    │                 │    │                 │
│ • Research      │───▶│ • Text Extract  │───▶│ • ChromaDB      │
│   Papers        │    │ • Table Parse   │    │ • Embeddings    │
│ • Documents     │    │ • Figure OCR    │    │ • Metadata      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │    │   Retrieval     │    │   Generation    │
│                 │    │                 │    │                 │
│ • Questions     │───▶│ • Similarity    │───▶│ • Azure OpenAI  │
│ • Research      │    │   Search        │    │   GPT Models    │
│   Needs         │    │ • Filtering     │    │ • Contextual    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Azure OpenAI API key
- 4GB+ RAM recommended

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Langchain_RAG_FINAL
   ```

2. **Set up environment** ```bash

   # Copy environment template

   cp .env.example .env

   # Edit .env and add your Azure OpenAI configuration

   # AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here

   # AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

   # AZURE_OPENAI_API_VERSION=2024-02-15-preview

   ```

   ```

3. **Run the startup script**
   ```bash
   python start.py
   ```

The startup script will:

- Check Python version compatibility
- Install all required dependencies
- Create necessary directories
- Start the FastAPI server

### Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p uploads chroma_db figures

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📖 Usage

### 1. API Documentation

Once the server is running, visit:

- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

### 2. Ingest Documents

Upload a PDF scientific paper:

```bash
curl -X POST "http://localhost:8000/api/v1/qa/ingest" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_paper.pdf"
```

Response:

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "your_paper.pdf",
  "status": "success",
  "chunks_processed": 45
}
```

### 3. Ask Questions

Query the ingested documents:

```bash
curl -X POST "http://localhost:8000/api/v1/qa/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What methodology was used in this research?",
       "top_k": 5
     }'
```

### 4. Generate Summaries

Get document summaries:

```bash
curl -X GET "http://localhost:8000/api/v1/summarize/550e8400-e29b-41d4-a716-446655440000?summary_type=comprehensive"
```

## 🛠️ API Endpoints

### Document Management

- `POST /api/v1/qa/ingest` - Upload and process PDF documents
- `GET /api/v1/qa/documents` - List all ingested documents
- `DELETE /api/v1/qa/documents/{document_id}` - Delete a document

### Question & Answer

- `POST /api/v1/qa/ask` - Ask questions about documents
- `POST /api/v1/qa/follow-up` - Get follow-up question suggestions
- `GET /api/v1/qa/methodology/{document_id}` - Explain research methodology

### Summarization

- `GET /api/v1/summarize/{document_id}` - Generate document summaries
- `POST /api/v1/summarize/` - Generate custom summaries
- `GET /api/v1/summarize/document/{document_id}/info` - Get document information

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Azure OpenAI Deployment Names
TEXT_DEPLOYMENT_NAME=gpt-35-turbo
TABLE_DEPLOYMENT_NAME=gpt-35-turbo
FIGURE_DEPLOYMENT_NAME=gpt-4-vision
EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002

# ChromaDB Settings
CHROMA_DIRECTORY=./chroma_db
CHROMA_COLLECTION=multimodal_papers

# Chunking Parameters
TEXT_CHUNK_SIZE=1000
TEXT_CHUNK_OVERLAP=200
TABLE_CHUNK_SIZE=500
TABLE_CHUNK_OVERLAP=50

# Legacy Model Names (for compatibility)
TEXT_MODEL_NAME=gpt-35-turbo
TABLE_MODEL_NAME=gpt-35-turbo
FIGURE_MODEL_NAME=gpt-4-vision
EMBEDDING_MODEL=text-embedding-ada-002
```

### Model Options

The system supports various Azure OpenAI models:

- **Text Processing**: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`
- **Vision Tasks**: `gpt-4-vision-preview`, `gpt-4-turbo` (with vision)
- **Embeddings**: `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`

## 🔧 Development

### Project Structure

```
app/
├── main.py              # FastAPI application entry point
├── config.py            # Configuration settings
├── routers/             # API route handlers
│   ├── qa.py           # Q&A endpoints
│   └── summary.py      # Summarization endpoints
├── services/           # Business logic
│   ├── ingest.py       # Document ingestion pipeline
│   ├── retrieval.py    # Vector search and retrieval
│   └── answering.py    # Question answering logic
├── models/             # AI model wrappers
│   ├── text_model.py   # Text processing
│   ├── table_model.py  # Table analysis
│   └── figure_model.py # Figure/image analysis
├── utils/              # Utility functions
│   ├── pdf_parser.py   # PDF parsing and extraction
│   └── chunker.py      # Content chunking strategies
└── storage/            # Data storage
    └── vectorstore.py  # Vector database wrapper
```

### Adding New Features

1. **New Endpoints**: Add routes in `app/routers/`
2. **Processing Logic**: Implement in `app/services/`
3. **Model Integration**: Create wrappers in `app/models/`
4. **Utilities**: Add helper functions in `app/utils/`

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

## 📊 Performance Considerations

### Chunking Strategy

- **Text**: Recursive character splitting with scientific paper optimizations
- **Tables**: Row-based chunking with header preservation
- **Figures**: Individual processing with detailed captioning

### Memory Usage

- **Small documents** (1-10 pages): ~100-500 MB RAM
- **Large documents** (50+ pages): ~1-2 GB RAM
- **Concurrent processing**: Scale linearly with document count

### Processing Time

- **PDF Parsing**: 1-5 seconds per page
- **Embedding Generation**: 0.1-0.5 seconds per chunk
- **Q&A Response**: 2-10 seconds depending on context size

## 🚨 Troubleshooting

### Common Issues

1. **PyMuPDF Installation Error**

   ```bash
   # On Windows, try:
   pip install --force-reinstall PyMuPDF

   # Or use conda:
   conda install -c conda-forge pymupdf
   ```

2. **OpenAI API Errors**

   - Check API key validity
   - Verify sufficient API credits
   - Monitor rate limits

3. **Memory Issues**

   - Reduce chunk sizes in configuration
   - Process documents sequentially
   - Increase system RAM

4. **ChromaDB Persistence**
   ```bash
   # Clear ChromaDB if corrupted
   rm -rf ./chroma_db
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔒 Security Considerations

- Store OpenAI API keys securely
- Validate uploaded file types and sizes
- Implement rate limiting for production use
- Consider document access controls
- Sanitize user inputs

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For questions and support:

- Check the [API documentation](http://localhost:8000/docs)
- Review the troubleshooting section
- Open an issue on the repository

## 🔮 Future Enhancements

- [ ] Support for additional file formats (DOCX, TXT)
- [ ] Advanced figure analysis with OCR
- [ ] Multi-language document support
- [ ] Real-time collaboration features
- [ ] Enhanced visualization dashboard
- [ ] Integration with research databases
