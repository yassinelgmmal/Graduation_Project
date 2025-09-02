# Qwen Table Summarizer API

A FastAPI-based web service that uses the `Qwen2-VL-2B-Instruct` model to extract and summarize data from scientific tables in images.

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/Tasneem14/qwen-table-summarizer.git
cd qwen-table-summarizer
```

## Running with Docker

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up --build

# Run in background
docker-compose up -d
```

### Option 2: Using Docker directly

```bash
# Build the Docker image
docker build -t qwen-table-summarizer .

# Run the container
docker run -p 8000:8000 qwen-table-summarizer
```

### GPU Support

The Docker Compose file is configured to use NVIDIA GPUs if available. Make sure you have:

- NVIDIA Container Toolkit installed (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker Compose version that supports GPU configuration

## Running without Docker

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Start the application**

```bash
uvicorn main:app --reload
```

## API Usage

Once the server is running, you can access:

- API documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Main endpoint: POST to http://localhost:8000/summarize_table/ with an image file

Example using curl:

```bash
curl -X POST -F "file=@table_image.jpg" http://localhost:8000/summarize_table/
```
