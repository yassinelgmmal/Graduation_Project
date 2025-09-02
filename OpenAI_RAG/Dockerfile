FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    libmagic-dev \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install specific version of numpy first to avoid compatibility issues
RUN pip install numpy==1.24.3

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure langchain-openai is installed explicitly
RUN pip install --no-cache-dir langchain-openai>=0.0.2

# Ensure langchain-chroma is installed
RUN pip install --no-cache-dir langchain-chroma>=0.0.2

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p uploads chroma_db figures

# Expose port
EXPOSE 8010

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
