# Fast PEGASUS Summarization API with GPU CUDA Support
# Uses PyTorch base image with pre-installed CUDA support for faster builds

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set environment variables for CUDA and Python
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=all
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_CACHE=/models
ENV HF_HOME=/models
ENV TORCH_HOME=/models
ENV LOG_LEVEL=DEBUG
ENV RESOURCE_TRACKING_INTERVAL=30
ENV DOCKER_CONTAINER=true

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    htop \
    vim \
    procps \
    psmisc \
    lsof \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Install additional debugging tools
RUN pip install --no-cache-dir \
    debugpy \
    memory_profiler \
    py-spy \
    pytimeparse

# Copy the local fine-tuned model
COPY Pegasus-Fine-Tuned/checkpoint-200 /models/pegasus-fine-tuned

# Verify model can be loaded (quick test)
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('/models/pegasus-fine-tuned'); print('Model files accessible')"

# Create app directory
WORKDIR /app

# Create logs directory with proper permissions
RUN mkdir -p /app/logs
RUN chmod 777 /app/logs

# Create a separate volume for debug dumps
RUN mkdir -p /app/debug_dumps
RUN chmod 777 /app/debug_dumps

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app /models
USER appuser

# Copy application files
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser docker_entrypoint.py .
COPY --chown=appuser:appuser debug_utils.py .
COPY --chown=appuser:appuser simple_debug_utils.py .

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Expose port
EXPOSE 8003

# Set command with enhanced debugging options
CMD ["python", "-u", "docker_entrypoint.py"]
