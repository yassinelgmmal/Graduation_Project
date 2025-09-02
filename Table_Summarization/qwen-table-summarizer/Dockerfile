FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8004

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application with proper error output
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004", "--log-level", "debug"]
