FROM python:3.10-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application code
COPY main.py .
COPY merged_model_final/ merged_model_final/

# Expose the port that FastAPI will run on
EXPOSE 8002

# Command to run the FastAPI application with uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
