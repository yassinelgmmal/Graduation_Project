# Multimodal RAG System (Containerized)

This README provides instructions for running the Multimodal RAG system in a Docker container to ensure consistent dependency versions and avoid compatibility issues.

## Running with Docker

### Prerequisites

Make sure you have Docker and Docker Compose installed on your system.

- [Install Docker](https://docs.docker.com/get-docker/)
- [Install Docker Compose](https://docs.docker.com/compose/install/)

### Building and Running the Container

1. Navigate to the project directory:

```bash
cd "path/to/Langchain_RAG_FINAL"
```

2. Build and start the Docker container:

```bash
docker-compose up --build
```

This will:

- Build the Docker image with all required dependencies
- Start the container with the FastAPI application
- Map port 8000 from the container to your local machine
- Mount the uploads, chroma_db, and figures directories as volumes

3. Access the application at [http://localhost:8000](http://localhost:8000)

4. To stop the container, press `Ctrl+C` in the terminal where docker-compose is running, or run:

```bash
docker-compose down
```

## Development Notes

### Why Containerization?

The application uses the `unstructured` library which has specific version requirements for dependencies like NumPy and OpenCV. Containerization ensures these dependencies are installed with compatible versions.

### Key Benefits

1. **Consistent Environment**: The same dependency versions are used across all deployments
2. **Isolation**: The container has its own isolated environment, avoiding conflicts with system packages
3. **Portability**: The application can be run on any system with Docker, regardless of the host OS
4. **Fixed Dependencies**: Specific versions of NumPy and other libraries are installed to ensure compatibility

### Volume Mounts

The Docker Compose configuration mounts three directories as volumes:

- `./uploads`: For uploaded documents
- `./chroma_db`: For the vector database
- `./figures`: For extracted figures

This ensures that data persists even when the container is restarted.

## Running Without Docker (Alternative)

If you prefer not to use Docker, you can run the application directly, but you'll need to ensure compatible versions of dependencies:

1. Create a new Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install specific NumPy version first:

```bash
pip install numpy==1.24.3
```

3. Install other dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Note: This approach may still encounter compatibility issues depending on your system configuration.
