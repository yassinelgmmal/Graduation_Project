# Docker Setup for PolySumm Frontend

This document explains how to build and run the PolySumm frontend using Docker.

## Prerequisites

- Docker installed on your machine
- Docker Compose installed on your machine (optional, but recommended)

## Building the Docker Image

### Using Docker Directly

To build the Docker image manually:

```bash
docker build -t polysumm-frontend .
```

To run the built image:

```bash
docker run -p 80:80 polysumm-frontend
```

### Using Docker Compose (Recommended)

For an easier setup, use Docker Compose:

```bash
docker-compose up -d
```

This will build and start the frontend container in detached mode.

## API Configuration

By default, the application is configured to connect to:

- Custom Models API at `http://localhost:8000`
- External API at `http://20.64.145.29:8010`

To modify these settings for a production environment, you have several options:

### Option 1: Update the config.js file before building

Edit the `src/config.js` file and update the API URLs before building the Docker image.

### Option 2: Use build-time arguments

Add build args to the Dockerfile and modify the docker-compose.yml file accordingly.

### Option 3: Runtime environment configuration

Implement environment variable injection at runtime using a custom entrypoint script.

## Development with Docker

For development, you can uncomment the volumes in the docker-compose.yml file to enable live updates without rebuilding the container.

## Troubleshooting

- **Connection issues to API**: Make sure the APIs are accessible from within the Docker container. You might need to use the Docker host IP or network-specific URLs.
- **Performance issues**: For Windows/Mac users, the Docker volume mounts might be slow. Consider using Docker's cache or alternative file sync solutions.
- **CORS errors**: If you encounter CORS issues, make sure your API servers allow requests from the Docker container's origin.

## Production Deployment

For production deployment:

1. Update the config.js with production API URLs
2. Build the image with `docker build -t polysumm-frontend:production .`
3. Consider using a registry to store your Docker image
4. Deploy using Docker Compose or Kubernetes
