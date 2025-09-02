#!/bin/bash

# Build and run the Docker container in development mode

# Build the Docker image
echo "Building Docker image..."
docker build -t polysumm-frontend:dev .

# Run the container with development settings
echo "Running container in development mode..."
docker run -p 80:80 \
  -e BASE_URL="http://host.docker.internal:8010" \
  -e CUSTOM_MODELS_URL="http://host.docker.internal:8000" \
  -e DEBUG="true" \
  --name polysumm-dev \
  polysumm-frontend:dev

# Note: host.docker.internal maps to the host machine's IP address from inside Docker
# This makes it possible to connect to services running on the host machine
