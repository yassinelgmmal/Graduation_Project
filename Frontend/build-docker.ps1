# Build using different Dockerfile approaches
# This script helps you try different approaches if one fails

param (
    [switch]$UseAlt = $false,
    [switch]$Clean = $false
)

# Clean Docker cache if requested
if ($Clean) {
    Write-Host "Cleaning Docker build cache..." -ForegroundColor Yellow
    docker builder prune -f
}

# Determine which Dockerfile to use
$dockerfile = if ($UseAlt) { "Dockerfile.alt" } else { "Dockerfile" }
Write-Host "Building with $dockerfile..." -ForegroundColor Green

# Build the Docker image
docker build -t polysumm-frontend:latest -f $dockerfile .

# Check if build was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful! Running container..." -ForegroundColor Green
    
    # Stop any existing container
    docker stop polysumm-container 2>$null
    docker rm polysumm-container 2>$null
    
    # Run the container
    docker run -d -p 80:80 --name polysumm-container `
        -e BASE_URL="http://host.docker.internal:8010" `
        -e CUSTOM_MODELS_URL="http://host.docker.internal:8000" `
        polysumm-frontend:latest
    
    Write-Host "Container is running at http://localhost" -ForegroundColor Cyan
} else {
    Write-Host "Build failed. Try using the alternative Dockerfile with: ./build-docker.ps1 -UseAlt" -ForegroundColor Red
}
