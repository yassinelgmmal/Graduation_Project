#!/usr/bin/env python3
"""
Startup script for the Multimodal RAG system.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_env_file():
    """Check if .env file exists and has required variables."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print("⚠️  .env file not found. Please copy .env.example to .env and configure it:")
            print(f"   cp {env_example} {env_file}")
            print("   Then edit .env with your Azure OpenAI API key")
        else:
            print("❌ Neither .env nor .env.example found")
        return False
      # Check for required variables
    with open(env_file, 'r') as f:
        content = f.read()
        
    if "AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here" in content:
        print("⚠️  Please update your Azure OpenAI API key in .env file")
        return False
    
    if "AZURE_OPENAI_API_KEY=" not in content:
        print("⚠️  AZURE_OPENAI_API_KEY not found in .env file")
        return False
    
    print("✅ .env file configured")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Error output:", e.stderr)
        return False

def create_directories():
    """Create necessary directories."""
    directories = ["uploads", "chroma_db", "figures"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Created necessary directories")

def run_server():
    """Start the FastAPI server."""
    print("🚀 Starting the Multimodal RAG server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("⏹️  Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

def main():
    """Main startup function."""
    print("🔬 Multimodal RAG System Startup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Check environment configuration
    if not check_env_file():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Start server
    run_server()

if __name__ == "__main__":
    main()
