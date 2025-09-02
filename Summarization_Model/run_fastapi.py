#!/usr/bin/env python3
"""
FastAPI Server Launcher for PEGASUS Summarization API

This script launches the FastAPI server with optimal configuration for production use.
"""

import uvicorn
import os
import sys

def main():
    """Launch the FastAPI server"""
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    print("\n" + "="*60)
    print("🚀 PEGASUS Document Summarization API - FastAPI")
    print("="*60)
    print(f"📍 Server will start on: http://localhost:8000")
    print(f"📖 Documentation: http://localhost:8000")
    print(f"🔧 Interactive API Docs: http://localhost:8000/docs")
    print(f"📚 ReDoc Documentation: http://localhost:8000/redoc")
    print(f"🏥 Health check: http://localhost:8000/health")
    print(f"🤖 Model info: http://localhost:8000/model-info")
    print(f"✨ Summarization: POST http://localhost:8000/summarize")
    print("="*60)
    print("Loading model and starting server...")
    print("="*60)
    
    # Configure uvicorn settings
    config = {
        "app": "app_fastapi:app",
        "host": "0.0.0.0",
        "port": 8000,
        "log_level": "info",
        "access_log": False,
        "reload": False,  # Set to True for development
        "workers": 1,     # Single worker to avoid model loading issues
    }
    
    # Run the server
    uvicorn.run(**config)

if __name__ == "__main__":
    main()
