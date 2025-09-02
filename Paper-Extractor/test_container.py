#!/usr/bin/env python3
"""
Test script for the containerized PDF extractor application
"""
import requests
import os
import time

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8001/", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_api_docs():
    """Test if API documentation is accessible"""
    try:
        response = requests.get("http://localhost:8001/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API documentation accessible")
            return True
        else:
            print(f"❌ API docs failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API docs test failed: {e}")
        return False

def main():
    print("Testing PDF Extractor Docker Container...")
    print("=" * 50)
    
    # Wait for container to start
    print("Waiting for container to start...")
    time.sleep(5)
    
    # Run tests
    health_ok = test_health_endpoint()
    docs_ok = test_api_docs()
    
    print("=" * 50)
    if health_ok and docs_ok:
        print("✅ All tests passed! Container is running correctly.")
        print("🌐 Access the API at: http://localhost:8001")
        print("📚 View API docs at: http://localhost:8001/docs")
    else:
        print("❌ Some tests failed. Check the container logs.")
        
if __name__ == "__main__":
    main()
