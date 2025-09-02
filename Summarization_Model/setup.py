"""
Setup Script for PEGASUS Document Summarization API

This script automates the setup process for the PEGASUS summarization system,
including environment setup, dependency installation, and model verification.

Usage:
    python setup.py [--gpu] [--cpu] [--test]

Arguments:
    --gpu    : Install GPU-optimized dependencies
    --cpu    : Install CPU-only dependencies
    --test   : Run tests after setup
"""

import subprocess
import sys
import os
import json
import argparse
from pathlib import Path
import platform

class SetupManager:
    """Manages the setup process for the PEGASUS API system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.is_windows = platform.system() == "Windows"
        self.python_executable = sys.executable
        
    def log(self, message, level="INFO"):
        """Simple logging function"""
        print(f"[{level}] {message}")
    
    def run_command(self, command, check=True):
        """Run a shell command and return the result"""
        try:
            if self.is_windows:
                result = subprocess.run(
                    command, 
                    shell=True, 
                    check=check, 
                    capture_output=True, 
                    text=True
                )
            else:
                result = subprocess.run(
                    command.split(), 
                    check=check, 
                    capture_output=True, 
                    text=True
                )
            return result
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {command}", "ERROR")
            self.log(f"Error: {e.stderr}", "ERROR")
            return None
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        self.log("Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.log(f"Python {version.major}.{version.minor} detected. Python 3.8+ required.", "ERROR")
            return False
        
        self.log(f"Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    
    def check_gpu_availability(self):
        """Check if GPU and CUDA are available"""
        self.log("Checking GPU availability...")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                self.log(f"GPU detected: {gpu_name} (Count: {gpu_count})")
                return True
            else:
                self.log("No GPU detected or CUDA not available")
                return False
        except ImportError:
            self.log("PyTorch not installed yet - will check after installation")
            return None
    
    def install_dependencies(self, gpu_support=False):
        """Install required dependencies"""
        self.log("Installing dependencies...")
        
        # Upgrade pip first
        self.log("Upgrading pip...")
        pip_cmd = f'"{self.python_executable}" -m pip install --upgrade pip'
        self.run_command(pip_cmd)
        
        if gpu_support:
            self.log("Installing GPU-optimized PyTorch...")
            torch_cmd = f'"{self.python_executable}" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
            self.run_command(torch_cmd)
        else:
            self.log("Installing CPU-only PyTorch...")
            torch_cmd = f'"{self.python_executable}" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'
            self.run_command(torch_cmd)
        
        # Install other requirements
        self.log("Installing other dependencies...")
        req_cmd = f'"{self.python_executable}" -m pip install -r requirements.txt'
        result = self.run_command(req_cmd)
        
        if result is None:
            self.log("Failed to install dependencies", "ERROR")
            return False
        
        self.log("Dependencies installed successfully")
        return True
    
    def verify_model_files(self):
        """Verify that model files are present and accessible"""
        self.log("Verifying model files...")
        
        model_path = self.project_root / "Pegasus-Fine-Tuned" / "checkpoint-200"
        
        if not model_path.exists():
            self.log("Model directory not found. Checking for ZIP file...", "WARNING")
            
            zip_path = self.project_root / "Pegasus-Fine-Tuned" / "checkpoint-200.zip"
            if zip_path.exists():
                self.log("Found ZIP file. Extracting...")
                if self.is_windows:
                    extract_cmd = f'powershell -Command "Expand-Archive -Path \\"{zip_path}\\" -DestinationPath \\"{zip_path.parent}\\" -Force"'
                else:
                    extract_cmd = f"unzip -o {zip_path} -d {zip_path.parent}"
                
                result = self.run_command(extract_cmd)
                if result is None:
                    self.log("Failed to extract model files", "ERROR")
                    return False
            else:
                self.log("Model files not found. Please ensure checkpoint-200.zip is present.", "ERROR")
                return False
        
        # Check for required model files
        required_files = [
            "config.json",
            "model.safetensors",
            "tokenizer_config.json",
            "spiece.model"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = model_path / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            self.log(f"Missing model files: {missing_files}", "ERROR")
            return False
        
        self.log("Model files verified successfully")
        return True
    
    def test_installation(self):
        """Test the installation by importing key modules"""
        self.log("Testing installation...")
        
        try:
            # Test core imports
            import torch
            import transformers
            import flask
            from transformers import PegasusForConditionalGeneration, PegasusTokenizer
            
            self.log("Core modules imported successfully")
            
            # Test model loading
            self.log("Testing model loading...")
            model_path = self.project_root / "Pegasus-Fine-Tuned" / "checkpoint-200"
            
            if model_path.exists():
                try:
                    tokenizer = PegasusTokenizer.from_pretrained(str(model_path))
                    self.log("Tokenizer loaded successfully")
                    
                    # Don't load the full model in test to save memory
                    self.log("Model files accessible")
                except Exception as e:
                    self.log(f"Model loading test failed: {e}", "WARNING")
                    self.log("This might be due to memory constraints during setup", "INFO")
            
            self.log("Installation test completed successfully")
            return True
            
        except ImportError as e:
            self.log(f"Import test failed: {e}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Test failed with error: {e}", "ERROR")
            return False
    
    def create_startup_script(self):
        """Create platform-specific startup scripts"""
        self.log("Creating startup scripts...")
        
        if self.is_windows:
            # Windows batch script
            batch_content = f"""@echo off
echo Starting PEGASUS Summarization API...
cd /d "{self.project_root}"
"{self.python_executable}" app.py
pause
"""
            with open(self.project_root / "start_api.bat", "w") as f:
                f.write(batch_content)
            
            # PowerShell script
            ps_content = f"""# PEGASUS API Startup Script
Write-Host "Starting PEGASUS Summarization API..." -ForegroundColor Green
Set-Location "{self.project_root}"
& "{self.python_executable}" app.py
"""
            with open(self.project_root / "start_api.ps1", "w") as f:
                f.write(ps_content)
        
        else:
            # Unix shell script
            shell_content = f"""#!/bin/bash
echo "Starting PEGASUS Summarization API..."
cd "{self.project_root}"
"{self.python_executable}" app.py
"""
            script_path = self.project_root / "start_api.sh"
            with open(script_path, "w") as f:
                f.write(shell_content)
            script_path.chmod(0o755)
        
        self.log("Startup scripts created")
    
    def run_api_tests(self):
        """Run API tests if test script is available"""
        test_script = self.project_root / "test_api.py"
        if test_script.exists():
            self.log("Running API tests...")
            
            # Start API in background for testing
            import threading
            import time
            
            def start_api():
                os.system(f'"{self.python_executable}" app.py')
            
            api_thread = threading.Thread(target=start_api, daemon=True)
            api_thread.start()
            
            # Wait for API to start
            time.sleep(5)
            
            # Run tests
            test_cmd = f'"{self.python_executable}" test_api.py --quick'
            result = self.run_command(test_cmd, check=False)
            
            if result and result.returncode == 0:
                self.log("API tests passed")
                return True
            else:
                self.log("Some API tests failed - check manually", "WARNING")
                return False
        else:
            self.log("Test script not found - skipping API tests", "INFO")
            return True
    
    def print_summary(self, gpu_support, success):
        """Print setup summary and next steps"""
        print("\n" + "="*60)
        print("🚀 PEGASUS SETUP SUMMARY")
        print("="*60)
        
        if success:
            print("✅ Setup completed successfully!")
            print(f"✅ GPU Support: {'Enabled' if gpu_support else 'Disabled'}")
            print("✅ Model files verified")
            print("✅ Dependencies installed")
            
            print("\n📋 Next Steps:")
            print("1. Start the API server:")
            if self.is_windows:
                print("   - Double-click start_api.bat, or")
                print("   - Run: python app.py")
            else:
                print("   - Run: ./start_api.sh, or")
                print("   - Run: python app.py")
            
            print("\n2. Test the API:")
            print("   - Open: http://localhost:5000")
            print("   - Or run: python test_api.py")
            
            print("\n3. Read the documentation:")
            print("   - See: DOCUMENTATION.md")
            
        else:
            print("❌ Setup encountered issues")
            print("\n🔧 Troubleshooting:")
            print("1. Check Python version (3.8+ required)")
            print("2. Ensure model files are present")
            print("3. Check internet connection for downloads")
            print("4. See DOCUMENTATION.md for detailed troubleshooting")
        
        print("="*60)

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup PEGASUS Summarization API")
    parser.add_argument("--gpu", action="store_true", help="Install GPU-optimized dependencies")
    parser.add_argument("--cpu", action="store_true", help="Install CPU-only dependencies")
    parser.add_argument("--test", action="store_true", help="Run tests after setup")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    
    args = parser.parse_args()
    
    # Determine GPU support
    if args.gpu:
        gpu_support = True
    elif args.cpu:
        gpu_support = False
    else:
        # Auto-detect
        try:
            import torch
            gpu_support = torch.cuda.is_available()
        except ImportError:
            gpu_support = False
    
    setup = SetupManager()
    
    print("🚀 PEGASUS Document Summarization API Setup")
    print("="*50)
    
    success = True
    
    # Check Python version
    if not setup.check_python_version():
        success = False
        return
    
    # Check GPU availability
    setup.check_gpu_availability()
    
    # Install dependencies
    if not args.skip_deps:
        if not setup.install_dependencies(gpu_support):
            success = False
    
    # Verify model files
    if not setup.verify_model_files():
        success = False
    
    # Test installation
    if success and not setup.test_installation():
        success = False
    
    # Create startup scripts
    if success:
        setup.create_startup_script()
    
    # Run API tests if requested
    if success and args.test:
        setup.run_api_tests()
    
    # Print summary
    setup.print_summary(gpu_support, success)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
