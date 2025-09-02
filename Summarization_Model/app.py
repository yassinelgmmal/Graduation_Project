"""
PEGASUS Fine-tuned Document Summarization API

A RESTful API service that provides document summarization capabilities using a fine-tuned PEGASUS model.
The model is specifically fine-tuned for scientific paper summarization and can handle various
document types with proper context window management and length optimization.

Author: Generated for GP Final Project
Model: Zeyad12/pegasus-large-summarizer (Hugging Face hosted fine-tuned PEGASUS)
Base Model: Fine-tuned PEGASUS on Scientific Papers Dataset
Technology: FastAPI (Modern Python Web Framework)
"""

from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import warnings
import logging
import time
from typing import Dict, List, Optional, Any
import re
import os
import sys
import platform
from dataclasses import dataclass
import json
import uuid
import psutil
import gc
from collections import deque
import threading

# Function to get memory usage information
def get_memory_usage():
    """Get detailed memory usage information for debugging"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    mem_data = {
        "rss_mb": memory_info.rss / (1024 * 1024),  # RSS in MB
        "vms_mb": memory_info.vms / (1024 * 1024),  # VMS in MB
        "percent": process.memory_percent(),
        "total_system_mb": psutil.virtual_memory().total / (1024 * 1024),
        "available_system_mb": psutil.virtual_memory().available / (1024 * 1024),
    }
    
    return mem_data

# Function to get GPU information if available
def get_gpu_info():
    """Get GPU usage information for debugging if GPU is available"""
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    gpu_data = {
        "gpu_available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0),
    }
    
    # Try to get memory information if available
    try:
        gpu_data["memory_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_data["memory_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        gpu_data["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception as e:
        gpu_data["memory_error"] = str(e)
    
    return gpu_data

# Try to import transformers components with fallback for SentencePiece issues
try:
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    PEGASUS_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ PEGASUS models imported successfully")
except ImportError as e:
    if "sentencepiece" in str(e).lower():
        # SentencePiece not available, try alternative approach
        logger = logging.getLogger(__name__)
        logger.warning("⚠️ SentencePiece not available, trying alternative tokenizer...")
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            # We'll use a different model that doesn't require SentencePiece
            PegasusTokenizer = AutoTokenizer
            PegasusForConditionalGeneration = AutoModelForSeq2SeqLM
            PEGASUS_AVAILABLE = True
            logger.info("✅ Using alternative AutoTokenizer and AutoModel")
        except ImportError:
            PEGASUS_AVAILABLE = False
            logger.error("❌ No suitable models available")
    else:
        PEGASUS_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Failed to import transformers: {e}")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
import logging.handlers
import sys

# Create logs directory if it doesn't exist
log_dir = os.environ.get('LOG_DIR', 'logs')
log_level_str = os.environ.get('LOG_LEVEL', 'INFO')
log_level = getattr(logging, log_level_str.upper(), logging.INFO)

try:
    os.makedirs(log_dir, exist_ok=True)
    print(f"Created or verified log directory: {log_dir}")
    
    # Check if we can write to the log directory
    test_file_path = os.path.join(log_dir, "test_permissions.tmp")
    with open(test_file_path, 'w') as f:
        f.write("Testing write permissions")
    os.remove(test_file_path)
    print(f"Log directory is writable: {log_dir}")
    
    use_file_logging = True
except (IOError, PermissionError) as e:
    print(f"WARNING: Cannot write to log directory {log_dir}: {e}")
    print("File logging will be disabled. Using console logging only.")
    use_file_logging = False

# Configure logging with rotating file handler
log_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] [%(threadName)s-%(thread)d] %(filename)s:%(lineno)d:%(funcName)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set up console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

handlers = [console_handler]

# Add file handler if we can write to the log directory
if use_file_logging:
    log_file_path = os.path.join(log_dir, "pegasus_api.log")
    try:
        # Size-based rotation
        size_file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file_path,
            maxBytes=10485760,  # 10MB
            backupCount=10,
            encoding='utf-8'
        )
        size_file_handler.setFormatter(log_formatter)
        size_file_handler.setLevel(log_level)
        handlers.append(size_file_handler)
        
        # Time-based rotation (daily)
        time_log_file_path = os.path.join(log_dir, "pegasus_api_daily.log")
        time_file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=time_log_file_path,
            when='midnight',
            interval=1,  # Daily rotation
            backupCount=14,  # Keep logs for 14 days
            encoding='utf-8'
        )
        time_file_handler.setFormatter(log_formatter)
        time_file_handler.setLevel(log_level)
        handlers.append(time_file_handler)
        
        # Separate error log file
        error_log_file_path = os.path.join(log_dir, "pegasus_api_errors.log")
        error_file_handler = logging.handlers.RotatingFileHandler(
            filename=error_log_file_path,
            maxBytes=5242880,  # 5MB
            backupCount=5,
            encoding='utf-8'
        )
        error_file_handler.setFormatter(log_formatter)
        error_file_handler.setLevel(logging.ERROR)  # Only log errors and above
        handlers.append(error_file_handler)
        
        print(f"File logging enabled: {log_file_path}, {time_log_file_path}, {error_log_file_path}")
    except Exception as e:
        print(f"WARNING: Failed to create log file: {e}")
        print("File logging will be disabled. Using console logging only.")

# Configure root logger
logging.basicConfig(level=log_level, handlers=handlers)
logger = logging.getLogger(__name__)
logger.info("Logging initialized")
if use_file_logging:
    logger.info(f"Log file path: {os.path.abspath(log_file_path)}")
logger.info(f"Log level set to: {logging.getLevelName(log_level)}")

# Import debug utilities
try:
    from debug_utils import MemoryProfiler, ThreadDiagnostics, generate_diagnostics_report
    DEBUG_UTILS_AVAILABLE = True
    logger.info("✅ Debug utilities imported successfully")
    
    # Initialize memory profiler
    memory_profiler = MemoryProfiler()
except ImportError as e:
    DEBUG_UTILS_AVAILABLE = False
    logger.warning(f"⚠️ Debug utilities not available: {e}")

# Pydantic models for request/response validation
class SummarizationRequest(BaseModel):
    text: str = Field(..., description="Text to summarize", min_length=10, max_length=50000)
    max_length: Optional[int] = Field(None, description="Maximum length of summary", ge=50, le=1000)
    config: Optional[Dict[str, Any]] = Field(None, description="Custom configuration parameters")

class SummarizationResponse(BaseModel):
    summary: str
    input_length: int
    input_tokens: int
    output_length: int
    output_tokens: int
    processing_time: float
    chunks_processed: int
    model_used: str
    request_id: Optional[str] = None
    memory_usage: Optional[Dict[str, Any]] = None
    success: bool = True

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    gpu_available: bool
    timestamp: str
    memory_usage_mb: float = None
    gpu_memory_info: Dict[str, Any] = None
    cpu_percent: float = None
    uptime_seconds: float = None
    host_info: Dict[str, Any] = None

class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    device: str
    gpu_memory: Optional[str]
    model_parameters: Optional[str]
    supported_tasks: List[str]
    max_input_length: int
    max_output_length: int

class ErrorResponse(BaseModel):
    error: str
    success: bool = False
    timestamp: Optional[str] = None

@dataclass
class SummarizationConfig:
    """Configuration class for summarization parameters"""
    max_input_length: int = 1024
    max_output_length: int = 512
    min_output_length: int = 100
    num_beams: int = 4
    length_penalty: float = 2.0
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3
    diversity_penalty: float = 0.5
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95

class PegasusSummarizer:
    """
    PEGASUS-based document summarizer with fine-tuned model capabilities
    """
    
    def __init__(self, model_path: str = "Zeyad12/pegasus-large-summarizer"):
        """
        Initialize the PEGASUS summarizer
        
        Args:
            model_path (str): Hugging Face model identifier or local path to fine-tuned model
        """
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.config = SummarizationConfig()
        
        logger.info(f"Initializing PEGASUS Summarizer on device: {self.device}")
        self.load_model()

    def load_model(self):
        """Load the PEGASUS model and tokenizer from Hugging Face or fallback alternatives"""
        logger.info(f"Starting model loading process...")
        
        if not PEGASUS_AVAILABLE:
            logger.error("PEGASUS models are not available. Required dependencies missing.")
            raise RuntimeError("PEGASUS models are not available. Please install required dependencies.")
        
        try:
            logger.info(f"Attempting to load model from: {self.model_path}")
            
            # First try to load the specified model (prioritize Hugging Face models)
            if not self.model_path.startswith("./") and not os.path.exists(self.model_path):
                # This is likely a Hugging Face model identifier
                try:
                    logger.info(f"Loading from Hugging Face model hub: {self.model_path}")
                    tokenizer_start = time.time()
                    self.tokenizer = PegasusTokenizer.from_pretrained(self.model_path)
                    tokenizer_time = time.time() - tokenizer_start
                    logger.debug(f"Tokenizer loaded in {tokenizer_time:.2f}s")
                    
                    model_start = time.time()
                    self.model = PegasusForConditionalGeneration.from_pretrained(self.model_path)
                    model_time = time.time() - model_start
                    logger.debug(f"Model loaded in {model_time:.2f}s")
                    
                    device_start = time.time()
                    self.model.to(self.device)
                    device_time = time.time() - device_start
                    logger.debug(f"Model moved to {self.device} in {device_time:.2f}s")
                    
                    self.model.eval()
                    logger.info(f"✅ Successfully loaded Hugging Face model: {self.model_path}")
                    return
                except Exception as hf_error:
                    logger.warning(f"⚠️ Failed to load Hugging Face model {self.model_path}: {hf_error}")
                    logger.info("🔄 Trying alternative approaches...")
            
            # Check if it's a local model path and try to load it
            if os.path.exists(self.model_path):
                logger.info(f"Loading from local path: {self.model_path}")
                # Load tokenizer
                tokenizer_start = time.time()
                self.tokenizer = PegasusTokenizer.from_pretrained(self.model_path)
                tokenizer_time = time.time() - tokenizer_start
                logger.debug(f"Tokenizer loaded in {tokenizer_time:.2f}s")
                
                # Load model
                model_start = time.time()
                self.model = PegasusForConditionalGeneration.from_pretrained(self.model_path)
                model_time = time.time() - model_start
                logger.debug(f"Model loaded in {model_time:.2f}s")
                
                device_start = time.time()
                self.model.to(self.device)
                device_time = time.time() - device_start
                logger.debug(f"Model moved to {self.device} in {device_time:.2f}s")
                
                self.model.eval()
                
                logger.info("✅ Local fine-tuned PEGASUS model loaded successfully!")
                return
            else:
                logger.warning(f"⚠️ Model path not found: {self.model_path}")
            
        except Exception as e:
            logger.error(f"❌ Error loading primary model: {e}")
            logger.debug(f"Error details: {str(e)}", exc_info=True)
            
        # Fallback to alternative models
        logger.info("🔄 Falling back to alternative models...")
        
        # Try different fallback models that don't require SentencePiece
        fallback_models = [
            "facebook/bart-large-cnn",  # BART for summarization
            "sshleifer/distilbart-cnn-12-6",  # Smaller BART variant
            "t5-small",                 # T5 small model
            "google/pegasus-xsum"       # PEGASUS variant (last resort)
        ]
        
        for model_name in fallback_models:
            try:
                logger.info(f"Trying fallback model: {model_name}")
                fallback_start = time.time()
                
                tokenizer_start = time.time()
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer_time = time.time() - tokenizer_start
                logger.debug(f"Fallback tokenizer loaded in {tokenizer_time:.2f}s")
                
                model_start = time.time()
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                model_time = time.time() - model_start
                logger.debug(f"Fallback model loaded in {model_time:.2f}s")
                
                device_start = time.time()
                self.model.to(self.device)
                device_time = time.time() - device_start
                logger.debug(f"Fallback model moved to {self.device} in {device_time:.2f}s")
                
                self.model.eval()
                self.model_path = model_name  # Update the model path
                
                fallback_time = time.time() - fallback_start
                logger.info(f"✅ Fallback model {model_name} loaded successfully in {fallback_time:.2f}s!")
                return
                
            except Exception as fallback_error:
                logger.warning(f"⚠️ Failed to load {model_name}: {fallback_error}")
                logger.debug(f"Fallback error details: {str(fallback_error)}", exc_info=True)
                continue
        
        # If all models fail, raise an error
        logger.error("❌ Failed to load any summarization model after trying all fallbacks")
        raise RuntimeError("Failed to load any summarization model")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text for better summarization
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Preprocessed text
        """
        logger.debug(f"Starting text preprocessing. Original length: {len(text)} chars")
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        logger.debug("Removed excessive whitespace")
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
        logger.debug("Removed special characters")
        
        # Remove very long sequences of repeated characters
        text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)
        logger.debug("Removed long repeated character sequences")
        
        # Ensure proper sentence endings
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        logger.debug("Fixed sentence endings")
        
        processed_text = text.strip()
        logger.debug(f"Text preprocessing complete. Final length: {len(processed_text)} chars")
        return processed_text
    
    def chunk_text(self, text: str, max_chunk_length: int = 800) -> List[str]:
        """
        Split long text into manageable chunks for processing
        
        Args:
            text (str): Input text to chunk
            max_chunk_length (int): Maximum tokens per chunk
            
        Returns:
            List[str]: List of text chunks
        """
        logger.debug(f"Starting text chunking. Text length: {len(text)}, Max chunk length: {max_chunk_length} tokens")
        
        # First, try to split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        logger.debug(f"Split text into {len(sentences)} sentences")
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Estimate token count (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = len(current_chunk + " " + sentence) // 4
            
            if estimated_tokens <= max_chunk_length:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    logger.debug(f"Created chunk of approximately {len(current_chunk) // 4} tokens")
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            logger.debug(f"Created final chunk of approximately {len(current_chunk) // 4} tokens")
        
        # If we still have chunks that are too long, split them further
        final_chunks = []
        for i, chunk in enumerate(chunks):
            if len(chunk) // 4 > max_chunk_length:
                logger.debug(f"Chunk {i+1} is still too long ({len(chunk) // 4} tokens). Splitting further...")
                # Split by character count as last resort
                words = chunk.split()
                current_word_chunk = ""
                
                for word in words:
                    if len(current_word_chunk + " " + word) // 4 <= max_chunk_length:
                        current_word_chunk += " " + word if current_word_chunk else word
                    else:
                        if current_word_chunk:
                            final_chunks.append(current_word_chunk.strip())
                            logger.debug(f"Created sub-chunk of approximately {len(current_word_chunk) // 4} tokens")
                        current_word_chunk = word
                
                if current_word_chunk:
                    final_chunks.append(current_word_chunk.strip())
                    logger.debug(f"Created final sub-chunk of approximately {len(current_word_chunk) // 4} tokens")
            else:
                final_chunks.append(chunk)
        
        logger.debug(f"Text chunking complete. Created {len(final_chunks)} chunks")
        return final_chunks
    
    def summarize_chunk(self, text: str, max_length: int = None) -> str:
        """
        Summarize a single chunk of text
        
        Args:
            text (str): Text chunk to summarize
            max_length (int): Maximum length of summary
            
        Returns:
            str: Generated summary
        """
        if max_length is None:
            max_length = self.config.max_output_length
        
        logger.debug(f"Summarizing chunk of {len(text)} chars with max output length: {max_length}")
        
        # Tokenize input
        tokenize_start = time.time()
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_input_length,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        input_token_count = inputs.input_ids.shape[1]
        tokenize_time = time.time() - tokenize_start
        logger.debug(f"Tokenized input text into {input_token_count} tokens in {tokenize_time:.2f}s")
        
        # Generate summary
        generate_start = time.time()
        with torch.no_grad():
            # Calculate appropriate num_beam_groups for diversity penalty
            num_beam_groups = 1
            if self.config.diversity_penalty > 0:
                num_beam_groups = min(self.config.num_beams // 2, 2)
            
            logger.debug(f"Using beam search with {self.config.num_beams} beams, {num_beam_groups} beam groups")
            
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=max(self.config.min_output_length // 2, 20),
                num_beams=self.config.num_beams,
                length_penalty=self.config.length_penalty,
                early_stopping=self.config.early_stopping,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                diversity_penalty=self.config.diversity_penalty if num_beam_groups > 1 else 0.0,
                num_beam_groups=num_beam_groups,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        output_token_count = outputs.shape[1]
        generate_time = time.time() - generate_start
        logger.debug(f"Generated {output_token_count} output tokens in {generate_time:.2f}s")
        
        # Decode output
        decode_start = time.time()
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        decode_time = time.time() - decode_start
        logger.debug(f"Decoded output in {decode_time:.2f}s. Summary length: {len(summary)} chars")
        
        return summary.strip()
        
    def summarize(self, text: str, max_length: int = None, 
                  custom_config: Optional[Dict] = None, request_id: str = None) -> Dict[str, Any]:
        """
        Main summarization function with comprehensive handling
        
        Args:
            text (str): Input text to summarize
            max_length (int): Maximum length of output summary
            custom_config (Dict): Custom configuration parameters
            request_id (str): Optional request ID for tracing
            
        Returns:
            Dict: Summary results with metadata
        """
        start_time = time.time()
        req_id = request_id or "unknown"
        logger.info(f"[{req_id}] Starting summarization process. Input text length: {len(text)} chars")
        
        # Log memory usage at start
        mem_start = get_memory_usage()
        logger.debug(f"[{req_id}] Initial memory usage: {mem_start['rss_mb']:.2f}MB, {mem_start['percent']:.1f}%")
        
        # Log GPU status if available
        if torch.cuda.is_available():
            gpu_info = get_gpu_info()
            logger.debug(f"[{req_id}] GPU memory allocated: {gpu_info.get('memory_allocated_mb', 'N/A')}MB")
        
        # Update config if custom parameters provided
        if custom_config:
            logger.info(f"[{req_id}] Using custom configuration: {custom_config}")
            for key, value in custom_config.items():
                if hasattr(self.config, key):
                    old_value = getattr(self.config, key)
                    setattr(self.config, key, value)
                    logger.debug(f"[{req_id}] Updated config: {key} = {old_value} -> {value}")
                else:
                    logger.warning(f"[{req_id}] Unknown config parameter: {key}")
        
        # Set default max_length if not provided
        if max_length is None:
            max_length = self.config.max_output_length
            logger.debug(f"[{req_id}] Using default max_length: {max_length}")
        else:
            logger.debug(f"[{req_id}] Using provided max_length: {max_length}")
        
        # Preprocess input text
        preprocess_start = time.time()
        processed_text = self.preprocess_text(text)
        preprocess_time = time.time() - preprocess_start
        logger.debug(f"[{req_id}] Preprocessing completed in {preprocess_time:.2f}s")
        
        # Calculate input metrics
        input_length = len(processed_text)
        input_tokens = len(self.tokenizer.encode(processed_text, truncation=False))
        logger.info(f"[{req_id}] Input metrics: {input_length} chars, approximately {input_tokens} tokens")
        
        # Handle long texts by chunking
        chunk_start = time.time()
        chunks = self.chunk_text(processed_text)
        chunk_time = time.time() - chunk_start
        logger.info(f"[{req_id}] Text split into {len(chunks)} chunks in {chunk_time:.2f}s")
        
        summaries = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"[{req_id}] Processing chunk {i+1}/{len(chunks)} with {len(chunk)} chars")
            chunk_max_length = max_length // len(chunks) if len(chunks) > 1 else max_length
            chunk_max_length = max(chunk_max_length, 50)  # Ensure minimum length
            logger.debug(f"[{req_id}] Adjusted max_length for chunk {i+1}: {chunk_max_length}")
            
            # Check memory before processing chunk
            if i > 0 and i % 3 == 0:  # Check every 3 chunks
                mem_chunk = get_memory_usage()
                logger.debug(f"[{req_id}] Memory after {i} chunks: {mem_chunk['rss_mb']:.2f}MB")
                
                if torch.cuda.is_available():
                    gpu_info = get_gpu_info()
                    logger.debug(f"[{req_id}] GPU memory after {i} chunks: {gpu_info.get('memory_allocated_mb', 'N/A')}MB")
            
            chunk_summary_start = time.time()
            chunk_summary = self.summarize_chunk(chunk, chunk_max_length)
            chunk_summary_time = time.time() - chunk_summary_start
            logger.info(f"[{req_id}] Chunk {i+1} summarized in {chunk_summary_time:.2f}s. Result: {len(chunk_summary)} chars")
            
            summaries.append(chunk_summary)
        
        # Combine summaries
        logger.debug(f"[{req_id}] Combining {len(summaries)} summaries")
        final_summary = " ".join(summaries)
        
        # If we still have multiple summaries and they're still too long, summarize again
        if len(summaries) > 1 and len(final_summary.split()) > max_length:
            logger.info(f"[{req_id}] Combined summary still too long ({len(final_summary.split())} words). Re-summarizing...")
            resummary_start = time.time()
            final_summary = self.summarize_chunk(final_summary, max_length)
            resummary_time = time.time() - resummary_start
            logger.info(f"[{req_id}] Re-summarization completed in {resummary_time:.2f}s")
        
        # Calculate output metrics
        output_length = len(final_summary)
        output_tokens = len(self.tokenizer.encode(final_summary, truncation=False))
        processing_time = time.time() - start_time
        logger.info(f"[{req_id}] Output metrics: {output_length} chars, approximately {output_tokens} tokens")
        
        # Log final memory usage
        mem_end = get_memory_usage()
        mem_diff = mem_end['rss_mb'] - mem_start['rss_mb']
        logger.debug(f"[{req_id}] Final memory usage: {mem_end['rss_mb']:.2f}MB (Δ{mem_diff:+.2f}MB)")
        
        # Run garbage collection if significant memory increase
        if mem_diff > 100:  # More than 100MB increase
            logger.debug(f"[{req_id}] Large memory increase detected. Running garbage collection.")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log memory after cleanup
            mem_after_gc = get_memory_usage()
            gc_diff = mem_after_gc['rss_mb'] - mem_end['rss_mb']
            logger.debug(f"[{req_id}] Memory after GC: {mem_after_gc['rss_mb']:.2f}MB (Δ{gc_diff:+.2f}MB)")
        
        logger.info(f"[{req_id}] Total summarization completed in {processing_time:.2f}s")
        
        return {
            "summary": final_summary,
            "input_length": input_length,
            "input_tokens": input_tokens,
            "output_length": output_length,
            "output_tokens": output_tokens,
            "processing_time": processing_time,
            "chunks_processed": len(chunks),
            "model_used": self.model_path,
            "request_id": req_id,
            "success": True
        }

# Initialize FastAPI app
app = FastAPI(
    title="PEGASUS Summarization API",
    description="A high-performance document summarization API using fine-tuned PEGASUS models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware for request ID and timing
@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    """Middleware to add request ID and track timing for each request"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.start_time = time.time()
    
    # Log request start with memory information
    mem_info = get_memory_usage()
    gpu_info = get_gpu_info()
    logger.info(f"Request {request_id} started | Path: {request.url.path} | "
                f"Memory: {mem_info['rss_mb']:.2f}MB | "
                f"GPU: {gpu_info['gpu_available']}")
    
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - request.state.start_time
    
    # Update memory information
    mem_info_after = get_memory_usage()
    memory_diff = mem_info_after['rss_mb'] - mem_info['rss_mb']
    
    logger.info(f"Request {request_id} completed | Time: {process_time:.2f}s | "
                f"Status: {response.status_code} | "
                f"Memory: {mem_info_after['rss_mb']:.2f}MB (Δ{memory_diff:+.2f}MB)")
    
    return response

# Initialize the summarizer
logger.info("🚀 Starting PEGASUS Summarization API with FastAPI...")
try:
    # Use local model path if running in Docker, otherwise fallback to default
    model_path = os.environ.get('MODEL_PATH', '/models/pegasus-fine-tuned' if os.path.exists('/models/pegasus-fine-tuned') else 'Zeyad12/pegasus-large-summarizer')    
    logger.info(f"Using model path: {model_path}")
    summarizer = PegasusSummarizer(model_path=model_path)
    logger.info("✅ Summarizer initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize summarizer: {e}")
    summarizer = None

# Initialize the resource tracker if the class is available
try:
    if 'ResourceHistoryTracker' in globals():
        resource_tracker = ResourceHistoryTracker(max_history=1000)
        logger.info("✅ Resource tracker initialized successfully")
    else:
        logger.warning("⚠️ ResourceHistoryTracker class not available, skipping initialization")
except Exception as e:
    logger.error(f"❌ Failed to initialize resource tracker: {e}")

# HTML template for the API documentation page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PEGASUS Summarization API - FastAPI</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .endpoint { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .method { background: #3498db; color: white; padding: 3px 8px; border-radius: 3px; font-weight: bold; }
        .post { background: #27ae60; }
        .get { background: #3498db; }
        .code { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }
        .test-form { background: #e8f5e8; padding: 20px; border-radius: 5px; margin: 20px 0; }
        textarea { width: 100%; height: 150px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        button { background: #27ae60; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #229954; }
        .result { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px; }
        .feature { background: #fff3cd; padding: 10px; border-radius: 5px; margin: 5px 0; }
        .badge { background: #28a745; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }
        .tech-badge { background: #e74c3c; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }
        .link-button { display: inline-block; background: #3498db; color: white; padding: 8px 16px; text-decoration: none; border-radius: 5px; margin: 5px; }
        .link-button:hover { background: #2980b9; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 PEGASUS Document Summarization API</h1>
        
        <div class="feature">
            <strong>🔬 Model:</strong> Zeyad12/pegasus-large-summarizer (Hugging Face hosted)
            <span class="badge">PRODUCTION READY</span>
            <span class="tech-badge">FastAPI</span>
        </div>
        
        <div style="margin: 20px 0;">
            <a href="/docs" class="link-button">📖 Interactive API Docs</a>
            <a href="/redoc" class="link-button">📚 ReDoc Documentation</a>
        </div>
        
        <h2>📋 API Endpoints</h2>
        
        <div class="endpoint">
            <span class="method get">GET</span> <strong>/</strong> - API Documentation (this page)
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <strong>/health</strong> - Health check endpoint
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <strong>/summarize</strong> - Generate document summary
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <strong>/model-info</strong> - Get model information and capabilities
        </div>

        <h2>🔧 Usage Examples</h2>
        
        <h3>Basic Summarization Request:</h3>
        <div class="code">
curl -X POST "http://localhost:8003/summarize" \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "Your document text here...",
    "max_length": 200
  }'
        </div>

        <h3>Advanced Configuration:</h3>
        <div class="code">
curl -X POST "http://localhost:8003/summarize" \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "Your document text here...",
    "max_length": 300,
    "config": {
      "num_beams": 6,
      "length_penalty": 1.5,
      "temperature": 0.8
    }
  }'
        </div>

        <h2>🧪 Test the API</h2>
        <div class="test-form">
            <h3>Try Summarization:</h3>
            <textarea id="inputText" placeholder="Enter your document text here...">
The transformer architecture has revolutionized natural language processing by introducing the attention mechanism as the core component. Unlike traditional recurrent neural networks, transformers can process sequences in parallel, leading to significant improvements in training efficiency and model performance. The key innovation lies in the self-attention mechanism, which allows the model to weigh the importance of different words in a sequence when processing each word. This has enabled the development of large-scale language models like BERT, GPT, and T5, which have achieved state-of-the-art results across various NLP tasks including machine translation, text summarization, and question answering.
            </textarea>
            <br><br>
            <label>Max Length: <input type="number" id="maxLength" value="150" min="50" max="500"></label>
            <br><br>
            <button onclick="testSummarization()">Generate Summary</button>
            <div id="result" class="result" style="display: none;"></div>
        </div>

        <h2>📊 FastAPI Features</h2>
        <ul>
            <li><strong>⚡ High Performance:</strong> Built on Starlette and Pydantic for maximum speed</li>
            <li><strong>📋 Automatic Documentation:</strong> Interactive API docs at /docs and /redoc</li>
            <li><strong>🔍 Type Validation:</strong> Automatic request/response validation with Pydantic</li>
            <li><strong>🛡️ Error Handling:</strong> Comprehensive error handling with proper HTTP status codes</li>
            <li><strong>🔧 Easy Testing:</strong> Built-in support for testing and async operations</li>
            <li><strong>📈 Modern Python:</strong> Full support for Python type hints and async/await</li>
        </ul>

        <h2>📖 Response Format</h2>
        <div class="code">
{
  "summary": "Generated summary text...",
  "input_length": 1250,
  "input_tokens": 312,
  "output_length": 180,
  "output_tokens": 45,
  "processing_time": 2.34,
  "chunks_processed": 1,
  "model_used": "Zeyad12/pegasus-large-summarizer",
  "success": true
}
        </div>
    </div>

    <script>
        async function testSummarization() {
            const text = document.getElementById('inputText').value;
            const maxLength = document.getElementById('maxLength').value;
            const resultDiv = document.getElementById('result');
            
            if (!text.trim()) {
                alert('Please enter some text to summarize');
                return;
            }
            
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '<strong>Processing...</strong>';
            
            try {
                const response = await fetch('/summarize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: text,
                        max_length: parseInt(maxLength)
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    resultDiv.innerHTML = `
                        <h4>✅ Summary Generated:</h4>
                        <p><strong>Summary:</strong> ${data.summary}</p>
                        <p><strong>Processing Time:</strong> ${data.processing_time.toFixed(2)}s</p>
                        <p><strong>Input/Output Tokens:</strong> ${data.input_tokens} → ${data.output_tokens}</p>
                        <p><strong>Chunks Processed:</strong> ${data.chunks_processed}</p>
                        <p><strong>Model Used:</strong> ${data.model_used}</p>
                    `;
                } else {
                    resultDiv.innerHTML = `<h4>❌ Error:</h4><p>${data.error}</p>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<h4>❌ Network Error:</h4><p>${error.message}</p>`;
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the API documentation page"""
    return HTMLResponse(content=HTML_TEMPLATE, status_code=200)

@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint for monitoring and debugging"""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"[{request_id}] Health check endpoint requested")
    
    # Get basic system information
    mem_info = get_memory_usage()
    gpu_info = get_gpu_info()
    
    # Calculate approximate uptime
    start_time = getattr(app, "start_time", time.time())
    uptime = time.time() - start_time
    
    # Get host information
    host_info = {
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "docker": os.path.exists("/.dockerenv"),
        "log_level": logging.getLevelName(logger.level),
        "pid": os.getpid()
    }
    
    if summarizer is None:
        logger.error(f"[{request_id}] Health check failed: Summarizer not initialized")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "device": "none",
                "gpu_available": torch.cuda.is_available(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "memory_usage_mb": mem_info["rss_mb"],
                "cpu_percent": psutil.cpu_percent(),
                "uptime_seconds": uptime,
                "host_info": host_info,
                "success": False
            }
        )
    
    logger.debug(f"[{request_id}] Memory usage: {mem_info['rss_mb']:.2f}MB, {mem_info['percent']:.1f}%")
    
    if gpu_info["gpu_available"]:
        logger.debug(f"[{request_id}] GPU info: {gpu_info}")
    
    response = HealthResponse(
        status="healthy",
        model_loaded=summarizer.model is not None,
        device=str(summarizer.device),
        gpu_available=torch.cuda.is_available(),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        memory_usage_mb=mem_info["rss_mb"],
        gpu_memory_info=gpu_info,
        cpu_percent=psutil.cpu_percent(),
        uptime_seconds=uptime,
        host_info=host_info
    )
    
    logger.info(f"[{request_id}] Health check successful: Model loaded={response.model_loaded}, Device={response.device}")
    return response

@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information and capabilities"""
    logger.info("Model info endpoint requested")
    
    if summarizer is None:
        logger.error("Model info request failed: Summarizer not initialized")
        raise HTTPException(status_code=503, detail="Summarizer not initialized")
    
    gpu_memory = None
    model_parameters = None
    
    if torch.cuda.is_available():
        gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        logger.debug(f"GPU memory: {gpu_memory}")
    
    if summarizer.model is not None:
        total_params = sum(p.numel() for p in summarizer.model.parameters())
        model_parameters = f"{total_params / 1000000:.1f}M parameters"
        logger.debug(f"Model parameters: {model_parameters}")
    
    response = ModelInfoResponse(
        model_name=summarizer.model_path,
        model_type="PEGASUS/BART/T5 for Text Summarization",
        device=str(summarizer.device),
        gpu_memory=gpu_memory,
        model_parameters=model_parameters,
        supported_tasks=["Text Summarization", "Document Summarization", "Scientific Paper Summarization"],
        max_input_length=summarizer.config.max_input_length,
        max_output_length=summarizer.config.max_output_length
    )
    
    logger.info(f"Model info served: {response.model_name} on {response.device}")
    return response

@app.post("/summarize", response_model=SummarizationResponse)
async def summarize_text(request: SummarizationRequest):
    """Generate summary for the provided text"""
    request_id = f"req_{int(time.time())}"
    logger.info(f"[{request_id}] Summarization request received. Text length: {len(request.text)} chars")
    
    if summarizer is None:
        logger.error(f"[{request_id}] Summarization failed: Summarizer not initialized")
        raise HTTPException(status_code=503, detail="Summarizer not initialized")
    
    try:
        # Validate input
        if len(request.text.strip()) < 10:
            logger.warning(f"[{request_id}] Validation error: Text too short ({len(request.text.strip())} chars)")
            raise HTTPException(status_code=400, detail="Text must be at least 10 characters long")
        
        if len(request.text) > 50000:
            logger.warning(f"[{request_id}] Validation error: Text too long ({len(request.text)} chars)")
            raise HTTPException(status_code=400, detail="Text too long. Maximum 50,000 characters allowed")
        
        # Log configuration
        if request.max_length:
            logger.info(f"[{request_id}] Requested max_length: {request.max_length}")
        
        if request.config:
            logger.info(f"[{request_id}] Custom config provided: {request.config}")
        
        # Perform summarization
        summarize_start = time.time()
        result = summarizer.summarize(
            text=request.text,
            max_length=request.max_length,
            custom_config=request.config
        )
        summarize_time = time.time() - summarize_start
        
        logger.info(f"[{request_id}] Summarization successful in {summarize_time:.2f}s. Input: {result['input_tokens']} tokens, Output: {result['output_tokens']} tokens")
        
        # Log summary snippet for debugging
        summary_snippet = result['summary'][:100] + "..." if len(result['summary']) > 100 else result['summary']
        logger.debug(f"[{request_id}] Summary snippet: {summary_snippet}")
        
        return SummarizationResponse(**result)
        
    except Exception as e:
        logger.error(f"[{request_id}] Summarization error: {e}")
        logger.debug(f"[{request_id}] Error details:", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/summarize-debug", response_model=SummarizationResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def summarize_with_debug(request: Request, data: SummarizationRequest):
    """
    Generate a summary with enhanced debugging capabilities
    
    Same as the regular summarize endpoint but with additional debugging information
    for troubleshooting issues in Docker container environments.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"[{request_id}] Debug summarization request received. Text length: {len(data.text)} chars")
    
    if summarizer is None:
        logger.error(f"[{request_id}] Summarizer not initialized")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Summarizer not initialized", 
                "success": False, 
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "request_id": request_id
            }
        )
    
    try:
        # Log memory state before processing
        mem_before = get_memory_usage()
        logger.debug(f"[{request_id}] Memory before summarization: {mem_before['rss_mb']:.2f}MB, {mem_before['percent']:.1f}%")
        
        # Log GPU state if available
        gpu_info = get_gpu_info()
        if gpu_info['gpu_available']:
            logger.debug(f"[{request_id}] GPU memory before: Allocated: {gpu_info.get('memory_allocated_mb', 'N/A')}MB, "
                        f"Reserved: {gpu_info.get('memory_reserved_mb', 'N/A')}MB")
        
        # Call the summarization function
        summarization_result = summarizer.summarize(
            text=data.text,
            max_length=data.max_length,
            custom_config=data.config,
            request_id=request_id  # Pass request ID to track through the pipeline
        )
        
        # Log memory state after processing
        mem_after = get_memory_usage()
        memory_diff = mem_after['rss_mb'] - mem_before['rss_mb']
        logger.debug(f"[{request_id}] Memory after summarization: {mem_after['rss_mb']:.2f}MB, "
                    f"Δ{memory_diff:+.2f}MB, {mem_after['percent']:.1f}%")
        
        # Check GPU memory after processing
        if gpu_info['gpu_available']:
            gpu_after = get_gpu_info()
            logger.debug(f"[{request_id}] GPU memory after: Allocated: {gpu_after.get('memory_allocated_mb', 'N/A')}MB, "
                        f"Reserved: {gpu_after.get('memory_reserved_mb', 'N/A')}MB")
        
        # Clean up to help manage memory
        if memory_diff > 100:  # If memory usage increased significantly
            logger.info(f"[{request_id}] Large memory increase detected ({memory_diff:.2f}MB). Running garbage collection.")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log memory after cleanup
            mem_cleanup = get_memory_usage()
            cleanup_diff = mem_cleanup['rss_mb'] - mem_after['rss_mb']
            logger.debug(f"[{request_id}] Memory after cleanup: {mem_cleanup['rss_mb']:.2f}MB, "
                        f"Δ{cleanup_diff:+.2f}MB")
        
        logger.info(f"[{request_id}] Summarization completed successfully. Summary length: {len(summarization_result['summary'])} chars")
        return summarization_result
        
    except Exception as e:
        logger.error(f"[{request_id}] Error in summarization: {str(e)}", exc_info=True)
        
        # Get detailed error context
        error_context = {
            "error_type": type(e).__name__,
            "request_id": request_id,
            "text_length": len(data.text),
            "max_length_requested": data.max_length,
            "memory_usage_mb": get_memory_usage()['rss_mb'],
            "device": str(summarizer.device) if summarizer else "unknown",
            "gpu_available": torch.cuda.is_available()
        }
        
        logger.error(f"[{request_id}] Error context: {json.dumps(error_context)}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error generating summary: {str(e)}",
                "success": False,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "request_id": request_id
            }
        )

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "available_endpoints": ["/", "/health", "/model-info", "/summarize", "/docs", "/redoc"],
            "success": False
        }
    )

@app.exception_handler(405)
async def method_not_allowed_handler(request: Request, exc):
    """Handle 405 errors"""
    return JSONResponse(
        status_code=405,
        content={
            "error": "Method not allowed",
            "success": False
        }
    )

@app.get("/debug/logs", response_class=JSONResponse)
async def get_debug_logs(request: Request, lines: int = 100):
    """
    Get the last N lines of the debug logs
    
    Retrieves the most recent debug logs from the log file for troubleshooting.
    This is especially useful in Docker container environments where accessing
    the log files directly might be difficult.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"[{request_id}] Debug logs requested. Lines: {lines}")
    
    if not use_file_logging:
        return JSONResponse(
            status_code=404,
            content={
                "error": "File logging is not enabled",
                "success": False
            }
        )
    
    try:
        log_entries = []
        
        # Get the log file path from the configuration
        log_file_path = os.path.join(log_dir, "pegasus_api.log")
        
        if not os.path.exists(log_file_path):
            logger.error(f"[{request_id}] Log file not found: {log_file_path}")
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"Log file not found: {log_file_path}",
                    "success": False
                }
            )
        
        # Get file info
        file_size = os.path.getsize(log_file_path)
        file_modified = time.ctime(os.path.getmtime(log_file_path))
        
        # Read the last N lines from the log file
        with open(log_file_path, 'r', encoding='utf-8') as f:
            # Use deque with maxlen to efficiently get the last N lines
            from collections import deque
            last_lines = deque(maxlen=lines)
            for line in f:
                last_lines.append(line.strip())
        
        return JSONResponse(
            content={
                "logs": list(last_lines),
                "file_info": {
                    "path": log_file_path,
                    "size_bytes": file_size,
                    "size_mb": file_size / (1024 * 1024),
                    "last_modified": file_modified
                },
                "log_config": {
                    "log_dir": log_dir,
                    "log_level": logging.getLevelName(logger.level)
                },
                "success": True
            }
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Error reading log file: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error reading log file: {str(e)}",
                "success": False
            }
        )

@app.get("/debug/system", response_class=JSONResponse)
async def get_system_diagnostics(request: Request):
    """
    Get detailed system diagnostics
    
    Provides comprehensive information about the system environment, Docker container,
    Python process, memory usage, GPU information, and other details useful for debugging.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"[{request_id}] System diagnostics requested")
    
    try:
        # Basic system information
        memory_info = get_memory_usage()
        gpu_info = get_gpu_info()
        
        # Get CPU information
        cpu_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_freq": psutil.cpu_freq()._asdict() if hasattr(psutil.cpu_freq(), "_asdict") else None,
        }
        
        # Get Python process information
        process = psutil.Process(os.getpid())
        process_info = {
            "pid": process.pid,
            "name": process.name(),
            "create_time": time.ctime(process.create_time()),
            "status": process.status(),
            "num_threads": process.num_threads(),
            "username": process.username(),
            "memory_percent": process.memory_percent(),
            "memory_info": {k: v / (1024 * 1024) for k, v in process.memory_info()._asdict().items()},  # Convert to MB
            "cpu_percent": process.cpu_percent(interval=0.1),
            "connections": len(process.connections()),
            "open_files": len(process.open_files()) if hasattr(process, "open_files") else None,
        }
        
        # Get Docker information
        docker_info = {
            "in_docker": os.path.exists("/.dockerenv"),
            "container_id": None
        }
        
        try:
            # Try to get container ID if in Docker
            if docker_info["in_docker"]:
                with open("/proc/self/cgroup", "r") as f:
                    docker_info["container_id"] = f.read().split("\n")[0].split("/")[-1]
        except Exception as e:
            docker_info["error"] = str(e)
        
        # Get environment variables (filter out sensitive ones)
        safe_env_vars = {}
        excluded_vars = ["PASSWORD", "SECRET", "KEY", "TOKEN", "CREDENTIAL"]
        
        for key, value in os.environ.items():
            # Skip environment variables that might contain sensitive information
            if any(excluded in key.upper() for excluded in excluded_vars):
                safe_env_vars[key] = "[FILTERED]"
            else:
                safe_env_vars[key] = value
        
        # Get disk usage for the current directory
        disk_info = {}
        try:
            usage = psutil.disk_usage(os.getcwd())
            disk_info = {
                "total_gb": usage.total / (1024**3),
                "used_gb": usage.used / (1024**3),
                "free_gb": usage.free / (1024**3),
                "percent": usage.percent
            }
        except Exception as e:
            disk_info["error"] = str(e)
        
        # Get network information
        network_info = {}
        try:
            network_io = psutil.net_io_counters()
            network_info = {
                "bytes_sent_mb": network_io.bytes_sent / (1024**2),
                "bytes_recv_mb": network_io.bytes_recv / (1024**2),
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv,
            }
        except Exception as e:
            network_info["error"] = str(e)
        
        # Collect system diagnostics
        diagnostics = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request_id": request_id,
            "system": {
                "platform": sys.platform,
                "python_version": sys.version,
                "python_implementation": platform.python_implementation(),
                "hostname": platform.node(),
                "processor": platform.processor()
            },
            "memory": memory_info,
            "cpu": cpu_info,
            "disk": disk_info,
            "network": network_info,
            "process": process_info,
            "docker": docker_info,
            "gpu": gpu_info,
            "logs": {
                "log_dir": log_dir,
                "log_level": logging.getLevelName(logger.level),
                "file_logging_enabled": use_file_logging
            },
            "environment": safe_env_vars,
            "model": {
                "model_path": summarizer.model_path if summarizer else None,
                "device": str(summarizer.device) if summarizer else "none",
                "model_loaded": summarizer is not None and summarizer.model is not None
            }
        }
        
        logger.info(f"[{request_id}] System diagnostics collected successfully")
        return JSONResponse(content=diagnostics)
        
    except Exception as e:
        logger.error(f"[{request_id}] Error collecting system diagnostics: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error collecting system diagnostics: {str(e)}",
                "success": False,
                "request_id": request_id
            }
        )

# Function to get detailed thread diagnostics
def get_thread_diagnostics():
    """Get detailed information about all running threads"""
    thread_info = []
    
    for thread in threading.enumerate():
        thread_data = {
            "name": thread.name,
            "id": thread.ident,
            "daemon": thread.daemon,
            "alive": thread.is_alive()
        }
        
        # Add additional information for main thread
        if thread.name == "MainThread":
            thread_data["main"] = True
        
        thread_info.append(thread_data)
    
    return {
        "active_count": threading.active_count(),
        "threads": thread_info,
        "current_thread": {
            "name": threading.current_thread().name,
            "id": threading.current_thread().ident
        }
    }


# Resource history tracking class
class ResourceHistoryTracker:
    """Tracks historical resource usage for diagnostics"""
    
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.lock = threading.Lock()
        self.tracking_thread = None
        self.tracking_enabled = False
        self.tracking_interval = 60  # Default to 60 seconds
    
    def add_snapshot(self):
        """Add a snapshot of current resource usage"""
        with self.lock:
            try:
                snapshot = {
                    "timestamp": time.time(),
                    "memory": get_memory_usage(),
                    "cpu": {
                        "percent": psutil.cpu_percent(interval=0.1),
                        "count": psutil.cpu_count()
                    },
                    "gpu": get_gpu_info(),
                    "process": {
                        "threads": threading.active_count(),
                        "thread_ids": [t.ident for t in threading.enumerate()]
                    }
                }
                self.history.append(snapshot)
                logger.debug(f"Resource snapshot added. History size: {len(self.history)}")
            except Exception as e:
                logger.error(f"Error adding resource snapshot: {str(e)}")
    
    def start_tracking(self, interval=60):
        """Start periodic resource tracking in the background"""
        if self.tracking_thread and self.tracking_thread.is_alive():
            logger.warning("Resource tracking already running")
            return False
        
        self.tracking_interval = interval
        self.tracking_enabled = True
        
        def tracking_worker():
            logger.info(f"Resource tracking started with interval of {interval}s")
            while self.tracking_enabled:
                try:
                    self.add_snapshot()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in resource tracking thread: {str(e)}")
                    time.sleep(interval)  # Sleep and try again
        
        self.tracking_thread = threading.Thread(
            target=tracking_worker,
            name="ResourceTracker",
            daemon=True
        )
        self.tracking_thread.start()
        return True
    
    def stop_tracking(self):
        """Stop the resource tracking thread"""
        self.tracking_enabled = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=2.0)
            self.tracking_thread = None
            logger.info("Resource tracking stopped")
    
    def get_history(self, limit=None):
        """Get the resource history, optionally limited to the last N entries"""
        with self.lock:
            if limit and limit < len(self.history):
                return list(self.history)[-limit:]
            return list(self.history)
    
    def get_statistics(self):
        """Calculate statistics from the collected history"""
        with self.lock:
            if not self.history:
                return {"error": "No history collected yet"}
            
            stats = {
                "samples": len(self.history),
                "timespan_seconds": None,
                "memory": {
                    "min_mb": float('inf'),
                    "max_mb": 0,
                    "avg_mb": 0,
                    "current_mb": None
                },
                "cpu": {
                    "min_percent": float('inf'),
                    "max_percent": 0,
                    "avg_percent": 0
                }
            }
            
            # Calculate memory and CPU statistics
            total_memory = 0
            total_cpu = 0
            
            for idx, snapshot in enumerate(self.history):
                mem_mb = snapshot["memory"]["rss_mb"]
                cpu_percent = snapshot["cpu"]["percent"]
                
                # Update min/max values
                stats["memory"]["min_mb"] = min(stats["memory"]["min_mb"], mem_mb)
                stats["memory"]["max_mb"] = max(stats["memory"]["max_mb"], mem_mb)
                stats["cpu"]["min_percent"] = min(stats["cpu"]["min_percent"], cpu_percent)
                stats["cpu"]["max_percent"] = max(stats["cpu"]["max_percent"], cpu_percent)
                
                # Accumulate for averages
                total_memory += mem_mb
                total_cpu += cpu_percent
                
                # Set current value (from the latest snapshot)
                if idx == len(self.history) - 1:
                    stats["memory"]["current_mb"] = mem_mb
            
            # Calculate averages
            if self.history:
                stats["memory"]["avg_mb"] = total_memory / len(self.history)
                stats["cpu"]["avg_percent"] = total_cpu / len(self.history)
                
                # Calculate timespan
                if len(self.history) >= 2:
                    start_time = self.history[0]["timestamp"]
                    end_time = self.history[-1]["timestamp"]
                    stats["timespan_seconds"] = end_time - start_time
            
            return stats

# Initialize the resource tracker
resource_tracker = ResourceHistoryTracker(max_history=1000)

@app.on_event("startup")
async def startup_event():
    """Startup event handler for the FastAPI application"""
    logger.info("🚀 Starting PEGASUS Summarization API")
    app.start_time = time.time()
    
    # Start resource tracking if available
    try:
        if 'resource_tracker' in globals() and resource_tracker is not None:
            interval = int(os.environ.get("RESOURCE_TRACKING_INTERVAL", "60"))
            logger.info(f"Starting resource tracking with interval: {interval}s")
            resource_tracker.start_tracking(interval=interval)
        else:
            logger.warning("⚠️ Resource tracker not available, skipping tracking")
    except Exception as e:
        logger.error(f"❌ Failed to start resource tracking: {e}")
    
    # Store a startup snapshot
    try:
        startup_info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "memory": get_memory_usage(),
            "gpu": get_gpu_info(),
            "hostname": platform.node(),
            "platform": sys.platform,
            "python_version": sys.version.split()[0]
        }
        
        # Add thread diagnostics if available
        if 'get_thread_diagnostics' in globals():
            startup_info["threads"] = get_thread_diagnostics()
            logger.info(f"Application startup complete: {startup_info['memory']['rss_mb']:.2f}MB, {startup_info['threads']['active_count']} threads")
        else:
            logger.info(f"Application startup complete: {startup_info['memory']['rss_mb']:.2f}MB")
            
        app.startup_info = startup_info
    except Exception as e:
        logger.error(f"❌ Failed to create startup info: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler for the FastAPI application"""
    logger.info("Shutting down PEGASUS Summarization API")
    
    # Stop resource tracking if available
    try:
        if 'resource_tracker' in globals() and resource_tracker is not None:
            resource_tracker.stop_tracking()
            logger.info("Resource tracking stopped")
    except Exception as e:
        logger.error(f"❌ Error stopping resource tracker: {e}")
    
    # Final memory usage report
    try:
        final_memory = get_memory_usage()
        uptime = time.time() - getattr(app, "start_time", time.time())
        
        logger.info(f"Final memory usage: {final_memory['rss_mb']:.2f}MB")
        logger.info(f"Application uptime: {uptime:.2f}s")
    except Exception as e:
        logger.error(f"❌ Error generating final report: {e}")
    
    logger.info("Shutdown complete")

@app.get("/debug/threads", response_class=JSONResponse)
async def get_thread_info(request: Request):
    """
    Get detailed information about all running threads
    
    This endpoint provides comprehensive information about all active threads
    in the application, which is useful for debugging concurrency issues
    and resource leaks in the Docker container environment.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"[{request_id}] Thread diagnostics requested")
    
    try:
        thread_data = get_thread_diagnostics()
        logger.info(f"[{request_id}] Thread diagnostics collected: {thread_data['active_count']} active threads")
        
        return JSONResponse(
            content={
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "request_id": request_id,
                "thread_diagnostics": thread_data,
                "success": True
            }
        )
    except Exception as e:
        logger.error(f"[{request_id}] Error collecting thread diagnostics: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error collecting thread diagnostics: {str(e)}",
                "success": False,
                "request_id": request_id
            }
        )

@app.get("/debug/resource-history", response_class=JSONResponse)
async def get_resource_history(request: Request, limit: int = 10):
    """
    Get historical resource usage data
    
    This endpoint returns historical resource usage data collected by the
    background resource tracker, which is useful for monitoring memory leaks,
    performance issues, and other resource-related problems in the Docker
    container environment.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"[{request_id}] Resource history requested. Limit: {limit}")
    
    # Check if resource_tracker is available
    if 'resource_tracker' not in globals() or resource_tracker is None:
        logger.warning(f"[{request_id}] Resource tracker not available")
        return JSONResponse(
            status_code=404,
            content={
                "error": "Resource tracker not available",
                "request_id": request_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": False
            }
        )
    
    try:
        history = resource_tracker.get_history(limit=limit)
        stats = resource_tracker.get_statistics()
        
        response_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request_id": request_id,
            "tracker_status": {
                "enabled": resource_tracker.tracking_enabled,
                "interval_seconds": resource_tracker.tracking_interval,
                "max_history": resource_tracker.max_history,
                "current_history_size": len(resource_tracker.history)
            },
            "statistics": stats,
            "history": history,
            "success": True
        }
        
        logger.info(f"[{request_id}] Resource history collected: {len(history)} entries")
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"[{request_id}] Error getting resource history: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error getting resource history: {str(e)}",
                "success": False,
                "request_id": request_id
            }
        )

@app.post("/debug/trigger-gc", response_class=JSONResponse)
async def trigger_garbage_collection(request: Request):
    """
    Manually trigger garbage collection
    
    This endpoint allows manual triggering of Python's garbage collection
    and CUDA memory cache clearing, which can help diagnose and resolve
    memory-related issues in the Docker container environment.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"[{request_id}] Manual garbage collection triggered")
    
    try:
        # Get memory before GC
        mem_before = get_memory_usage()
        logger.info(f"[{request_id}] Memory before GC: {mem_before['rss_mb']:.2f}MB, {mem_before['percent']:.1f}%")
        
        # Get GPU memory before GC
        gpu_before = get_gpu_info()
        if gpu_before["gpu_available"]:
            logger.info(f"[{request_id}] GPU memory before GC: {gpu_before.get('memory_allocated_mb', 'N/A')}MB allocated")
        
        # Run garbage collection
        gc_start = time.time()
        collected = gc.collect(generation=2)  # Full collection
        gc_time = time.time() - gc_start
        
        # Clear CUDA cache if available
        cuda_cleared = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            cuda_cleared = True
        
        # Get memory after GC
        mem_after = get_memory_usage()
        memory_diff = mem_after['rss_mb'] - mem_before['rss_mb']
        
        # Get GPU memory after GC
        gpu_after = None
        gpu_diff = None
        if gpu_before["gpu_available"]:
            gpu_after = get_gpu_info()
            gpu_diff = gpu_after.get('memory_allocated_mb', 0) - gpu_before.get('memory_allocated_mb', 0)
            logger.info(f"[{request_id}] GPU memory after GC: {gpu_after.get('memory_allocated_mb', 'N/A')}MB allocated (Δ{gpu_diff:+.2f}MB)")
        
        logger.info(f"[{request_id}] GC completed in {gc_time:.2f}s, collected {collected} objects")
        logger.info(f"[{request_id}] Memory after GC: {mem_after['rss_mb']:.2f}MB (Δ{memory_diff:+.2f}MB)")
        
        return JSONResponse(
            content={
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "request_id": request_id,
                "gc_results": {
                    "objects_collected": collected,
                    "duration_seconds": gc_time,
                    "cuda_cache_cleared": cuda_cleared,
                    "memory_before_mb": mem_before['rss_mb'],
                    "memory_after_mb": mem_after['rss_mb'],
                    "memory_diff_mb": memory_diff,
                    "gpu_memory_before_mb": gpu_before.get('memory_allocated_mb') if gpu_before["gpu_available"] else None,
                    "gpu_memory_after_mb": gpu_after.get('memory_allocated_mb') if gpu_before["gpu_available"] else None,
                    "gpu_memory_diff_mb": gpu_diff if gpu_before["gpu_available"] else None
                },
                "success": True
            }
        )
    except Exception as e:
        logger.error(f"[{request_id}] Error during garbage collection: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error during garbage collection: {str(e)}",
                "success": False,
                "request_id": request_id
            }
        )

@app.get("/debug/memory-profiler", response_class=JSONResponse)
async def memory_profiler_debug(request: Request, take_snapshot: bool = True, dump_snapshots: bool = False):
    """
    Memory profiler debugging endpoint
    
    This endpoint provides access to the memory profiler functionality,
    which helps diagnose memory leaks and other memory-related issues
    in the Docker container environment.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"[{request_id}] Memory profiler debug requested")
    
    # Check for DEBUG_UTILS_AVAILABLE and memory_profiler
    if not globals().get('DEBUG_UTILS_AVAILABLE', False) or 'memory_profiler' not in globals() or memory_profiler is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Memory profiler not available",
                "debug_utils_available": globals().get('DEBUG_UTILS_AVAILABLE', False),
                "memory_profiler_available": 'memory_profiler' in globals() and memory_profiler is not None,
                "success": False,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "request_id": request_id,
                "basic_memory_info": get_memory_usage()  # At least return basic memory info
            }
        )
    
    try:
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request_id": request_id,
            "debug_utils_available": DEBUG_UTILS_AVAILABLE,
            "success": True
        }
        
        # Take a memory snapshot if requested
        if take_snapshot:
            snapshot = memory_profiler.take_snapshot(label=f"api_request_{request_id}")
            result["snapshot"] = snapshot
            logger.info(f"[{request_id}] Memory snapshot taken: {snapshot['memory']['rss_mb']:.2f}MB")
        
        # Get object type counts
        object_counts = memory_profiler.get_object_types_count(limit=20)
        result["object_counts"] = object_counts
        
        # Dump all snapshots if requested
        if dump_snapshots:
            dump_path = memory_profiler.dump_snapshots()
            result["dump_path"] = dump_path
            logger.info(f"[{request_id}] Memory snapshots dumped to: {dump_path}")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"[{request_id}] Error in memory profiler debug: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error in memory profiler debug: {str(e)}",
                "success": False,
                "request_id": request_id
            }
        )

@app.get("/debug/diagnostics-report", response_class=JSONResponse)
async def generate_full_diagnostics(request: Request):
    """
    Generate a comprehensive diagnostics report
    
    This endpoint generates a detailed diagnostics report with system,
    process, memory, and thread information, which is useful for
    troubleshooting issues in the Docker container environment.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    logger.info(f"[{request_id}] Full diagnostics report requested")
    
    # Check if debug utilities are available
    if not globals().get('DEBUG_UTILS_AVAILABLE', False):
        # Return basic diagnostics even if full diagnostics aren't available
        try:
            basic_diagnostics = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "request_id": request_id,
                "debug_utils_available": False,
                "system_info": {
                    "platform": sys.platform,
                    "python_version": sys.version,
                    "hostname": platform.node() if hasattr(platform, "node") else "unknown"
                },
                "memory_info": get_memory_usage(),
                "gpu_info": get_gpu_info(),
                "process_info": {
                    "pid": os.getpid(),
                    "thread_count": threading.active_count(),
                    "current_thread": threading.current_thread().name
                },
                "success": True
            }
            
            logger.info(f"[{request_id}] Basic diagnostics generated (debug_utils not available)")
            return JSONResponse(content=basic_diagnostics)
        except Exception as e:
            logger.error(f"[{request_id}] Error generating basic diagnostics: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"Error generating basic diagnostics: {str(e)}",
                    "debug_utils_available": False,
                    "success": False,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "request_id": request_id
                }
            )
    
    try:
        # Check if each component is available
        report_components = {}
        
        # Generate the diagnostics report if available
        if 'generate_diagnostics_report' in globals():
            report_path = generate_diagnostics_report()
            report_components["report_path"] = report_path
        else:
            logger.warning(f"[{request_id}] generate_diagnostics_report function not available")
        
        # Dump thread stacks if available
        if 'ThreadDiagnostics' in globals():
            thread_dump = ThreadDiagnostics.dump_thread_stacks()
            report_components["thread_dump_path"] = thread_dump
        else:
            logger.warning(f"[{request_id}] ThreadDiagnostics class not available")
        
        # Take memory snapshot if available
        if 'memory_profiler' in globals() and memory_profiler is not None:
            memory_snapshot = memory_profiler.take_snapshot(label=f"diagnostics_report_{request_id}")
            snapshot_path = memory_profiler.dump_snapshots()
            report_components["memory_snapshot_path"] = snapshot_path
            report_components["memory_snapshot"] = memory_snapshot
        else:
            logger.warning(f"[{request_id}] memory_profiler not available")
          # Add basic diagnostics even if full diagnostics aren't fully available
        basic_diagnostics = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request_id": request_id,
            "debug_utils_available": True,
            "system_info": {
                "platform": sys.platform,
                "python_version": sys.version,
                "hostname": platform.node() if hasattr(platform, "node") else "unknown"
            },
            "memory_info": get_memory_usage(),
            "gpu_info": get_gpu_info(),
            "process_info": {
                "pid": os.getpid(),
                "thread_count": threading.active_count(),
                "current_thread": threading.current_thread().name
            },
            "report_components": report_components,
            "success": True
        }
        
        logger.info(f"[{request_id}] Diagnostics report generated successfully")
        return JSONResponse(content=basic_diagnostics)
        
    except Exception as e:
        logger.error(f"[{request_id}] Error generating diagnostics report: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error generating diagnostics report: {str(e)}",
                "success": False,
                "request_id": request_id
            }
        )
