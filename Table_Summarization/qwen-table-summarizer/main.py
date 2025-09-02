from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from PIL import Image
import torch
import io
import gc
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen Table Summarizer", version="1.0")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize pipe as None to lazy-load it
pipe = None

def load_model():
    global pipe
    logger.info("Starting model loading...")
    try:
        pipe = pipeline(
            "image-text-to-text",
            model="Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

# --- Utilities ---

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()

def preprocess_image(image_bytes, max_size=(400, 400)):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(max_size, Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file")

# --- Routes ---

@app.on_event("startup")
def startup_event():
    logger.info("Starting application...")
    if torch.cuda.is_available():
        logger.info(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("⚠️ GPU NOT available")
    
    # We don't load the model at startup anymore - it will be loaded on demand

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"message": "Qwen2 Table Summarizer API is running"}

@app.get("/health")
def health():
    global pipe
    
    model_status = "not_loaded"
    device = "unknown"
    
    if pipe is not None:
        model_status = "loaded"
        device = str(pipe.device)
    
    health_info = {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "model_status": model_status,
        "device": device,
        "model": "Qwen2-VL-2B-Instruct"
    }
    
    logger.info(f"Health check: {health_info}")
    return health_info

@app.post("/summarize_table/")
async def summarize_table(file: UploadFile = File(...)):
    global pipe
    
    try:
        # Load model if not already loaded
        if pipe is None:
            logger.info("Model not loaded yet. Loading now...")
            success = load_model()
            if not success:
                return JSONResponse(
                    status_code=500, 
                    content={"error": "Failed to load model. Please check logs."}
                )
        
        logger.info(f"Processing file: {file.filename}")
        
        # Validate image
        if not file.content_type.startswith("image/"):
            logger.warning(f"Invalid content type: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()
        image = preprocess_image(contents)
        logger.info("Image preprocessed successfully")

        # Prompt
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": (
                        "Summarize the key data and insights from this scientific table clearly and concisely. "
                        "Use bullet points or numbered lists to highlight important numbers and trends. "
                        "Keep it precise and easy to understand."
                    )}
                ]
            }
        ]

        logger.info("Running inference...")
        # Run inference
        try:
            with torch.inference_mode():
                result = pipe(prompt, max_new_tokens=400)
            
            # Safely extract response
            generated = result[0].get("generated_text", [])
            summary = next(
                (msg.get("content") for msg in generated if msg.get("role") == "assistant"),
                "No summary generated."
            )
            
            logger.info("Inference completed successfully")
            return JSONResponse(content={"summary": summary})
            
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            return JSONResponse(
                status_code=500, 
                content={"error": f"Model inference failed: {str(e)}"}
            )

    except Exception as e:
        logger.error(f"General error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        clear_memory()
        logger.info("Memory cleared")
