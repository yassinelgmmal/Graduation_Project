from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
from contextlib import asynccontextmanager
import torch.nn.functional as F
from fastapi.middleware.cors import CORSMiddleware

# Path to your Hugging Face model folder
MODEL_PATH = r"merged_model_final"

# Dictionary to store the model, tokenizer, and label map
ml_models = {}

# Request body schema
class TextInput(BaseModel):
    text: str

# Lifespan event to handle startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Attempting to load model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model path not found: {MODEL_PATH}")

    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()

        # Load label map
        label_map_path = os.path.join(MODEL_PATH, "label_map.json")
        if not os.path.exists(label_map_path):
            raise RuntimeError("Missing label_map.json in model directory")

        with open(label_map_path, "r", encoding="utf-8") as f:
            raw_label_map = json.load(f)

        if "id2label" not in raw_label_map:
            raise RuntimeError("label_map.json must contain 'id2label' key.")

        label_map = {int(k): v for k, v in raw_label_map["id2label"].items()}

        # Store in global dictionary
        ml_models["tokenizer"] = tokenizer
        ml_models["model"] = model
        ml_models["label_map"] = label_map

        print("✅ Model, tokenizer, and label map loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Startup failed: {e}")

    yield

    # Cleanup
    ml_models.clear()
    print("✅ Resources released on shutdown.")

# Initialize FastAPI app
app = FastAPI(
    title="Paper Classification API",
    description="Classifies scientific paper abstracts into academic categories.",
    version="1.2.0",
    lifespan=lifespan
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Paper Classification API!"}

# Prediction endpoint
@app.post("/predict")
def predict(data: TextInput):
    if "tokenizer" not in ml_models or "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        tokenizer = ml_models["tokenizer"]
        model = ml_models["model"]
        label_map = ml_models["label_map"]

        # Tokenize the input text
        inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True)
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get prediction
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        pred_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_index].item()
        predicted_label = label_map.get(pred_index, str(pred_index))

        return {
            "input_text": data.text,
            "predicted_label": predicted_label,
            "predicted_class_index": pred_index,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
