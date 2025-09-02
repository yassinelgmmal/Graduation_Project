from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from PIL import Image
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
import os
from huggingface_hub import login
import io

app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

HF_TOKEN = os.getenv("HF_TOKEN", "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")
login(token=HF_TOKEN)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

proc_dir = "paligemma/processor"
base_id = "google/paligemma-3b-ft-scicap-224"
adapter_local = "./paligemma-finetuned-caption"

if os.listdir(proc_dir):
    processor = PaliGemmaProcessor.from_pretrained(proc_dir)
else:
    processor = PaliGemmaProcessor.from_pretrained(base_id, token=HF_TOKEN)
    processor.save_pretrained(proc_dir)

base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    base_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map=device if device == "cuda" else None,
    token=HF_TOKEN,
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_local,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map=device if device == "cuda" else None,
    local_files_only=True,
)


def caption_single_image(image: Image.Image) -> str:
    image = image.convert("RGB")

    prompt = "caption"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            min_length=50,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            num_beams=1,
            early_stopping=False,
        )

    caption = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    return caption.replace(prompt, "").strip() if prompt in caption else caption.strip()

@app.post("/captions")
async def generate_captions(files: List[UploadFile] = File(...)):
    results = {}
    for file in files:
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            caption = caption_single_image(image)
            results[file.filename] = caption
        except Exception as e:
            results[file.filename] = f"Error: {str(e)}"
    return JSONResponse(content={"captions": results})

# Run with: uvicorn app:app --host localhost --port 8010