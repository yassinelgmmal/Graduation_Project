import os
import torch

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Detectron2 model configs
DETECTRON2_CONFIG_PATH = os.path.join(BASE_DIR, "models", "config.yaml")
DETECTRON2_MODEL_PATH = os.path.join(BASE_DIR, "models", "model_final.pth")

# PaddleOCR model directories
OCR_DET_MODEL_DIR = os.path.join(BASE_DIR, "models", "en_PP-OCRv3_det_infer")
OCR_REC_MODEL_DIR = os.path.join(BASE_DIR, "models", "en_PP-OCRv3_rec_infer")

# Other settings
# In containerized environments, prefer CPU for stability unless explicitly configured for GPU
import os
if os.getenv('FORCE_CPU', 'false').lower() == 'true':
    DEVICE = "cpu"
else:
    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Torch is using: {'cuda' if DEVICE == 'gpu' else 'cpu'}")
if DEVICE == "gpu":
    if torch.cuda.is_available():
        print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] GPU requested but CUDA is not available. Falling back to CPU.")
        DEVICE = "cpu"
else:
    print("[INFO] Running on CPU mode.")

SCORE_THRESHOLD = 0.8

# Label map for layout detection
LABEL_MAP = {
    0: "Text",
    1: "Title",
    2: "List",
    3: "Table",
    4: "Figure"
}
