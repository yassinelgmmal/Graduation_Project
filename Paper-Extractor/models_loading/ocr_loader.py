import os
import layoutparser as lp
from paddleocr import PaddleOCR
from config import (
    OCR_DET_MODEL_DIR,
    OCR_REC_MODEL_DIR,
    DEVICE
)

def load_ocr_model(det_model_dir: str = OCR_DET_MODEL_DIR, rec_model_dir: str = OCR_REC_MODEL_DIR) -> PaddleOCR:
    return PaddleOCR(
        det_model_dir=det_model_dir,
        rec_model_dir=rec_model_dir,
        cls_model_dir=None,
        use_angle_cls=True,
        lang='en',
        device=DEVICE

    )
