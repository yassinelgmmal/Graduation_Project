import cv2
import numpy as np
from typing import List
from paddleocr import PaddleOCR


def recognize_text_from_crops(crops: List[np.ndarray], ocr_model: PaddleOCR) -> List[str]:
    """
    Run OCR on image crops and return recognized texts.
    
    Parameters:
        crops (List[np.ndarray]): List of cropped image regions (in BGR format).
        ocr_model (PaddleOCR): Initialized PaddleOCR model.
    
    Returns:
        List[str]: Recognized texts from each crop.
    """
    recognized_texts = []

    for idx, crop in enumerate(crops):
        try:
            if crop is None or crop.size == 0:
                print(f"Skipping empty crop at index {idx}")
                recognized_texts.append("")
                continue

            # Convert BGR to RGB
            processed_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Run OCR with recognition only
            result = ocr_model.ocr(processed_crop, det=False, rec=True, cls=True)

            text = ""
            if result and result[0]:
                for line in result[0]:
                    text_line = line[1][0]  # format: [bbox, (text, confidence)]
                    text += text_line + " "

            recognized_texts.append(text.strip())

        except Exception as e:
            print(f"OCR failed at crop {idx}: {str(e)}")
            recognized_texts.append("")

    return recognized_texts
