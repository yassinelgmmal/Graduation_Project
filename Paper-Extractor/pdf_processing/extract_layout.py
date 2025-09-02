import cv2
import numpy as np
from typing import List, Dict
import layoutparser as lp


def debug_model_output(raw_output):
    print("\n=== DEBUGGING MODEL OUTPUT ===")
    print(f"Type: {type(raw_output)}")
    if isinstance(raw_output, lp.Layout):
        print(f"Number of elements detected: {len(raw_output)}")
        for element in raw_output:
            print(f"Element type: {element.type}, coordinates: {element.coordinates}, score: {element.score}")
    else:
        print("Unexpected output format")


def extract_elements(image, model, target_types=("Table", "Figure")) -> List[Dict]:
    raw_output = model.detect(image)

    debug_model_output(raw_output)

    elements = []
    for block in raw_output:
        try:
            if block.type in target_types and block.score >= 0.8:
                x1, y1, x2, y2 = map(int, block.coordinates)
                elements.append({
                    'type': block.type,
                    'score': float(block.score),
                    'coordinates': [x1, y1, x2, y2]
                })
        except Exception as e:
            print(f"Skipping malformed block: {str(e)}")
            continue

    return elements



def crop_regions(image: np.ndarray, elements: List[Dict], padding: int = 5, extra_padding: int = 7) -> List[np.ndarray]:
    """
    Crop regions from the image based on provided bounding boxes.
    Extra padding added to prevent skipping of the last letter during OCR.
    """
    crops = []
    h, w, _ = image.shape

    for idx, elem in enumerate(elements):
        try:
            # Extract coordinates for the crop
            x1, y1, x2, y2 = elem['coordinates']
            
            # Apply padding with extra padding for edge cases
            x1 = max(0, int(x1) - padding - extra_padding)
            y1 = max(0, int(y1) - padding - extra_padding)
            x2 = min(w, int(x2) + padding + extra_padding)
            y2 = min(h, int(y2) + padding + extra_padding)

            # Skip too small crops (adjust threshold as needed)
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                print(f"Skipping too small crop at index {idx}")
                continue

            # Crop the region from the image
            crop = image[y1:y2, x1:x2]
            crops.append(crop)

        except Exception as e:
            print(f"Skipping invalid crop at index {idx}: {str(e)}")
    
    return crops
