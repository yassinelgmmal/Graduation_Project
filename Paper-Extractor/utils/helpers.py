import os
import cv2

def save_crops(crops, folder, prefix):
    os.makedirs(folder, exist_ok=True)
    for idx, crop in enumerate(crops):
        path = os.path.join(folder, f"{prefix}_{idx + 1}.png")
        cv2.imwrite(path, crop)