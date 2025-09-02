from models_loading.detection_loader import load_model
from models_loading.ocr_loader import load_ocr_model
from pdf_processing.convert_pdf_to_images import convert_pdf_to_images
from pdf_processing.extract_layout import extract_elements, crop_regions
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image


def process_pdf(pdf_path: str):
    full_text = ""
    tables = []
    figures = []

    try:
        # Convert PDF to images
        images = convert_pdf_to_images(pdf_path)
        print(f"Total pages: {len(images)}")

        # Load models
        detection_model = load_model()
        ocr_model = load_ocr_model()

        for page_idx, image in enumerate(images):
            try:
                print(f"\n=== Processing Page {page_idx + 1} ===")
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                elements = extract_elements(
                    image_cv,
                    detection_model,
                    target_types=("Text", "Title", "List", "Table", "Figure")
                )

                if not elements:
                    print(f"No elements detected on page {page_idx + 1}")
                    continue

                crops = crop_regions(image_cv, elements, padding=5)

                for elem_idx, (element, crop) in enumerate(zip(elements, crops)):
                    elem_type = element['type']

                    try:
                        if crop is None or crop.size == 0:
                            print(f"Empty crop for element {elem_idx + 1}, skipping.")
                            continue

                        if elem_type in ["Text", "Title", "List"]:
                            processed_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            ocr_result = ocr_model.ocr(processed_crop, cls=True)

                            text = ""
                            if ocr_result and ocr_result[0]:
                                for line in ocr_result[0]:
                                    text_line = line[1][0]
                                    text += text_line + " "

                            full_text += clean_multiline_text(text) + "\n"

                        elif elem_type == "Table":
                            tables.append({
                                "image": crop_to_base64(crop)
                            })

                        elif elem_type == "Figure":
                            figures.append({
                                "image": crop_to_base64(crop)
                            })

                    except Exception as e:
                        print(f"Failed processing element {elem_idx + 1} ({elem_type}): {str(e)}")
                        continue

            except Exception as e:
                print(f"Error processing page {page_idx + 1}: {str(e)}")
                continue

    except Exception as e:
        print(f"Processing failed: {str(e)}")

    return {
        "full_text": full_text.strip(),
        "tables": tables,
        "figures": figures
    }


def clean_multiline_text(text):
    lines = text.split("\n")
    return " ".join(line.strip() for line in lines if line.strip())


def crop_to_base64(crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")
