import fitz  # PyMuPDF
from PIL import Image

def convert_pdf_to_images(pdf_path, dpi=300):
    images = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        print(f"Converted {len(images)} pages to images")
        return images
    except Exception as e:
        print(f"PDF conversion error: {str(e)}")
        raise