import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
from typing import List, Dict


def extract_text_pymupdf(path: str) -> List[Dict[str, str]]:
    """
    Extracts text from PDF pages using PyMuPDF.

    Args:
        path: Path to PDF file.

    Returns:
        List of dicts with keys 'page' and 'text'.
    """
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = re.sub(r'\n{2,}', '\n', text).strip()
        pages.append({"page": i + 1, "text": text})
    return pages


def extract_images_from_page(page) -> List[bytes]:
    """
    Extracts all images from a PDF page.

    Args:
        page: PyMuPDF page object.

    Returns:
        List of image bytes.
    """
    images = []
    for img in page.get_images(full=True):
        xref = img[0]
        base_image = page.parent.extract_image(xref)
        images.append(base_image["image"])
    return images


def ocr_page_image(image_bytes: bytes, lang: str = "eng") -> str:
    """
    Runs OCR on an image.

    Args:
        image_bytes: Image in bytes.
        lang: Language for Tesseract OCR.

    Returns:
        Extracted text from image.
    """
    img = Image.open(io.BytesIO(image_bytes))
    txt = pytesseract.image_to_string(img, lang=lang)
    txt = re.sub(r'\n{2,}', '\n', txt).strip()
    return txt


def extract_with_ocr_if_needed(path: str, ocr_threshold: int = 40) -> List[Dict[str, str]]:
    """
    Extracts text from PDF pages, using OCR if page text is too short.

    Args:
        path: Path to PDF file.
        ocr_threshold: Minimum text length before OCR is applied.

    Returns:
        List of dicts, each containing 'page' and 'text'.
    """
    pages = extract_text_pymupdf(path)
    doc = fitz.open(path)
    result_pages = []

    for i, page in enumerate(doc):
        text = pages[i]["text"]

        if len(text.strip()) < ocr_threshold:
            # Attempt OCR from images
            imgs = extract_images_from_page(page)
            ocr_texts = []
            for img_bytes in imgs:
                try:
                    ocr_texts.append(ocr_page_image(img_bytes))
                except Exception:
                    pass

            # Fallback to page rasterization
            if not ocr_texts:
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes()
                try:
                    ocr_texts.append(ocr_page_image(img_bytes))
                except Exception:
                    pass

            text = "\n".join(ocr_texts).strip() or text

        result_pages.append({"page": i + 1, "text": text})

    return result_pages
