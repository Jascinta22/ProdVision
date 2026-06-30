"""
ocr.py
Thin wrapper around Pytesseract. Configures the Tesseract executable path,
runs OCR on a preprocessed image, and returns both the raw extracted text
and the average word-level confidence score reported by Tesseract.
"""

import os
import platform
import numpy as np
import pytesseract
from pytesseract import Output

# --- Tesseract path configuration -----------------------------------------
# On Windows, Tesseract is typically installed via:
#   winget install UB-Mannheim.TesseractOCR --silent --accept-package-agreements --accept-source-agreements
# which places the binary at the path below by default. On Linux/Mac it is
# usually already on PATH after `apt install tesseract-ocr` / `brew install tesseract`.
if platform.system() == "Windows":
    default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_path):
        pytesseract.pytesseract.tesseract_cmd = default_path

# Allow overriding via environment variable for flexibility/deployment.
custom_path = os.environ.get("TESSERACT_CMD")
if custom_path:
    pytesseract.pytesseract.tesseract_cmd = custom_path


def run_ocr(image: np.ndarray, lang: str = "eng") -> dict:
    """
    Run OCR on a preprocessed (numpy array) image.

    Returns:
        dict with keys:
            "text": full extracted text (str)
            "confidence": average word confidence, 0-100 (float)
    """
    try:
        raw_text = pytesseract.image_to_string(image, lang=lang)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR engine was not found. Install it (see README) and/or "
            "set the TESSERACT_CMD environment variable to the executable path."
        ) from exc

    data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT)
    confidences = [int(c) for c in data.get("conf", []) if c not in ("-1", -1)]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "text": raw_text.strip(),
        "confidence": round(avg_confidence, 2),
    }
