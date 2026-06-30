"""
preprocess.py
OpenCV-based preprocessing pipeline to clean up a product label image
before it is passed to the OCR engine.
Real product-label photos (glare, curved packaging, uneven lighting,
busy backgrounds) tend to break a naive "grayscale -> global Otsu
threshold" pipeline: Otsu picks ONE global brightness cutoff, so if
lighting varies across the label it can blow out half the text to pure
white or pure black. This pipeline instead:
    1. Resizes to a standard width (keeping aspect ratio) for speed/accuracy,
       upscaling small photos so text has enough pixels to OCR.
    2. Converts to grayscale.
    3. Boosts local contrast with CLAHE (adaptive histogram equalization),
       which compensates for glare/shadow instead of a single global level.
    4. Lightly denoises (small median blur) to remove sensor noise without
       smearing thin text strokes.
    5. Binarizes with *adaptive* thresholding (a local neighborhood mean),
       which copes with uneven lighting far better than global Otsu.
    6. Sharpens edges slightly to crisp up character strokes.
If the binarized result still looks unusable, `run_ocr` can be pointed at
the grayscale (non-binarized) version instead - see `preprocess_image`'s
`return_grayscale_fallback` option.
"""
import cv2
import numpy as np
STANDARD_WIDTH = 1600
MIN_USABLE_WIDTH = 800
def resize_image(image: np.ndarray, width: int = STANDARD_WIDTH) -> np.ndarray:
    """Resize image to a standard width, preserving aspect ratio.
    Upscales small/low-res photos (common with phone camera crops) so
    Tesseract has enough pixel detail per character, and downsizes very
    large photos for speed.
    """
    h, w = image.shape[:2]
    if w == width:
        return image
    scale = width / float(w)
    new_dim = (width, max(1, int(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, new_dim, interpolation=interpolation)
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE (adaptive contrast enhancement) to even out lighting
    differences across the label (glare on one side, shadow on the other)."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(image)
def denoise(image: np.ndarray) -> np.ndarray:
    """Remove sensor/compression noise with a light median blur. A small
    kernel is used so thin character strokes survive."""
    return cv2.medianBlur(image, 3)
def sharpen(image: np.ndarray) -> np.ndarray:
    """Lightly sharpen edges via an unsharp-mask style kernel, to crisp
    up character strokes that got softened by resizing/blurring."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
def threshold(image: np.ndarray) -> np.ndarray:
    """Binarize using adaptive (local-mean) thresholding rather than a
    single global Otsu cutoff, so uneven lighting across the label doesn't
    wash out half the text."""
    block_size = 35 if min(image.shape[:2]) > 200 else 15
    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        10,
    )
def preprocess_image(image_path: str, save_path: str = None,
                      return_grayscale_fallback: bool = False) -> np.ndarray:
    """
    Run the full preprocessing pipeline on an image file.
    Args:
        image_path: path to the source image.
        save_path: optional path to write the processed image to disk.
        return_grayscale_fallback: if True, return the contrast-enhanced
            grayscale image (skipping binarization) instead of the
            thresholded black/white image. Some photos OCR better as
            clean grayscale than as a forced binary image - useful as a
            fallback path if extraction quality is poor.
    Returns:
        The processed image as a numpy array.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    image = resize_image(image)
    gray = to_grayscale(image)
    contrasted = enhance_contrast(gray)
    denoised = denoise(contrasted)
    sharpened = sharpen(denoised)
    if return_grayscale_fallback:
        result = sharpened
    else:
        result = threshold(sharpened)
    if save_path:
        cv2.imwrite(save_path, result)
    return result
