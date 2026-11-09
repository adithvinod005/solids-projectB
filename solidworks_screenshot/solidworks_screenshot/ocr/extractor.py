"""
ocr/extractor.py

Runs OCR (via Tesseract, through the pytesseract wrapper) over the full
screenshot and/or specific detected regions, producing OcrResult objects.

DESIGN NOTES:
- OCR is treated as optional infrastructure, not a hard dependency of the
  whole pipeline. If Tesseract isn't installed on the machine, this module
  reports that clearly and returns an empty result list rather than
  crashing the whole program - screenshot loading, preprocessing, and
  region detection should still work and produce a JSON output.
- We run OCR per-region rather than only on the whole image, because
  cropping to a specific area (like just the status bar strip) tends to
  give Tesseract a much easier, more accurate job than the entire
  screenshot at once.
"""

from __future__ import annotations

import numpy as np

from config import OCR_MIN_CONFIDENCE
from models.output_schema import BBox, OcrResult, Region

try:
    import pytesseract
    from pytesseract import Output as _TesseractOutput

    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class OcrUnavailableError(Exception):
    """Raised when OCR is requested but Tesseract/pytesseract isn't available."""


def is_ocr_available() -> bool:
    """
    Check whether OCR can actually run on this machine.

    This checks both that the `pytesseract` Python package is importable
    AND that it can successfully find the underlying Tesseract binary
    (pytesseract raises its own error if the binary isn't on PATH).
    """
    if not TESSERACT_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _extract_from_crop(
    crop: np.ndarray, offset_x: int, offset_y: int, region_name: str | None
) -> list[OcrResult]:
    """
    Run Tesseract on a single image crop and translate its output into
    OcrResult objects with bounding boxes in ORIGINAL image coordinates
    (i.e. offset back by the crop's position).
    """
    if crop.size == 0:
        return []

    data = pytesseract.image_to_data(crop, output_type=_TesseractOutput.DICT)

    results: list[OcrResult] = []
    num_boxes = len(data["text"])

    for i in range(num_boxes):
        text = data["text"][i].strip()
        confidence = float(data["conf"][i])

        # Tesseract uses -1 for boxes it couldn't assign a confidence to
        # (usually whitespace/non-text detections) - skip those, along
        # with anything below our configured minimum confidence.
        if not text or confidence < OCR_MIN_CONFIDENCE:
            continue

        x = data["left"][i] + offset_x
        y = data["top"][i] + offset_y
        w = data["width"][i]
        h = data["height"][i]

        bbox = BBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h)
        results.append(
            OcrResult(text=text, confidence=confidence, bbox=bbox, region_name=region_name)
        )

    return results


def extract_text_from_regions(
    ocr_ready_image: np.ndarray, regions: list[Region]
) -> list[OcrResult]:
    """
    Run OCR on each detected region individually.

    `ocr_ready_image` should be the enhanced grayscale image produced by
    image/preprocessing.py's `enhance_for_ocr`, at the SAME scale as the
    region bounding boxes (i.e. computed from the resized image).

    Returns an empty list (rather than raising) if OCR isn't available,
    since the rest of the pipeline should still be able to run and produce
    partial results.
    """
    if not is_ocr_available():
        return []

    all_results: list[OcrResult] = []

    for region in regions:
        bbox = region.bbox
        crop = ocr_ready_image[bbox.y_min:bbox.y_max, bbox.x_min:bbox.x_max]
        results = _extract_from_crop(crop, offset_x=bbox.x_min, offset_y=bbox.y_min, region_name=region.name)
        all_results.extend(results)

    return all_results


def extract_text_full_image(ocr_ready_image: np.ndarray) -> list[OcrResult]:
    """
    Run OCR across the entire image without restricting to specific
    regions. Useful as a fallback, or when region detection didn't find
    anything useful.
    """
    if not is_ocr_available():
        return []

    return _extract_from_crop(ocr_ready_image, offset_x=0, offset_y=0, region_name=None)
