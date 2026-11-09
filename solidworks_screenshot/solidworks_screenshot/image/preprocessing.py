"""
image/preprocessing.py

Basic, generic image cleanup that happens BEFORE region detection or OCR.

This module deliberately does not know anything about SOLIDWORKS UI
layout - that logic lives in regions.py. This file only answers:
"is this image a reasonable size, and is it easy to read text/shapes from?"
"""

from __future__ import annotations

import cv2
import numpy as np

from config import MAX_IMAGE_DIMENSION


def resize_if_needed(
    image: np.ndarray, max_dimension: int = MAX_IMAGE_DIMENSION
) -> tuple[np.ndarray, bool]:
    """
    Downscale the image if its longest side exceeds `max_dimension`.

    Screenshots can come from very high-resolution monitors; keeping a
    consistent working size makes region-ratio heuristics and OCR speed
    predictable. Aspect ratio is always preserved.

    Returns (resized_image, was_resized).
    """
    height, width = image.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_dimension:
        return image, False

    scale = max_dimension / longest_side
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, True


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Produce a version of the image optimized for text readability.

    SOLIDWORKS UI text is usually small and anti-aliased against light
    backgrounds. A mild contrast boost (CLAHE) followed by a slight sharpen
    tends to help OCR engines without introducing artifacts the way
    aggressive thresholding can.

    Returns a grayscale image - OCR engines generally don't need color.
    """
    gray = to_grayscale(image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)

    # Light sharpening kernel - subtle on purpose. Over-sharpening can
    # introduce ringing artifacts that hurt OCR more than it helps.
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, sharpen_kernel)

    return sharpened


def preprocess_screenshot(image: np.ndarray) -> dict[str, np.ndarray]:
    """
    Run the standard preprocessing pipeline on a raw screenshot.

    Returns a dict with multiple useful variants, since different later
    stages want different things:
        - "resized":     the (possibly downscaled) original color image,
                          used as the base for region detection and cropping
        - "ocr_ready":    grayscale + contrast/sharpen enhanced, used for OCR
        - "was_resized":  bool flag for the output schema

    Keeping these variants separate (rather than mutating one image in
    place) means each downstream stage always gets the representation
    that suits it best.
    """
    resized, was_resized = resize_if_needed(image)
    ocr_ready = enhance_for_ocr(resized)

    return {
        "resized": resized,
        "ocr_ready": ocr_ready,
        "was_resized": was_resized,
    }
