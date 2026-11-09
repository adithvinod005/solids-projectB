"""
image/loader.py

Responsible for one thing only: getting a screenshot off disk and into a
validated, in-memory image (a NumPy array in BGR order, the format OpenCV
uses everywhere else in this project).

Keeping loading/validation separate from preprocessing means:
- errors about "this isn't a real image" surface immediately and clearly
- preprocessing.py can assume it always receives a valid image
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config import SUPPORTED_IMAGE_EXTENSIONS


class ImageLoadError(Exception):
    """Raised when a screenshot file can't be found, read, or decoded."""


def validate_image_path(path: str | Path) -> Path:
    """
    Check that a path points to a file we can plausibly load as an image.

    Raises ImageLoadError with a clear message if not. Returns the resolved
    Path on success so callers don't need to re-resolve it.
    """
    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        raise ImageLoadError(f"File not found: {resolved}")

    if not resolved.is_file():
        raise ImageLoadError(f"Path is not a file: {resolved}")

    extension = resolved.suffix.lower().lstrip(".")
    if extension not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ImageLoadError(
            f"Unsupported file extension '.{extension}'. Supported: {supported}"
        )

    return resolved


def load_image(path: str | Path) -> np.ndarray:
    """
    Load a screenshot from disk as a BGR NumPy array (OpenCV's native format).

    Raises ImageLoadError if the file doesn't exist, isn't a supported
    extension, or can't actually be decoded as an image (e.g. corrupted
    file, or a renamed non-image file).
    """
    resolved = validate_image_path(path)

    # cv2.imread returns None on failure rather than raising - we convert
    # that into an explicit exception so failures aren't silent.
    image = cv2.imread(str(resolved))

    if image is None:
        raise ImageLoadError(
            f"Could not decode image (file may be corrupted or not a real "
            f"image): {resolved}"
        )

    return image


def get_image_dimensions(image: np.ndarray) -> tuple[int, int]:
    """Return (width, height) for a loaded image. OpenCV stores shape as (h, w, channels)."""
    height, width = image.shape[:2]
    return width, height
