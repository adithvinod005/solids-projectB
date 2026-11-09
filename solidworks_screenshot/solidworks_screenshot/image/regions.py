"""
image/regions.py

Detects likely UI regions within a SOLIDWORKS screenshot.

V1 goal:
    Identify major visual areas from pixels only.

Current regions:
    - left_ui_panel
    - graphics_area
    - status_bar
    - optional dialog

The left SOLIDWORKS UI column is treated as one physical region.
FeatureManager vs PropertyManager classification is intentionally left
for OCR/vision analysis.

This module does not diagnose problems or suggest solutions.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import (
    FEATURE_TREE_X_RATIO,
    FEATURE_TREE_Y_RATIO,
    GRAPHICS_AREA_X_RATIO,
    GRAPHICS_AREA_Y_RATIO,
    STATUS_BAR_X_RATIO,
    STATUS_BAR_Y_RATIO,
    HEURISTIC_REGION_CONFIDENCE,
)

from models.output_schema import BBox, Region


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _ratio_to_bbox(
    width: int,
    height: int,
    x_ratio: tuple[float, float],
    y_ratio: tuple[float, float],
) -> BBox:

    x_min = int(round(x_ratio[0] * width))
    x_max = int(round(x_ratio[1] * width))

    y_min = int(round(y_ratio[0] * height))
    y_max = int(round(y_ratio[1] * height))

    return BBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )


def _crop(
    image: np.ndarray,
    bbox: BBox,
) -> np.ndarray:

    height, width = image.shape[:2]

    x_min = max(0, min(width, bbox.x_min))
    x_max = max(0, min(width, bbox.x_max))

    y_min = max(0, min(height, bbox.y_min))
    y_max = max(0, min(height, bbox.y_max))

    if x_max <= x_min or y_max <= y_min:
        return np.empty((0, 0), dtype=image.dtype)

    return image[
        y_min:y_max,
        x_min:x_max,
    ]


def _edge_density(
    image_region: np.ndarray,
) -> float:

    if image_region.size == 0:
        return 0.0

    if image_region.ndim == 3:

        gray = cv2.cvtColor(
            image_region,
            cv2.COLOR_BGR2GRAY,
        )

    else:

        gray = image_region

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    return (
        float(np.count_nonzero(edges))
        / float(edges.size)
    )


# ---------------------------------------------------------------------------
# Left UI panel
# ---------------------------------------------------------------------------

def detect_left_ui_panel(
    image: np.ndarray,
) -> Region:

    height, width = image.shape[:2]

    bbox = _ratio_to_bbox(
        width,
        height,
        FEATURE_TREE_X_RATIO,
        FEATURE_TREE_Y_RATIO,
    )

    crop = _crop(
        image,
        bbox,
    )

    density = _edge_density(
        crop
    )

    confidence = HEURISTIC_REGION_CONFIDENCE

    if density > 0.05:

        confidence = min(
            0.9,
            confidence + 0.3,
        )

    elif density < 0.01:

        confidence = max(
            0.1,
            confidence - 0.3,
        )

    return Region(
        name="left_ui_panel",
        bbox=bbox,
        confidence=confidence,
        detection_method="heuristic",
    )


# ---------------------------------------------------------------------------
# Graphics area
# ---------------------------------------------------------------------------

def detect_graphics_area_region(
    image: np.ndarray,
) -> Region:

    height, width = image.shape[:2]

    bbox = _ratio_to_bbox(
        width,
        height,
        GRAPHICS_AREA_X_RATIO,
        GRAPHICS_AREA_Y_RATIO,
    )

    return Region(
        name="graphics_area",
        bbox=bbox,
        confidence=0.7,
        detection_method="heuristic",
    )


# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------

def detect_status_bar_region(
    image: np.ndarray,
) -> Region:

    height, width = image.shape[:2]

    bbox = _ratio_to_bbox(
        width,
        height,
        STATUS_BAR_X_RATIO,
        STATUS_BAR_Y_RATIO,
    )

    return Region(
        name="status_bar",
        bbox=bbox,
        confidence=HEURISTIC_REGION_CONFIDENCE,
        detection_method="heuristic",
    )


# ---------------------------------------------------------------------------
# Dialog detection
# ---------------------------------------------------------------------------

def detect_dialog_region(
    image: np.ndarray,
) -> Region | None:
    """
    Detect a likely floating SOLIDWORKS dialog.

    This version is deliberately conservative.

    The previous detector incorrectly identified a large bright portion
    of the graphics area as a dialog. A real dialog should generally:

        1. Be reasonably centered.
        2. Have a substantial but not enormous area.
        3. Have a clear rectangular boundary.
        4. Have a reasonable width/height ratio.
        5. Not extend almost to the screen edge.
        6. Not be an extremely tall narrow strip.

    If those conditions are not met, no dialog is returned.
    """

    height, width = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # ---------------------------------------------------------------
    # Find bright regions.
    # ---------------------------------------------------------------

    _, bright_mask = cv2.threshold(
        gray,
        210,
        255,
        cv2.THRESH_BINARY,
    )

    # Close small gaps so a dialog background is more likely to form
    # one connected component.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (7, 7),
    )

    bright_mask = cv2.morphologyEx(
        bright_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    contours, _ = cv2.findContours(
        bright_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best_candidate: BBox | None = None
    best_score = 0.0

    image_area = float(
        width * height
    )

    image_center_x = width / 2.0
    image_center_y = height / 2.0

    # ---------------------------------------------------------------
    # Evaluate candidates.
    # ---------------------------------------------------------------

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = w * h

        area_ratio = (
            area / image_area
        )

        # Reject tiny areas and huge screen-sized areas.
        if area_ratio < 0.05:
            continue

        if area_ratio > 0.35:
            continue

        # Reject extremely narrow shapes.
        aspect_ratio = w / float(h)

        if aspect_ratio < 0.5:
            continue

        if aspect_ratio > 3.5:
            continue

        # -----------------------------------------------------------
        # Reject candidates touching screen boundaries.
        # -----------------------------------------------------------

        margin = 10

        touches_screen_edge = (
            x <= margin
            or y <= margin
            or x + w >= width - margin
            or y + h >= height - margin
        )

        if touches_screen_edge:
            continue

        # -----------------------------------------------------------
        # Calculate distance from screen center.
        # -----------------------------------------------------------

        candidate_center_x = (
            x + w / 2.0
        )

        candidate_center_y = (
            y + h / 2.0
        )

        distance_x = abs(
            candidate_center_x
            - image_center_x
        )

        distance_y = abs(
            candidate_center_y
            - image_center_y
        )

        normalized_distance = (
            distance_x / width
            + distance_y / height
        )

        # A floating dialog should normally be reasonably near the
        # center of the application window.
        if normalized_distance > 0.65:
            continue

        # -----------------------------------------------------------
        # Check rectangularity.
        # -----------------------------------------------------------

        contour_area = cv2.contourArea(
            contour
        )

        if contour_area <= 0:
            continue

        rectangularity = (
            contour_area / float(area)
        )

        # Very irregular bright regions are unlikely to be dialogs.
        if rectangularity < 0.55:
            continue

        # -----------------------------------------------------------
        # Score candidate.
        # -----------------------------------------------------------

        center_score = max(
            0.0,
            1.0
            - normalized_distance,
        )

        size_score = min(
            1.0,
            area_ratio / 0.20,
        )

        score = (
            0.5 * center_score
            + 0.3 * rectangularity
            + 0.2 * size_score
        )

        if score > best_score:

            best_score = score

            best_candidate = BBox(
                x_min=x,
                y_min=y,
                x_max=x + w,
                y_max=y + h,
            )

    # ---------------------------------------------------------------
    # Require a strong candidate.
    # ---------------------------------------------------------------

    if (
        best_candidate is None
        or best_score < 0.65
    ):

        return None

    return Region(
        name="dialog",
        bbox=best_candidate,
        confidence=round(
            min(
                0.9,
                best_score,
            ),
            3,
        ),
        detection_method="heuristic",
    )


# ---------------------------------------------------------------------------
# Main region detector
# ---------------------------------------------------------------------------

def detect_all_regions(
    image: np.ndarray,
) -> list[Region]:
    """
    Detect all major SOLIDWORKS screenshot regions.

    Current V1 regions:

        left_ui_panel
        graphics_area
        status_bar
        optional dialog
    """

    regions = [
        detect_left_ui_panel(
            image
        ),

        detect_graphics_area_region(
            image
        ),

        detect_status_bar_region(
            image
        ),
    ]

    dialog = detect_dialog_region(
        image
    )

    if dialog is not None:

        regions.append(
            dialog
        )

    return regions