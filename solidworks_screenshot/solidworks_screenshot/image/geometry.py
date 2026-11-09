"""
image/geometry.py

V1 visual geometry analysis for SOLIDWORKS screenshots.

Purpose:
    Analyze the graphics area at the pixel level and extract
    geometry-related visual observations.

This module DOES NOT:
    - diagnose the final SOLIDWORKS problem
    - suggest solutions
    - recommend fixes
    - modify the screenshot

It only produces visual evidence that a future AI model can use.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np


@dataclass
class GeometryObservation:
    """
    Pixel-level observations from the SOLIDWORKS graphics area.
    """

    graphics_width: int
    graphics_height: int

    edge_pixel_ratio: float
    dark_pixel_ratio: float

    connected_component_count: int
    large_component_count: int

    geometry_bbox: list[int] | None

    horizontal_line_count: int
    vertical_line_count: int
    diagonal_line_count: int

    visual_geometry_detected: bool

    detection_method: str = "opencv_heuristic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# SAFE CROP
# ============================================================

def _safe_crop(
    image: np.ndarray,
    bbox: list[int],
) -> np.ndarray:
    """
    Crop an image using:

        [x_min, y_min, x_max, y_max]

    Coordinates are clipped to image boundaries.
    """

    height, width = image.shape[:2]

    x_min, y_min, x_max, y_max = bbox

    x_min = max(0, min(x_min, width))
    x_max = max(0, min(x_max, width))

    y_min = max(0, min(y_min, height))
    y_max = max(0, min(y_max, height))

    if x_max <= x_min or y_max <= y_min:
        return image[0:0, 0:0]

    return image[
        y_min:y_max,
        x_min:x_max,
    ]


# ============================================================
# EDGE DETECTION
# ============================================================

def _detect_edges(
    gray: np.ndarray,
) -> np.ndarray:
    """
    Detect visible edges in the graphics area.

    This is a basic pixel-level detector and does not attempt
    to understand actual CAD topology.
    """

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    edges = cv2.Canny(
        blurred,
        threshold1=50,
        threshold2=150,
    )

    return edges


# ============================================================
# DARK PIXEL RATIO
# ============================================================

def _detect_dark_pixels(
    gray: np.ndarray,
) -> float:
    """
    Estimate the proportion of dark pixels.
    """

    dark_threshold = 100

    dark_pixels = (
        gray < dark_threshold
    )

    return float(
        np.mean(dark_pixels)
    )


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

def _find_connected_components(
    binary_image: np.ndarray,
) -> tuple[int, list[int]]:
    """
    Find connected components in a binary image.

    Returns:

        component_count
        component_areas
    """

    num_labels, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary_image,
            connectivity=8,
        )
    )

    if num_labels <= 1:
        return 0, []

    areas = stats[
        1:,
        cv2.CC_STAT_AREA,
    ]

    return (
        num_labels - 1,
        [
            int(area)
            for area in areas
        ],
    )


# ============================================================
# GEOMETRY BOUNDING BOX
# ============================================================

def _find_geometry_bbox(
    binary_image: np.ndarray,
) -> list[int] | None:
    """
    Find the bounding box of significant visual structures.

    Small components are ignored to reduce the influence of
    isolated pixels and image noise.
    """

    num_labels, _, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary_image,
            connectivity=8,
        )
    )

    if num_labels <= 1:
        return None

    image_height, image_width = (
        binary_image.shape[:2]
    )

    min_area = max(
        100,
        int(
            image_width
            * image_height
            * 0.0002
        ),
    )

    boxes = []

    for label in range(
        1,
        num_labels,
    ):

        x = int(
            stats[
                label,
                cv2.CC_STAT_LEFT,
            ]
        )

        y = int(
            stats[
                label,
                cv2.CC_STAT_TOP,
            ]
        )

        width = int(
            stats[
                label,
                cv2.CC_STAT_WIDTH,
            ]
        )

        height = int(
            stats[
                label,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < min_area:
            continue

        boxes.append(
            (
                x,
                y,
                x + width,
                y + height,
            )
        )

    if not boxes:
        return None

    x_min = min(
        box[0]
        for box in boxes
    )

    y_min = min(
        box[1]
        for box in boxes
    )

    x_max = max(
        box[2]
        for box in boxes
    )

    y_max = max(
        box[3]
        for box in boxes
    )

    return [
        x_min,
        y_min,
        x_max,
        y_max,
    ]


# ============================================================
# LINE CLASSIFICATION
# ============================================================

def _classify_lines(
    edges: np.ndarray,
) -> tuple[int, int, int]:
    """
    Detect approximately:

        horizontal lines
        vertical lines
        diagonal lines

    This is only pixel-level evidence.

    It does NOT claim that a detected line is a specific
    SOLIDWORKS sketch entity or model edge.
    """

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=30,
        maxLineGap=8,
    )

    if lines is None:
        return (
            0,
            0,
            0,
        )

    horizontal = 0
    vertical = 0
    diagonal = 0

    for line in lines:

        # ----------------------------------------------------
        # HoughLinesP may return different array shapes
        # depending on the OpenCV build.
        #
        # Flatten the line safely instead of assuming:
        #
        #     line[0] == [x1, y1, x2, y2]
        # ----------------------------------------------------

        values = np.asarray(
            line
        ).reshape(-1)

        if values.size != 4:
            continue

        x1, y1, x2, y2 = (
            values
        )

        dx = x2 - x1
        dy = y2 - y1

        # Ignore zero-length detections.
        if dx == 0 and dy == 0:
            continue

        angle = np.degrees(
            np.arctan2(
                dy,
                dx,
            )
        )

        angle = abs(angle)

        if angle > 90:
            angle = 180 - angle

        # ----------------------------------------------------
        # Approximately horizontal
        # ----------------------------------------------------

        if angle <= 10:

            horizontal += 1

        # ----------------------------------------------------
        # Approximately vertical
        # ----------------------------------------------------

        elif abs(angle - 90) <= 10:

            vertical += 1

        # ----------------------------------------------------
        # Diagonal / other orientation
        # ----------------------------------------------------

        else:

            diagonal += 1

    return (
        horizontal,
        vertical,
        diagonal,
    )


# ============================================================
# MAIN GRAPHICS ANALYSIS
# ============================================================

def analyze_graphics_area(
    image: np.ndarray,
    graphics_bbox: list[int],
) -> GeometryObservation:
    """
    Analyze the SOLIDWORKS graphics area.

    Args:
        image:
            Resized BGR screenshot.

        graphics_bbox:
            Graphics region:

                [x_min, y_min, x_max, y_max]

    Returns:
        GeometryObservation
    """

    # --------------------------------------------------------
    # Crop graphics area
    # --------------------------------------------------------

    graphics = _safe_crop(
        image,
        graphics_bbox,
    )

    # --------------------------------------------------------
    # Handle invalid crop
    # --------------------------------------------------------

    if graphics.size == 0:

        return GeometryObservation(
            graphics_width=0,
            graphics_height=0,
            edge_pixel_ratio=0.0,
            dark_pixel_ratio=0.0,
            connected_component_count=0,
            large_component_count=0,
            geometry_bbox=None,
            horizontal_line_count=0,
            vertical_line_count=0,
            diagonal_line_count=0,
            visual_geometry_detected=False,
        )

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        graphics,
        cv2.COLOR_BGR2GRAY,
    )

    # --------------------------------------------------------
    # Edge detection
    # --------------------------------------------------------

    edges = _detect_edges(
        gray
    )

    edge_pixel_ratio = float(
        np.mean(
            edges > 0
        )
    )

    # --------------------------------------------------------
    # Dark pixel analysis
    # --------------------------------------------------------

    dark_pixel_ratio = (
        _detect_dark_pixels(
            gray
        )
    )

    # --------------------------------------------------------
    # Convert edges to binary image
    # --------------------------------------------------------

    binary = (
        (edges > 0)
        .astype(np.uint8)
        * 255
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    (
        component_count,
        areas,
    ) = _find_connected_components(
        binary
    )

    image_area = (
        graphics.shape[0]
        * graphics.shape[1]
    )

    large_component_threshold = max(
        200,
        int(
            image_area
            * 0.0005
        ),
    )

    large_component_count = sum(
        area >= large_component_threshold
        for area in areas
    )

    # --------------------------------------------------------
    # Geometry bounding box
    # --------------------------------------------------------

    geometry_bbox = (
        _find_geometry_bbox(
            binary
        )
    )

    # --------------------------------------------------------
    # Line classification
    # --------------------------------------------------------

    (
        horizontal_line_count,
        vertical_line_count,
        diagonal_line_count,
    ) = _classify_lines(
        edges
    )

    # --------------------------------------------------------
    # Conservative geometry detection
    # --------------------------------------------------------

    total_lines = (
        horizontal_line_count
        + vertical_line_count
        + diagonal_line_count
    )

    visual_geometry_detected = (
        edge_pixel_ratio > 0.002
        and (
            large_component_count >= 1
            or total_lines >= 3
        )
    )

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    return GeometryObservation(
        graphics_width=int(
            graphics.shape[1]
        ),
        graphics_height=int(
            graphics.shape[0]
        ),
        edge_pixel_ratio=round(
            edge_pixel_ratio,
            4,
        ),
        dark_pixel_ratio=round(
            dark_pixel_ratio,
            4,
        ),
        connected_component_count=(
            component_count
        ),
        large_component_count=(
            large_component_count
        ),
        geometry_bbox=(
            geometry_bbox
        ),
        horizontal_line_count=(
            horizontal_line_count
        ),
        vertical_line_count=(
            vertical_line_count
        ),
        diagonal_line_count=(
            diagonal_line_count
        ),
        visual_geometry_detected=(
            visual_geometry_detected
        ),
    )


# ============================================================
# REGION WRAPPER
# ============================================================

def analyze_graphics_area_from_region(
    image: np.ndarray,
    region: Any,
) -> GeometryObservation:
    """
    Convenience wrapper for a Region object.

    Supports:

        region.bbox.as_list()

    or a BBox object with:

        x_min
        y_min
        x_max
        y_max
    """

    bbox = region.bbox

    if isinstance(
        bbox,
        list,
    ):

        coordinates = bbox

    elif hasattr(
        bbox,
        "as_list",
    ):

        coordinates = (
            bbox.as_list()
        )

    else:

        coordinates = [
            int(bbox.x_min),
            int(bbox.y_min),
            int(bbox.x_max),
            int(bbox.y_max),
        ]

    return analyze_graphics_area(
        image,
        coordinates,
    )