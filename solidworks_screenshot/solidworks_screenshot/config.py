"""
config.py

Central place for paths and tunable settings.

Keeping these values in one file means the rest of the codebase never
hardcodes a folder name or a magic threshold - if you need to change
how sensitive region detection is, or where output files land, you
only need to edit this file.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
SCREENSHOTS_DIR: Path = PROJECT_ROOT / "screenshots"

DEFAULT_OUTPUT_FILENAME: str = "analysis.json"

# ---------------------------------------------------------------------------
# SolidWorks C# API export
# ---------------------------------------------------------------------------

# The C# analyzer (SolidWorksV1) lives in a completely separate project
# folder from this Python pipeline, so this can't be derived relative to
# PROJECT_ROOT - it's hardcoded to where SolidWorksV1's ApiJsonExporter.cs
# currently writes analysis_api.json.
#
# If the C# project folder ever moves, this is the one line to update.
SOLIDWORKS_API_DATA_PATH: Path = Path(
    r"C:\Users\Admin\OneDrive\Desktop\solids proj\SolidWorksV1\analysis_api.json"
)

# Folder containing SolidWorksV1.csproj - used to auto-run the C#
# analyzer (`dotnet run --project <this>`) before reading
# analysis_api.json, so Python always gets fresh data for whatever's
# currently open in SolidWorks without a separate manual step.
SOLIDWORKS_CSHARP_PROJECT_DIR: Path = (
    SOLIDWORKS_API_DATA_PATH.parent
)

# ---------------------------------------------------------------------------
# Image handling
# ---------------------------------------------------------------------------

# Accepted image file extensions (lowercase, without the dot).
SUPPORTED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

# If a screenshot is larger than this on its longest side, it gets resized
# down before processing. This keeps OCR and region detection fast and
# consistent regardless of the user's monitor resolution.
MAX_IMAGE_DIMENSION: int = 1920

# ---------------------------------------------------------------------------
# Region detection (image/regions.py)
# ---------------------------------------------------------------------------

# SOLIDWORKS' default UI layout is reasonably consistent, so we detect
# regions primarily as *proportions* of the overall window rather than
# fixed pixel values. These are starting heuristics, not guarantees -
# every detected region carries a confidence score rather than being
# treated as certain.

# Feature tree / FeatureManager typically occupies the left strip of the
# window, spanning almost the full height.
FEATURE_TREE_X_RATIO = (0.0, 0.18)
FEATURE_TREE_Y_RATIO = (0.05, 0.95)

# PropertyManager panel appears in roughly the same left-hand column as the
# feature tree, but only while a feature/sketch is being edited. Since it
# overlaps the tree's screen position, we distinguish between them using
# simple visual cues (see image/regions.py) rather than position alone.
PROPERTY_MANAGER_X_RATIO = (0.0, 0.18)
PROPERTY_MANAGER_Y_RATIO = (0.05, 0.95)

# The graphics/model area occupies the large central region.
GRAPHICS_AREA_X_RATIO = (0.18, 1.0)
GRAPHICS_AREA_Y_RATIO = (0.05, 0.90)

# The status bar sits along the bottom edge of the window.
STATUS_BAR_X_RATIO = (0.0, 1.0)
STATUS_BAR_Y_RATIO = (0.95, 1.0)

# Confidence assigned to purely position-based heuristic regions.
# Kept deliberately modest since no real visual verification occurs yet.
HEURISTIC_REGION_CONFIDENCE = 0.5

# ---------------------------------------------------------------------------
# OCR (ocr/extractor.py)
# ---------------------------------------------------------------------------

# Minimum confidence (0-100, Tesseract's native scale) for a piece of OCR
# text to be included in the output at all.
OCR_MIN_CONFIDENCE = 40

# Keywords that, if found in OCR text, strongly suggest an error/warning
# message is visible. Used only to help set the `error_message_visible`
# flag - never to fabricate error text that wasn't actually read.
ERROR_KEYWORDS = [
    "error",
    "rebuild error",
    "what's wrong",
    "failed",
    "invalid",
    "warning",
]