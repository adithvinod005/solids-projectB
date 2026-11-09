"""
main.py

SOLIDWORKS Screenshot Analysis - V1

Pipeline:

    Screenshot
        ↓
    Image loading
        ↓
    Preprocessing
        ↓
    Region detection
        ↓
    OCR
        ↓
    Visual geometry analysis
        ↓
    Problem evidence
        ↓
    Optional vision model
        ↓
    JSON output

V1 SCOPE:
    Screenshot -> problem identification information

This screenshot pipeline does NOT generate:
    - solutions
    - fixes
    - recommendations
    - corrective actions
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config import (
    DEFAULT_OUTPUT_FILENAME,
    OUTPUT_DIR,
    SCREENSHOTS_DIR,
    SOLIDWORKS_API_DATA_PATH,
    SOLIDWORKS_CSHARP_PROJECT_DIR,
)

from image.loader import load_image
from image.preprocessing import preprocess_screenshot
from image.regions import detect_all_regions
from image.geometry import analyze_graphics_area

from ocr.extractor import extract_text_from_regions

from models.output_schema import (
    ImageInfo,
    ProblemEvidence,
    ScreenshotAnalysis,
    VisualObservations,
)

from vision.analyzer import (
    NullVisionModel,
    analyze_screenshot,
)
from vision.groq_client import (
    GroqVisionModel,
    GroqVisionModelError,
)
from vision.solution_generator import (
    SolutionGeneratorError,
    generate_solution,
)
from report_generator import generate_html_report


# ============================================================
# SOLIDWORKS API DATA
# ============================================================

# How long to wait for the C# analyzer before giving up and falling
# back to whatever analysis_api.json already exists. Generous, since
# SolidWorks itself can take a moment to respond over COM.
SOLIDWORKS_ANALYZER_TIMEOUT_SECONDS = 120


def run_solidworks_analyzer() -> bool:
    """
    Run the C# SolidWorks analyzer (`dotnet run`) against whatever
    document is currently open in SolidWorks, so analysis_api.json is
    always fresh before this pipeline reads it.

    The C# program ends with a "Press ENTER to exit..." prompt meant
    for interactive use - sending it a newline on stdin here lets it
    exit cleanly without needing to change that working C# code.

    Returns True if the analyzer ran (regardless of whether it found
    an active SolidWorks document), False if it couldn't be run at all
    (dotnet missing, project folder missing, timeout). Either way, the
    pipeline continues afterward using whatever analysis_api.json
    exists - a failed refresh here is not fatal, just means stale data.
    """

    if not SOLIDWORKS_CSHARP_PROJECT_DIR.exists():
        print(
            f"SolidWorks C# project folder not found, skipping "
            f"auto-refresh: {SOLIDWORKS_CSHARP_PROJECT_DIR}"
        )
        return False

    print(
        f"Refreshing SolidWorks API data (running C# analyzer "
        f"in {SOLIDWORKS_CSHARP_PROJECT_DIR})..."
    )

    try:
        result = subprocess.run(
            [
                "dotnet",
                "run",
                "--project",
                str(SOLIDWORKS_CSHARP_PROJECT_DIR),
            ],
            cwd=str(SOLIDWORKS_CSHARP_PROJECT_DIR),
            input="\n",
            capture_output=True,
            text=True,
            timeout=SOLIDWORKS_ANALYZER_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        print(
            "Could not run 'dotnet' - is the .NET SDK installed "
            "and on PATH? Continuing with existing analysis_api.json."
        )
        return False
    except subprocess.TimeoutExpired:
        print(
            f"C# analyzer did not finish within "
            f"{SOLIDWORKS_ANALYZER_TIMEOUT_SECONDS}s, giving up on "
            f"this refresh. Continuing with existing "
            f"analysis_api.json."
        )
        return False

    if result.stdout:
        print(result.stdout.strip())

    if result.returncode != 0:
        print(
            f"C# analyzer exited with code {result.returncode}. "
            f"Continuing with existing analysis_api.json."
        )
        if result.stderr:
            print(result.stderr.strip())
        return False

    return True


def load_solidworks_api_data(
    path: Path = SOLIDWORKS_API_DATA_PATH,
) -> Optional[dict[str, Any]]:
    """
    Load the SolidWorks C# API export (analysis_api.json) as a plain
    dict, ready to attach to ScreenshotAnalysis.solidworks_api_data.

    Returns None (rather than raising) if the file doesn't exist yet
    or can't be parsed - the screenshot pipeline should still produce
    a valid analysis.json even before the C# side has been run for
    this part.
    """

    if not path.exists():
        print(f"No SolidWorks API data found at: {path}")
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not load SolidWorks API data ({path}): {exc}")
        return None


# ============================================================
# DIAGNOSIS SUMMARY
# ============================================================

# Human-readable phrasing for each problem_type returned by
# _classify_problem_type(). Deliberately describes WHAT was
# observed, never what to do about it - no fix/action language.
PROBLEM_TYPE_DESCRIPTIONS = {
    "cut_not_intersect_model": (
        "does not intersect the model"
    ),
    "missing_reference": (
        "is missing a required face or plane reference"
    ),
    "geometric_conditions": (
        "failed due to invalid geometric conditions"
    ),
    "rebuild_error": (
        "failed to rebuild"
    ),
    "invalid_parameters": (
        "has invalid parameters for the selected geometry"
    ),
    "tangent_propagation": (
        "has a tangent propagation issue"
    ),
}


def _build_diagnosis_summary(
    evidence: ProblemEvidence,
) -> str | None:
    """
    Turn the structured ProblemEvidence fields into a single
    readable sentence describing what was observed.

    This is a synthesis of fields already extracted elsewhere in
    this file (problem_type, primary_feature, related_features) -
    it does not read OCR text again or infer anything new. It
    states what appears to be wrong; it never suggests what to
    do about it.
    """

    if not evidence.problem_visible:
        return None

    if evidence.primary_feature:
        feature_label = (
            evidence.primary_feature[0].upper()
            + evidence.primary_feature[1:]
        )
        subject = f"{feature_label} feature"
    else:
        subject = "A feature"

    description = PROBLEM_TYPE_DESCRIPTIONS.get(
        evidence.problem_type
    )

    if description is not None:
        sentence = f"{subject} {description}."
    else:
        sentence = (
            f"{subject} shows signs of a problem, but the "
            f"exact type could not be classified from the "
            f"screenshot text."
        )

    if evidence.related_features:
        related = ", ".join(evidence.related_features)
        sentence += f" Related feature(s): {related}."

    return sentence


# ============================================================
# PROBLEM PHRASE DETECTION
# ============================================================

PROBLEM_PHRASES = {
    "cut_not_intersect_model": [
        "cut does not intersect the model",
        "does not intersect the model",
        "not intersect the model",
        "intended cut does not intersect",
        "cut not intersect",
    ],

    "missing_reference": [
        "could not find face",
        "could not find plane",
        "cannot find face",
        "cannot find plane",
        "missing face",
        "missing plane",
        "find face or plane",
    ],

    "rebuild_error": [
        "rebuild error",
        "failed to rebuild",
        "failed to build",
        "not build",
        "cannot rebuild",
        "could not rebuild",
    ],

    "geometric_conditions": [
        "geometric conditions",
        "geometry conditions",
        "due to geometry conditions",
        "due to geometric conditions",
    ],

    "invalid_parameters": [
        "selected geometry and parameters",
        "selected geometry",
        "invalid parameter",
        "invalid parameters",
    ],

    "tangent_propagation": [
        "tangent propagation",
    ],

    "generic_error": [
        "error",
        "failed",
        "invalid",
    ],

    "generic_warning": [
        "warning",
    ],
}


# ============================================================
# FEATURE NAME DETECTION
# ============================================================

FEATURE_NAME_PATTERNS = [
    ("cut-extrude", r"\bcut[-\s]?extrud\w*\b"),
    ("boss-extrude", r"\bboss[-\s]?extrud\w*\b"),
    ("chamfer", r"\bchamfer\w*\b"),
    ("fillet", r"\bfillet\w*\b"),
    ("sketch", r"\bsketch\d*\b"),
    ("extrude", r"\bextrud\w*\b"),
    ("revolve", r"\brevolv\w*\b"),
    ("loft", r"\bloft\w*\b"),
    ("sweep", r"\bsweep\w*\b"),
    ("hole", r"\bhole\w*\b"),
    ("pattern", r"\bpattern\w*\b"),
    ("mirror", r"\bmirror\w*\b"),
]


# ============================================================
# PARAMETER DETECTION
# ============================================================

PARAMETER_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m|deg|°|in)\b",
    re.IGNORECASE,
)


# ============================================================
# TEXT UTILITIES
# ============================================================

def _normalise_text(text: str) -> str:
    """Normalise OCR text for matching."""

    text = text.lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("−", "-")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _all_ocr_text(ocr_results) -> str:
    """Combine OCR words into one searchable string."""

    return " ".join(
        item.text
        for item in ocr_results
        if item.text
    )


# ============================================================
# PROBLEM PHRASES
# ============================================================

def _find_problem_phrases(
    ocr_results,
) -> list[str]:
    """Identify known problem-related phrases from OCR."""

    matches: list[str] = []

    for item in ocr_results:

        text = _normalise_text(
            item.text
        )

        if not text:
            continue

        for phrase_group in PROBLEM_PHRASES.values():

            for phrase in phrase_group:

                if phrase in text:
                    matches.append(
                        item.text
                    )

    combined = _normalise_text(
        _all_ocr_text(ocr_results)
    )

    for phrase_group in PROBLEM_PHRASES.values():

        for phrase in phrase_group:

            if phrase in combined:
                matches.append(
                    phrase
                )

    unique_matches = []

    for match in matches:

        if match not in unique_matches:
            unique_matches.append(
                match
            )

    return unique_matches


# ============================================================
# GENERIC EVIDENCE
# ============================================================

def _find_generic_evidence(
    ocr_results,
) -> list[str]:
    """Collect OCR text containing error/warning terminology."""

    evidence = []

    keywords = [
        "error",
        "warning",
        "failed",
        "invalid",
        "wrong",
        "cannot",
        "could not",
        "does not",
        "not intersect",
        "rebuild",
        "geometry",
        "parameter",
        "tangent",
    ]

    for item in ocr_results:

        text = item.text.strip()

        if not text:
            continue

        lowered = _normalise_text(
            text
        )

        if any(
            keyword in lowered
            for keyword in keywords
        ):
            evidence.append(
                text
            )

    return evidence


# ============================================================
# FEATURE DETECTION
# ============================================================

def _detect_feature_names(
    ocr_results,
) -> list[str]:
    """Detect likely SOLIDWORKS feature names from OCR."""

    combined = _normalise_text(
        _all_ocr_text(ocr_results)
    )

    detected = []

    for feature_name, pattern in FEATURE_NAME_PATTERNS:

        if re.search(
            pattern,
            combined,
            re.IGNORECASE,
        ):
            detected.append(
                feature_name
            )

    return detected


def _detect_primary_feature(
    feature_names: list[str],
    problem_type: str | None,
) -> str | None:
    """Select the most likely primary feature."""

    if not feature_names:
        return None

    if problem_type == "cut_not_intersect_model":

        if "cut-extrude" in feature_names:
            return "cut-extrude"

    if problem_type == "missing_reference":

        for name in (
            "chamfer",
            "fillet",
            "cut-extrude",
            "boss-extrude",
            "extrude",
            "sketch",
        ):

            if name in feature_names:
                return name

    if problem_type in (
        "rebuild_error",
        "geometric_conditions",
        "invalid_parameters",
        "tangent_propagation",
    ):

        for name in (
            "chamfer",
            "fillet",
            "cut-extrude",
            "boss-extrude",
            "extrude",
            "revolve",
            "loft",
            "sweep",
            "sketch",
        ):

            if name in feature_names:
                return name

    priority = [
        "cut-extrude",
        "boss-extrude",
        "chamfer",
        "fillet",
        "extrude",
        "revolve",
        "loft",
        "sweep",
        "hole",
        "pattern",
        "mirror",
        "sketch",
    ]

    for name in priority:

        if name in feature_names:
            return name

    return feature_names[0]


# ============================================================
# PARAMETER EXTRACTION
# ============================================================

def _extract_parameter_values(
    ocr_results,
) -> list[str]:
    """
    Extract visible CAD-style parameter values.

    Status bar OCR is ignored.
    """

    values = []

    for item in ocr_results:

        region_name = (
            item.region_name
            or ""
        ).lower()

        if region_name == "status_bar":
            continue

        matches = PARAMETER_PATTERN.findall(
            item.text
        )

        for match in matches:

            clean = re.sub(
                r"\s+",
                "",
                match,
            )

            if clean not in values:
                values.append(
                    clean
                )

    return values


# ============================================================
# PROBLEM TYPE CLASSIFICATION
# ============================================================

def _classify_problem_type(
    ocr_results,
) -> str | None:
    """
    Determine the most specific problem category supported
    by the screenshot OCR.

    Priority:
        1. Cut intersection error
        2. Missing face/plane
        3. Geometric conditions
        4. Rebuild error
        5. Invalid parameters
        6. Tangent propagation

    Tangent propagation is treated as supporting evidence,
    not as the primary problem when a stronger error exists.
    """

    combined = _normalise_text(
        _all_ocr_text(ocr_results)
    )

    # 1. CUT DOES NOT INTERSECT MODEL
    cut_patterns = [
        "cut does not intersect the model",
        "does not intersect the model",
        "not intersect the model",
        "intended cut does not intersect",
        "cut not intersect",
    ]

    for pattern in cut_patterns:
        if pattern in combined:
            return "cut_not_intersect_model"

    # OCR-token fallback for fragmented text.
    words = combined.split()

    has_cut = (
        "cut" in words
        or any(
            word.startswith("cut-extrud")
            or word.startswith("cutextrud")
            for word in words
        )
    )

    has_not = (
        "not" in words
        or "doesn't" in words
        or "doesnt" in words
    )

    has_intersect = (
        "intersect" in words
        or any(
            word.startswith("intersect")
            for word in words
        )
    )

    has_model = (
        "model" in words
        or "models" in words
    )

    if has_cut and has_not and has_intersect and has_model:
        return "cut_not_intersect_model"

    # 2. MISSING FACE / PLANE
    missing_reference_patterns = [
        "could not find face",
        "could not find plane",
        "cannot find face",
        "cannot find plane",
        "missing face",
        "missing plane",
        "find face or plane",
    ]

    for pattern in missing_reference_patterns:
        if pattern in combined:
            return "missing_reference"

    has_find = "find" in words or "found" in words
    has_face = "face" in words or "faces" in words
    has_plane = "plane" in words or "planes" in words
    has_could = (
        "could" in words
        or "cannot" in words
        or "can't" in words
    )

    if has_find and (has_face or has_plane) and has_could:
        return "missing_reference"

    # 3. GEOMETRIC CONDITIONS
    # This MUST come before tangent propagation because
    # tangent propagation is a Chamfer parameter/evidence,
    # while geometry conditions describes the rebuild error.
    if (
        "geometric conditions" in combined
        or "geometry conditions" in combined
        or "due to geometric conditions" in combined
        or "due to geometry conditions" in combined
    ):
        return "geometric_conditions"

    # 4. REBUILD ERROR
    if (
        "rebuild error" in combined
        or "failed to rebuild" in combined
        or "failed to build" in combined
        or "not build" in combined
        or "cannot rebuild" in combined
        or "could not rebuild" in combined
    ):
        return "rebuild_error"

    # 5. INVALID PARAMETERS
    if (
        "selected geometry and parameters" in combined
        or "invalid parameter" in combined
        or "invalid parameters" in combined
    ):
        return "invalid_parameters"

    # 6. TANGENT PROPAGATION
    # Only reaches here if no stronger problem was detected.
    if "tangent propagation" in combined:
        return "tangent_propagation"

    return None


# ============================================================
# PROBLEM EVIDENCE
# ============================================================

def build_problem_evidence(
    ocr_results,
) -> ProblemEvidence:
    """Build structured problem evidence from OCR."""

    problem_type = _classify_problem_type(
        ocr_results
    )

    feature_names = _detect_feature_names(
        ocr_results
    )

    primary_feature = _detect_primary_feature(
        feature_names,
        problem_type,
    )

    problem_phrases = _find_problem_phrases(
        ocr_results
    )

    generic_evidence = _find_generic_evidence(
        ocr_results
    )

    parameter_values = _extract_parameter_values(
        ocr_results
    )

    problem_visible = (
        problem_type is not None
        or len(problem_phrases) > 0
        or len(generic_evidence) > 0
    )

    related_features = [
        name
        for name in feature_names
        if name != primary_feature
    ]

    error_messages = []

    for item in problem_phrases:

        if item not in error_messages:
            error_messages.append(
                item
            )

    additional_evidence = []

    for item in generic_evidence:

        if item not in additional_evidence:

            if item not in error_messages:
                additional_evidence.append(
                    item
                )

    problem_evidence = ProblemEvidence(
        problem_visible=problem_visible,
        problem_type=problem_type,
        primary_feature=primary_feature,
        related_features=related_features,
        error_messages=error_messages,
        additional_evidence=additional_evidence,
        visible_parameter_values=parameter_values,
    )

    problem_evidence.summary = _build_diagnosis_summary(
        problem_evidence
    )

    return problem_evidence


# ============================================================
# VISUAL OBSERVATIONS
# ============================================================

def build_visual_observations(
    regions,
    problem_evidence: ProblemEvidence,
) -> VisualObservations:
    """Build high-level screenshot observations."""

    region_names = {
        region.name
        for region in regions
    }

    error_message_visible = (
        len(
            problem_evidence.error_messages
        ) > 0
    )

    warning_message_visible = (
        any(
            "warning" in text.lower()
            for text in (
                problem_evidence
                .additional_evidence
            )
        )
        or any(
            "warning" in text.lower()
            for text in (
                problem_evidence
                .error_messages
            )
        )
    )

    return VisualObservations(
        error_message_visible=(
            error_message_visible
        ),
        warning_message_visible=(
            warning_message_visible
        ),
        left_ui_panel_visible=(
            "left_ui_panel"
            in region_names
        ),
        graphics_area_visible=(
            "graphics_area"
            in region_names
        ),
        status_bar_visible=(
            "status_bar"
            in region_names
        ),
        dialog_visible=(
            "dialog"
            in region_names
        ),
    )


# ============================================================
# IMAGE INFO
# ============================================================

def build_image_info(
    image,
    original_path: Path,
    was_resized: bool = False,
) -> ImageInfo:
    """Build image metadata."""

    height, width = image.shape[:2]

    return ImageInfo(
        width=int(width),
        height=int(height),
        original_path=str(
            original_path
        ),
        was_resized=was_resized,
    )


# ============================================================
# SCREENSHOT SELECTION
# ============================================================

def find_screenshot() -> Path:
    """Find the first supported screenshot."""

    screenshots = find_all_screenshots()

    return screenshots[0]


def find_all_screenshots() -> list[Path]:
    """
    Find every supported screenshot in SCREENSHOTS_DIR, sorted by name.

    Used for batch mode - running the pipeline over every screenshot in
    the folder in one go, rather than only the first one.
    """

    supported_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
    }

    screenshots = sorted(
        path
        for path in SCREENSHOTS_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in supported_extensions
    )

    if not screenshots:

        raise FileNotFoundError(
            "No screenshot found in:\n"
            f"{SCREENSHOTS_DIR}\n\n"
            "Place a SOLIDWORKS screenshot there "
            "and run the program again."
        )

    return screenshots


def select_screenshot() -> Path:
    """
    Choose which screenshot to analyze for this run.

    If there's only one screenshot in SCREENSHOTS_DIR, it's used
    automatically - no need to prompt for a choice of one. If there
    are several, prints a numbered list and asks the user to pick one,
    re-prompting on invalid input rather than guessing.
    """

    screenshots = find_all_screenshots()

    if len(screenshots) == 1:
        return screenshots[0]

    print(
        f"Found {len(screenshots)} screenshots in "
        f"{SCREENSHOTS_DIR}:"
    )
    print()

    for index, path in enumerate(screenshots, start=1):
        print(f"  {index}. {path.name}")

    print()

    while True:

        choice = input(
            f"Select a screenshot to analyze "
            f"(1-{len(screenshots)}): "
        ).strip()

        if choice.isdigit():

            choice_index = int(choice)

            if 1 <= choice_index <= len(screenshots):
                return screenshots[choice_index - 1]

        print(
            f"Invalid selection - enter a number from "
            f"1 to {len(screenshots)}."
        )


# ============================================================
# JSON OUTPUT
# ============================================================

def save_analysis(
    analysis: ScreenshotAnalysis,
    filename: str = DEFAULT_OUTPUT_FILENAME,
) -> Path:
    """
    Save structured analysis JSON.

    filename defaults to DEFAULT_OUTPUT_FILENAME (single-screenshot
    mode). Batch mode passes a per-screenshot filename instead, so
    running several screenshots doesn't overwrite the same file.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            analysis.to_dict(),
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


# ============================================================
# CONSOLE DISPLAY
# ============================================================

def print_summary(
    analysis: ScreenshotAnalysis,
) -> None:
    """Print a compact analysis summary."""

    print()
    print("=" * 60)
    print("SOLIDWORKS SCREENSHOT ANALYSIS")
    print("=" * 60)

    print()
    print("IMAGE")

    print(
        f"Size              : "
        f"{analysis.image.width} x "
        f"{analysis.image.height}"
    )

    print(
        f"Was resized       : "
        f"{analysis.image.was_resized}"
    )

    print()
    print("REGIONS")

    for region in analysis.regions:

        print(
            f"{region.name:18}"
            f"confidence={region.confidence:.2f} "
            f"bbox={region.bbox.as_list()}"
        )

    print()
    print("VISUAL OBSERVATIONS")

    visual = analysis.visual_observations

    print(
        f"error_message_visible : "
        f"{visual.error_message_visible}"
    )

    print(
        f"warning_message_visible: "
        f"{visual.warning_message_visible}"
    )

    print(
        f"left_ui_panel_visible : "
        f"{visual.left_ui_panel_visible}"
    )

    print(
        f"graphics_area_visible  : "
        f"{visual.graphics_area_visible}"
    )

    print(
        f"status_bar_visible     : "
        f"{visual.status_bar_visible}"
    )

    print(
        f"dialog_visible         : "
        f"{visual.dialog_visible}"
    )

    print()
    print("GEOMETRY ANALYSIS")

    geometry = analysis.geometry_analysis

    if geometry is None:

        print(
            "No graphics-area geometry analysis."
        )

    else:

        print(
            f"Graphics size         : "
            f"{geometry['graphics_width']} x "
            f"{geometry['graphics_height']}"
        )

        print(
            f"Edge pixel ratio      : "
            f"{geometry['edge_pixel_ratio']}"
        )

        print(
            f"Dark pixel ratio      : "
            f"{geometry['dark_pixel_ratio']}"
        )

        print(
            f"Connected components  : "
            f"{geometry['connected_component_count']}"
        )

        print(
            f"Large components      : "
            f"{geometry['large_component_count']}"
        )

        print(
            f"Geometry bbox         : "
            f"{geometry['geometry_bbox']}"
        )

        print(
            f"Horizontal lines      : "
            f"{geometry['horizontal_line_count']}"
        )

        print(
            f"Vertical lines        : "
            f"{geometry['vertical_line_count']}"
        )

        print(
            f"Diagonal lines        : "
            f"{geometry['diagonal_line_count']}"
        )

        print(
            f"Visual geometry       : "
            f"{geometry['visual_geometry_detected']}"
        )

    print()
    print("PROBLEM EVIDENCE")

    evidence = analysis.problem_evidence

    print(
        f"problem_visible       : "
        f"{evidence.problem_visible}"
    )

    print(
        f"problem_type          : "
        f"{evidence.problem_type}"
    )

    print(
        f"primary_feature       : "
        f"{evidence.primary_feature}"
    )

    print(
        f"related_features      : "
        f"{evidence.related_features}"
    )

    print(
        f"error_messages        : "
        f"{evidence.error_messages}"
    )

    print(
        f"additional_evidence   : "
        f"{evidence.additional_evidence}"
    )

    print(
        f"visible_parameters    : "
        f"{evidence.visible_parameter_values}"
    )

    print(
        f"summary               : "
        f"{evidence.summary}"
    )

    if analysis.vision_analysis:

        print()
        print("VISION MODEL")

        vision = analysis.vision_analysis

        print(
            f"Model                : "
            f"{vision.model_used}"
        )

        print(
            f"Detected problem     : "
            f"{vision.detected_problem}"
        )

        print(
            f"Confidence           : "
            f"{vision.confidence}"
        )

    print()
    print("SOLIDWORKS API DATA")

    if analysis.solidworks_api_data is None:

        print("Not available for this run.")

    else:

        api_data = analysis.solidworks_api_data

        print(
            f"Document title       : "
            f"{api_data.get('DocumentTitle', 'Unknown')}"
        )

        print(
            f"Features             : "
            f"{len(api_data.get('Features', []))}"
        )

        print(
            f"Sketches             : "
            f"{len(api_data.get('Sketches', {}))}"
        )

        print(
            f"Extrusions           : "
            f"{len(api_data.get('Extrusions', {}))}"
        )

        print(
            f"Problems             : "
            f"{len(api_data.get('Problems', []))}"
        )

    print()
    print("SUGGESTED SOLUTION")

    if analysis.solution is None:

        print(
            "Not available for this run "
            "(no confirmed problem, or generation failed)."
        )

    else:

        solution = analysis.solution

        print(
            f"Suggested fix : "
            f"{solution.suggested_fix}"
        )

        print(
            f"Confidence    : "
            f"{solution.confidence}"
        )

        if solution.reasoning:

            print("Reasoning:")

            for reason in solution.reasoning:

                print(f"  - {reason}")

    print()
    print("=" * 60)


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_screenshot(
    screenshot_path: Path,
) -> tuple[ScreenshotAnalysis, np.ndarray]:
    """
    Run the full pipeline (load, preprocess, regions, OCR, geometry,
    problem evidence, vision model, SolidWorks API data) on a single
    screenshot and return (ScreenshotAnalysis, resized_image).

    The resized image is returned alongside the analysis so callers
    (e.g. report_generator.py) can build a report without reloading
    and reprocessing the screenshot a second time.

    Does not save or print anything - that's the caller's job, so this
    can be reused identically by both single-screenshot and batch mode.
    """

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = load_image(
        screenshot_path
    )

    # --------------------------------------------------------
    # Preprocess screenshot
    # --------------------------------------------------------

    processed = preprocess_screenshot(
        image
    )

    processed_image = (
        processed["resized"]
    )

    ocr_image = (
        processed["ocr_ready"]
    )

    was_resized = (
        processed["was_resized"]
    )

    # --------------------------------------------------------
    # Detect regions
    #
    # IMPORTANT:
    # Your regions.py calls this function
    # detect_all_regions().
    # --------------------------------------------------------

    regions = detect_all_regions(
        processed_image
    )

    # --------------------------------------------------------
    # OCR
    #
    # OCR uses the OCR-enhanced image.
    # --------------------------------------------------------

    ocr_results = (
        extract_text_from_regions(
            ocr_image,
            regions,
        )
    )

    # --------------------------------------------------------
    # Visual geometry analysis
    #
    # Geometry uses the resized COLOR image.
    # --------------------------------------------------------

    geometry_analysis = None

    for region in regions:

        if region.name != "graphics_area":
            continue

        geometry = analyze_graphics_area(
            processed_image,
            region.bbox.as_list(),
        )

        geometry_analysis = (
            geometry.to_dict()
        )

        break

    # --------------------------------------------------------
    # Problem evidence
    # --------------------------------------------------------

    problem_evidence = (
        build_problem_evidence(
            ocr_results
        )
    )

    # --------------------------------------------------------
    # High-level visual observations
    # --------------------------------------------------------

    visual_observations = (
        build_visual_observations(
            regions,
            problem_evidence,
        )
    )

    # --------------------------------------------------------
    # SolidWorks C# API data
    #
    # Refresh analysis_api.json by running the C# analyzer against
    # whatever's currently open in SolidWorks, then load it. Loaded
    # here because the vision model step right after this needs it.
    # --------------------------------------------------------

    run_solidworks_analyzer()

    solidworks_api_data = load_solidworks_api_data()

    # --------------------------------------------------------
    # Vision model
    #
    # Tries a real Groq-backed model first; falls back to the
    # NullVisionModel stub if the API key/package isn't set up
    # yet, or if the request itself fails, so a missing/failed AI
    # call never crashes the whole pipeline run.
    # --------------------------------------------------------

    try:
        vision_model = GroqVisionModel()
    except GroqVisionModelError as exc:
        print(f"Groq vision model unavailable, using stub: {exc}")
        vision_model = NullVisionModel()

    try:
        vision_analysis = analyze_screenshot(
            processed_image,
            ocr_results,
            regions,
            problem_evidence,
            solidworks_api_data,
            vision_model,
        )
    except GroqVisionModelError as exc:
        print(f"Groq vision analysis failed, using stub result: {exc}")
        vision_analysis = analyze_screenshot(
            processed_image,
            ocr_results,
            regions,
            problem_evidence,
            solidworks_api_data,
            NullVisionModel(),
        )

    # --------------------------------------------------------
    # Solution generation
    #
    # Only runs if a real problem was actually identified above -
    # no point suggesting a fix for "no problem detected". Same
    # fallback pattern as the vision model: any failure here
    # (missing key, API error) results in solution=None rather
    # than crashing the run, since a missing suggested fix is far
    # less severe than losing the whole analysis.
    # --------------------------------------------------------

    solution = None

    try:
        solution = generate_solution(
            vision_analysis,
            problem_evidence,
            solidworks_api_data,
        )
    except SolutionGeneratorError as exc:
        print(f"Solution generation unavailable: {exc}")

    # --------------------------------------------------------
    # Image information
    # --------------------------------------------------------

    image_info = build_image_info(
        processed_image,
        screenshot_path,
        was_resized=was_resized,
    )

    # --------------------------------------------------------
    # Build final analysis
    # --------------------------------------------------------

    analysis = ScreenshotAnalysis(
        image=image_info,
        regions=regions,
        ocr=ocr_results,
        visual_observations=(
            visual_observations
        ),
        problem_evidence=(
            problem_evidence
        ),
        vision_analysis=(
            vision_analysis
        ),
        geometry_analysis=(
            geometry_analysis
        ),
        solidworks_api_data=(
            solidworks_api_data
        ),
        solution=(
            solution
        ),
    )

    return analysis, processed_image


def main() -> None:

    # --------------------------------------------------------
    # Locate the screenshot to process
    #
    # Real usage is one screenshot at a time - it corresponds to
    # whatever's currently open in SolidWorks, matched against the
    # current analysis_api.json. If several screenshots are present,
    # select_screenshot() prompts for which one to analyze this run.
    # find_all_screenshots() is still available for batch reliability
    # testing, but isn't used here.
    # --------------------------------------------------------

    screenshot_path = select_screenshot()

    print(
        f"Screenshot: "
        f"{screenshot_path}"
    )

    # --------------------------------------------------------
    # Run the pipeline
    # --------------------------------------------------------

    analysis, processed_image = process_screenshot(
        screenshot_path
    )

    # --------------------------------------------------------
    # Save and display
    # --------------------------------------------------------

    output_path = save_analysis(
        analysis
    )

    print_summary(
        analysis
    )

    print()
    print(
        f"JSON saved to:\n"
        f"{output_path}"
    )

    report_path = generate_html_report(
        analysis,
        screenshot_path,
        processed_image,
    )

    print(
        f"HTML report:\n"
        f"{report_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()