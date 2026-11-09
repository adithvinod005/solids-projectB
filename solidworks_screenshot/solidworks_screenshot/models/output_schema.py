"""
models/output_schema.py

Defines the structured data flowing through the SOLIDWORKS screenshot
analysis pipeline.

V1 SCOPE:
    Screenshot -> visual observations -> problem identification
                                       -> solution suggestion

The identification stage (regions, OCR, geometry, problem_evidence,
vision_analysis) is intentionally solution-free - it identifies
problems and evidence only. The dedicated `solution` field is the one
deliberate exception: it holds a suggested fix generated afterward,
from the already-identified problem plus SolidWorks API data, by a
separate step (see vision/solution_generator.py). Keeping it as its
own top-level field - rather than mixed into the identification data -
keeps that boundary visible in the schema itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# ============================================================
# FORBIDDEN SOLUTION / FIX FIELDS
# ============================================================

FORBIDDEN_FIELD_NAMES = {
    "solution",
    "suggested_action",
    "fix",
    "recommendation",
    "next_steps",
}


class SolutionFieldNotAllowedError(Exception):
    """Raised if a forbidden solution/fix field appears."""


def _assert_no_forbidden_fields(
    value: Any,
    path: str = "",
) -> None:
    """
    Recursively check dictionaries and lists for forbidden
    solution/fix fields.
    """

    if isinstance(value, dict):

        for key, sub_value in value.items():

            if (
                str(key).lower()
                in FORBIDDEN_FIELD_NAMES
            ):

                location = (
                    f"{path}.{key}"
                    if path
                    else str(key)
                )

                raise SolutionFieldNotAllowedError(
                    f"Forbidden field '{location}' found "
                    f"in screenshot-identification output. "
                    f"Identification stays solution-free; "
                    f"solutions belong in the dedicated "
                    f"'solution' field only."
                )

            _assert_no_forbidden_fields(
                sub_value,
                (
                    f"{path}.{key}"
                    if path
                    else str(key)
                ),
            )

    elif isinstance(value, list):

        for i, item in enumerate(value):

            _assert_no_forbidden_fields(
                item,
                f"{path}[{i}]",
            )


# ============================================================
# IMAGE INFORMATION
# ============================================================

@dataclass
class ImageInfo:
    width: int
    height: int
    original_path: str
    was_resized: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# BOUNDING BOX
# ============================================================

@dataclass
class BBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def as_list(self) -> list:
        return [
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
        ]


# ============================================================
# REGION
# ============================================================

@dataclass
class Region:
    name: str
    bbox: BBox
    confidence: float
    detection_method: str = "heuristic"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "confidence": round(
                self.confidence,
                3,
            ),
            "bbox": self.bbox.as_list(),
            "detection_method": (
                self.detection_method
            ),
        }


# ============================================================
# OCR RESULT
# ============================================================

@dataclass
class OcrResult:
    text: str
    confidence: float
    bbox: BBox
    region_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(
                self.confidence,
                3,
            ),
            "bbox": self.bbox.as_list(),
            "region": self.region_name,
        }


# ============================================================
# VISUAL OBSERVATIONS
# ============================================================

@dataclass
class VisualObservations:
    error_message_visible: bool = False
    warning_message_visible: bool = False

    left_ui_panel_visible: bool = False
    graphics_area_visible: bool = False
    status_bar_visible: bool = False
    dialog_visible: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# PROBLEM EVIDENCE
# ============================================================

@dataclass
class ProblemEvidence:
    """
    Evidence extracted from the screenshot.

    This class deliberately contains observations about
    problems, not solutions.
    """

    problem_visible: bool = False

    problem_type: Optional[str] = None

    primary_feature: Optional[str] = None

    related_features: list[str] = field(
        default_factory=list
    )

    error_messages: list[str] = field(
        default_factory=list
    )

    additional_evidence: list[str] = field(
        default_factory=list
    )

    visible_parameter_values: list[str] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # A single synthesized sentence describing WHAT was
    # observed (problem type + primary feature + related
    # features). Deliberately descriptive, never prescriptive -
    # it states what appears to be wrong, never what to do
    # about it. Populated by main.py's diagnosis builder.
    # --------------------------------------------------------

    summary: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "problem_visible": (
                self.problem_visible
            ),
            "problem_type": (
                self.problem_type
            ),
            "primary_feature": (
                self.primary_feature
            ),
            "related_features": (
                self.related_features
            ),
            "error_messages": (
                self.error_messages
            ),
            "additional_evidence": (
                self.additional_evidence
            ),
            "visible_parameter_values": (
                self.visible_parameter_values
            ),
            "summary": (
                self.summary
            ),
        }


# ============================================================
# VISION ANALYSIS
# ============================================================

@dataclass
class VisionAnalysisResult:
    """
    Result from the identification-stage AI/vision model.

    The model identifies the problem and supporting evidence.

    It does NOT provide a solution at this stage - see
    SolutionResult below for that, generated as a distinct,
    later step.
    """

    detected_problem: Optional[str] = None

    confidence: float = 0.0

    evidence: list[str] = field(
        default_factory=list
    )

    model_used: str = "none"

    def to_dict(self) -> dict:
        return {
            "detected_problem": (
                self.detected_problem
            ),
            "confidence": round(
                self.confidence,
                3,
            ),
            "evidence": (
                self.evidence
            ),
            "model_used": (
                self.model_used
            ),
        }


# ============================================================
# SOLUTION RESULT
# ============================================================

@dataclass
class SolutionResult:
    """
    A suggested fix for the already-identified problem.

    This is deliberately a SEPARATE, later stage from
    VisionAnalysisResult - it is generated afterward, from the
    confirmed diagnosis plus SolidWorks API data, by
    vision/solution_generator.py. It is the one place in this
    schema where solution/fix content is allowed; everything
    else in ScreenshotAnalysis stays identification-only (see
    _assert_no_forbidden_fields).
    """

    suggested_fix: Optional[str] = None

    confidence: float = 0.0

    reasoning: list[str] = field(
        default_factory=list
    )

    model_used: str = "none"

    def to_dict(self) -> dict:
        return {
            "suggested_fix": (
                self.suggested_fix
            ),
            "confidence": round(
                self.confidence,
                3,
            ),
            "reasoning": (
                self.reasoning
            ),
            "model_used": (
                self.model_used
            ),
        }


# ============================================================
# SCREENSHOT ANALYSIS
# ============================================================

@dataclass
class ScreenshotAnalysis:

    image: ImageInfo

    regions: list[Region] = field(
        default_factory=list
    )

    ocr: list[OcrResult] = field(
        default_factory=list
    )

    visual_observations: VisualObservations = field(
        default_factory=VisualObservations
    )

    problem_evidence: ProblemEvidence = field(
        default_factory=ProblemEvidence
    )

    vision_analysis: Optional[
        VisionAnalysisResult
    ] = None

    # --------------------------------------------------------
    # Pixel-level geometry observations from
    # image/geometry.py
    # --------------------------------------------------------

    geometry_analysis: Optional[
        dict[str, Any]
    ] = None

    # --------------------------------------------------------
    # Data received from the SolidWorks C# API
    # --------------------------------------------------------

    solidworks_api_data: Optional[
        dict[str, Any]
    ] = None

    # --------------------------------------------------------
    # Suggested fix for the identified problem - the one
    # deliberate solution field in this schema. See
    # SolutionResult above.
    # --------------------------------------------------------

    solution: Optional[
        SolutionResult
    ] = None

    def to_dict(self) -> dict:

        result = {
            "image": self.image.to_dict(),

            "regions": [
                region.to_dict()
                for region in self.regions
            ],

            "ocr": [
                item.to_dict()
                for item in self.ocr
            ],

            "visual_observations": (
                self.visual_observations.to_dict()
            ),

            "problem_evidence": (
                self.problem_evidence.to_dict()
            ),

            "vision_analysis": (
                self.vision_analysis.to_dict()
                if self.vision_analysis
                else None
            ),

            # --------------------------------------------
            # Geometry observations
            # --------------------------------------------

            "geometry_analysis": (
                self.geometry_analysis
            ),

            # --------------------------------------------
            # C# SolidWorks API data
            # --------------------------------------------

            "solidworks_api_data": (
                self.solidworks_api_data
            ),

            # --------------------------------------------
            # Suggested fix (the one allowed solution field)
            # --------------------------------------------

            "solution": (
                self.solution.to_dict()
                if self.solution
                else None
            ),
        }

        # ------------------------------------------------
        # The identification-stage output is checked
        # separately from solidworks_api_data and solution -
        # those two are the deliberate exceptions. Everything
        # else (regions, OCR, geometry, problem_evidence,
        # vision_analysis) must never contain a solution/fix/
        # recommendation field.
        # ------------------------------------------------

        identification_output = {
            key: value
            for key, value in result.items()
            if key not in ("solidworks_api_data", "solution")
        }

        _assert_no_forbidden_fields(
            identification_output
        )

        return result