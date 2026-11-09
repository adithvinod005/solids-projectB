"""
vision/analyzer.py

Defines the interface a future AI/vision model will implement in order to
turn (screenshot evidence + OCR text + detected regions + problem
evidence + SolidWorks API data) into an actual SOLIDWORKS problem
diagnosis.

THIS FILE DOES NOT CALL ANY AI PROVIDER ITSELF. It exists purely to:
1. Define a clean, stable contract (`VisionModelClient`) that any
   implementation - Claude, OpenAI, a local model, or a rules engine -
   can satisfy.
2. Provide a `NullVisionModel` stub so the rest of the pipeline (main.py)
   can run end-to-end even without a real model plugged in.

CURRENT IMPLEMENTATION:
See vision/claude_client.py for the Claude-backed implementation
(ClaudeVisionModel), passed into analyze_screenshot() in main.py.

SCOPE BOUNDARY - PROBLEM IDENTIFICATION ONLY:
Any implementation of `VisionModelClient` must identify and describe the
SOLIDWORKS problem visible in the screenshot - never propose how to fix
it. Prompt any real model accordingly (e.g. "identify the problem and the
visual evidence for it; do not suggest a fix or corrective action") and
keep the returned `VisionAnalysisResult` limited to `detected_problem`,
`confidence`, and `evidence`. Do not add or populate a solution/fix/
suggested_action/recommendation field - `models/output_schema.py`
actively rejects those field names at serialization time. Deciding what
to do about a detected problem is out of scope for this pipeline.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import numpy as np

from models.output_schema import (
    OcrResult,
    ProblemEvidence,
    Region,
    VisionAnalysisResult,
)


class VisionModelClient(Protocol):
    """
    Contract for any AI/vision backend that can analyze a SOLIDWORKS
    screenshot together with everything already extracted earlier in
    the pipeline: OCR text, detected regions, the phrase-classified
    problem evidence, and (if available) the SolidWorks C# API data.

    A concrete implementation might, for example, format all of this
    as a text prompt to an LLM and parse its response into a
    VisionAnalysisResult - or, if the implementation supports image
    input, send the screenshot itself too.
    """

    def analyze(
        self,
        image: np.ndarray,
        ocr_results: list[OcrResult],
        regions: list[Region],
        problem_evidence: ProblemEvidence,
        solidworks_api_data: Optional[dict[str, Any]],
    ) -> VisionAnalysisResult:
        """
        Analyze a screenshot and return a structured diagnosis.

        Args:
            image: the (resized) screenshot as a BGR NumPy array.
            ocr_results: text already extracted by ocr/extractor.py.
            regions: UI regions already detected by image/regions.py.
            problem_evidence: phrase-classified evidence already built
                by main.py's build_problem_evidence() (problem_type,
                primary_feature, related_features, error_messages,
                etc.) - the model's job is to reason over this, not
                redo the OCR classification from scratch.
            solidworks_api_data: the parsed analysis_api.json dict from
                the C# analyzer (features, sketches, extrusions,
                problems with error codes/causes), or None if it
                wasn't available for this run.

        Returns:
            A VisionAnalysisResult describing what problem (if any) was
            identified, how confident the model is, and what evidence
            supports that conclusion. Must NOT include a solution, fix,
            or suggested corrective action - identification only.
        """
        ...


class NullVisionModel:
    """
    Placeholder implementation used until a real AI model is connected.

    Rather than guessing at a diagnosis, this stub is explicit about the
    fact that no real analysis has happened yet - it reports zero
    confidence and empty evidence, so downstream consumers of the JSON
    output can clearly tell "no vision analysis was performed" apart from
    "the vision model looked and found nothing".
    """

    def analyze(
        self,
        image: np.ndarray,
        ocr_results: list[OcrResult],
        regions: list[Region],
        problem_evidence: ProblemEvidence,
        solidworks_api_data: Optional[dict[str, Any]],
    ) -> VisionAnalysisResult:
        return VisionAnalysisResult(
            detected_problem=None,
            confidence=0.0,
            evidence=[],
            model_used="none",
        )


def analyze_screenshot(
    image: np.ndarray,
    ocr_results: list[OcrResult],
    regions: list[Region],
    problem_evidence: ProblemEvidence,
    solidworks_api_data: Optional[dict[str, Any]],
    model: VisionModelClient | None = None,
) -> VisionAnalysisResult:
    """
    Run vision/AI analysis using the given model client.

    If no `model` is provided, falls back to `NullVisionModel`, so this
    function - and therefore the whole pipeline in main.py - always
    returns a valid, well-formed result, with or without a real AI model
    plugged in.
    """
    active_model: VisionModelClient = (
        model if model is not None else NullVisionModel()
    )
    return active_model.analyze(
        image,
        ocr_results,
        regions,
        problem_evidence,
        solidworks_api_data,
    )