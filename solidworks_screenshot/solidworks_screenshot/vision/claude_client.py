"""
vision/claude_client.py

Claude-backed implementation of VisionModelClient.

DESIGN NOTES:
- This is a TEXT-ONLY implementation: it does not send the screenshot
  image itself to the model. It reasons entirely over evidence already
  extracted earlier in the pipeline - OCR text, detected regions,
  phrase-classified problem evidence, and the SolidWorks C# API data
  (feature error codes, causes, sketch/extrusion parameters).
- Requires the `anthropic` package (`pip install anthropic`) and an
  ANTHROPIC_API_KEY environment variable. If either is missing, this
  raises a clear error rather than failing mysteriously mid-request.
- The prompt explicitly forbids the model from suggesting fixes,
  matching this pipeline's problem-identification-only scope. The
  response is requested as strict JSON so it can be parsed directly
  into a VisionAnalysisResult - no solution/fix/recommendation fields
  are accepted even if the model includes them (see _parse_response).
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np

from models.output_schema import (
    OcrResult,
    ProblemEvidence,
    Region,
    VisionAnalysisResult,
)

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


DEFAULT_MODEL = "claude-sonnet-5"

# Fields this pipeline is not allowed to populate anywhere in its
# output - see models/output_schema.py's FORBIDDEN_FIELD_NAMES. Checked
# again here so a model that ignores the prompt instruction can't sneak
# a fix/solution field into the parsed result.
_FORBIDDEN_RESPONSE_KEYS = {
    "solution",
    "suggested_action",
    "fix",
    "recommendation",
    "next_steps",
}

SYSTEM_PROMPT = """You are analyzing evidence from a SOLIDWORKS CAD session to identify what problem is occurring.

You will be given:
- OCR text extracted from the screenshot
- Detected UI regions
- Problem evidence already classified from OCR (problem_type, primary_feature, related_features, error messages)
- SolidWorks API data from the C# analyzer, if available (feature names, types, error codes, and parent/child dependencies)

Your job is ONLY to identify and describe the problem and the evidence supporting that identification.

Do NOT suggest a fix, solution, workaround, or next steps of any kind. Only describe what is wrong and why you believe so.

Respond with ONLY a JSON object in this exact shape, with no other text:

{
  "detected_problem": "<one sentence describing the problem, or null if no clear problem is identifiable>",
  "confidence": <float between 0.0 and 1.0>,
  "evidence": ["<short evidence point>", "<short evidence point>", ...]
}
"""


class ClaudeVisionModelError(Exception):
    """Raised when the Claude-backed vision model can't run or respond."""


class ClaudeVisionModel:
    """
    VisionModelClient implementation backed by the Claude API.

    Text-only: reasons over already-extracted evidence rather than the
    raw screenshot image. See module docstring for why.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        if not ANTHROPIC_AVAILABLE:
            raise ClaudeVisionModelError(
                "The 'anthropic' package is not installed. "
                "Run: pip install anthropic"
            )

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not resolved_key:
            raise ClaudeVisionModelError(
                "No Anthropic API key found. Set the ANTHROPIC_API_KEY "
                "environment variable, or pass api_key= explicitly."
            )

        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._model = model

    # ----------------------------------------------------------------
    # PROMPT BUILDING
    # ----------------------------------------------------------------

    def _build_user_prompt(
        self,
        ocr_results: list[OcrResult],
        regions: list[Region],
        problem_evidence: ProblemEvidence,
        solidworks_api_data: Optional[dict[str, Any]],
    ) -> str:

        ocr_lines = [
            f"- ({item.region_name or 'unknown region'}) {item.text}"
            for item in ocr_results
            if item.text.strip()
        ]

        region_lines = [
            f"- {region.name} (confidence={region.confidence:.2f})"
            for region in regions
        ]

        sections = [
            "OCR TEXT:",
            "\n".join(ocr_lines) or "(none extracted)",
            "",
            "DETECTED REGIONS:",
            "\n".join(region_lines) or "(none detected)",
            "",
            "PROBLEM EVIDENCE (already classified from OCR):",
            json.dumps(problem_evidence.to_dict(), indent=2),
        ]

        if solidworks_api_data is not None:
            sections += [
                "",
                "SOLIDWORKS API DATA (from the C# analyzer):",
                json.dumps(solidworks_api_data, indent=2),
            ]
        else:
            sections += [
                "",
                "SOLIDWORKS API DATA: not available for this run.",
            ]

        return "\n".join(sections)

    # ----------------------------------------------------------------
    # RESPONSE PARSING
    # ----------------------------------------------------------------

    def _parse_response(self, raw_text: str) -> VisionAnalysisResult:
        cleaned = raw_text.strip()

        # Strip markdown code fences if the model wrapped the JSON in
        # them despite the "ONLY a JSON object" instruction.
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ClaudeVisionModelError(
                f"Model response was not valid JSON: {exc}\n"
                f"Raw response: {raw_text}"
            ) from exc

        for forbidden_key in _FORBIDDEN_RESPONSE_KEYS:
            parsed.pop(forbidden_key, None)

        detected_problem = parsed.get("detected_problem")
        confidence = float(parsed.get("confidence", 0.0))
        evidence = parsed.get("evidence", [])

        if not isinstance(evidence, list):
            evidence = [str(evidence)]

        return VisionAnalysisResult(
            detected_problem=detected_problem,
            confidence=max(0.0, min(1.0, confidence)),
            evidence=[str(item) for item in evidence],
            model_used=self._model,
        )

    # ----------------------------------------------------------------
    # PUBLIC INTERFACE (VisionModelClient protocol)
    # ----------------------------------------------------------------

    def analyze(
        self,
        image: np.ndarray,
        ocr_results: list[OcrResult],
        regions: list[Region],
        problem_evidence: ProblemEvidence,
        solidworks_api_data: Optional[dict[str, Any]],
    ) -> VisionAnalysisResult:

        user_prompt = self._build_user_prompt(
            ocr_results, regions, problem_evidence, solidworks_api_data
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            raise ClaudeVisionModelError(
                f"Claude API request failed: {exc}"
            ) from exc

        text_blocks = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]

        if not text_blocks:
            raise ClaudeVisionModelError(
                "Claude API response contained no text content."
            )

        return self._parse_response("\n".join(text_blocks))