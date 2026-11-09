"""
vision/gemini_client.py

Gemini-backed implementation of VisionModelClient.

Uses Google AI Studio's free tier (no billing required, rate-limited) -
see https://aistudio.google.com to create a key.

DESIGN NOTES:
- Same text-only approach as the Claude implementation: reasons entirely
  over evidence already extracted earlier in the pipeline (OCR text,
  regions, problem_evidence, solidworks_api_data), not the raw
  screenshot image.
- Requires the `google-genai` package (`pip install google-genai`) and a
  GEMINI_API_KEY environment variable.
- Same JSON-only response contract and forbidden-field stripping as
  vision/claude_client.py, so both implementations are interchangeable
  behind the VisionModelClient interface.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import numpy as np

from models.output_schema import (
    OcrResult,
    ProblemEvidence,
    Region,
    VisionAnalysisResult,
)

try:
    from google import genai

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# Google's free tier gives this model 500 requests/day, vs. only 20/day
# for gemini-3.6-flash - much more headroom for a project making two
# calls per screenshot (vision + solution generation). If Google
# retires/renames it, check aistudio.google.com for the current
# free-tier model list and update this one line.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

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

# ----------------------------------------------------------------
# RETRY HANDLING
# ----------------------------------------------------------------
#
# The free tier is prone to transient 503 ("high demand, try again
# later") and 429 (rate limit) errors under load - these are not
# actual problems with the request, just temporary server-side
# congestion. Retrying a couple of times with a short backoff clears
# most of them without giving up and falling back to the stub.

MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0  # doubles each retry: 2s, then 4s

_TRANSIENT_ERROR_MARKERS = (
    "503",
    "UNAVAILABLE",
    "429",
    "RESOURCE_EXHAUSTED",
)


def _is_transient_error(exc: Exception) -> bool:
    """
    Heuristic check for whether an exception looks like a temporary
    server-side issue (worth retrying) rather than a real problem with
    the request itself (not worth retrying).
    """
    text = str(exc)
    return any(marker in text for marker in _TRANSIENT_ERROR_MARKERS)

SYSTEM_PROMPT = """You are analyzing evidence from a SOLIDWORKS CAD session to identify what problem is occurring.

You will be given:
- OCR text extracted from the screenshot
- Detected UI regions
- Problem evidence already classified from OCR (problem_type, primary_feature, related_features, error messages)
- SolidWorks API data from the C# analyzer, if available (feature names, types, error codes, and parent/child dependencies)

Your job is ONLY to identify and describe the problem and the evidence supporting that identification.

Do NOT suggest a fix, solution, workaround, or next steps of any kind. Only describe what is wrong and why you believe so.

Respond with ONLY a JSON object in this exact shape, with no other text, no markdown code fences:

{
  "detected_problem": "<one sentence describing the problem, or null if no clear problem is identifiable>",
  "confidence": <float between 0.0 and 1.0>,
  "evidence": ["<short evidence point>", "<short evidence point>", ...]
}
"""


class GeminiVisionModelError(Exception):
    """Raised when the Gemini-backed vision model can't run or respond."""


class GeminiVisionModel:
    """
    VisionModelClient implementation backed by the Gemini API
    (Google AI Studio free tier).

    Text-only: reasons over already-extracted evidence rather than the
    raw screenshot image. See module docstring for why.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        if not GENAI_AVAILABLE:
            raise GeminiVisionModelError(
                "The 'google-genai' package is not installed. "
                "Run: pip install google-genai"
            )

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")

        if not resolved_key:
            raise GeminiVisionModelError(
                "No Gemini API key found. Set the GEMINI_API_KEY "
                "environment variable (get one free at "
                "aistudio.google.com), or pass api_key= explicitly."
            )

        self._client = genai.Client(api_key=resolved_key)
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
        # them despite the "no markdown code fences" instruction.
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise GeminiVisionModelError(
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

        response = None

        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):

            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=user_prompt,
                    config={
                        "system_instruction": SYSTEM_PROMPT,
                    },
                )
                break

            except Exception as exc:

                is_last_attempt = attempt == MAX_RETRY_ATTEMPTS

                if not _is_transient_error(exc) or is_last_attempt:
                    raise GeminiVisionModelError(
                        f"Gemini API request failed: {exc}"
                    ) from exc

                wait_seconds = (
                    RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                )

                print(
                    f"Gemini API temporarily unavailable "
                    f"(attempt {attempt}/{MAX_RETRY_ATTEMPTS}), "
                    f"retrying in {wait_seconds:.0f}s: {exc}"
                )

                time.sleep(wait_seconds)

        response_text = getattr(response, "text", None)

        if not response_text:
            raise GeminiVisionModelError(
                "Gemini API response contained no text content."
            )

        return self._parse_response(response_text)