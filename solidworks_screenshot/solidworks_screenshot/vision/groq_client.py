"""
vision/groq_client.py

Groq-backed implementation of VisionModelClient.

Uses Groq's free developer tier (no credit card required, rate-limited
per model) - see https://console.groq.com to create a key.

DESIGN NOTES:
- Same text-only approach as the project's earlier Gemini implementation:
  reasons entirely over evidence already extracted earlier in the
  pipeline (OCR text, regions, problem_evidence, solidworks_api_data),
  not the raw screenshot image.
- Requires the `groq` package (`pip install groq`) and a GROQ_API_KEY
  environment variable.
- Groq's chat API is OpenAI-compatible. Model choice matters here:
  llama-3.3-70b-versatile and llama-3.1-8b-instant were deprecated by
  Groq in June 2026 - openai/gpt-oss-120b is the current recommended
  free-tier model for general reasoning tasks like this one. Free tier
  as of this writing: 30 RPM, 1,000 RPD, 200,000 TPD. Check
  console.groq.com/docs/models if this changes.
- Same JSON-only response contract and forbidden-field stripping as
  the project's other model clients.
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
    from groq import Groq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


DEFAULT_MODEL = "openai/gpt-oss-120b"

MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0  # doubles each retry: 2s, then 4s

_TRANSIENT_ERROR_MARKERS = (
    "429",
    "rate_limit",
    "503",
    "UNAVAILABLE",
    "500",
    "internal_server_error",
)

_FORBIDDEN_RESPONSE_KEYS = {
    "solution",
    "suggested_action",
    "fix",
    "recommendation",
    "next_steps",
}


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


class GroqVisionModelError(Exception):
    """Raised when the Groq-backed vision model can't run or respond."""


class GroqVisionModel:
    """
    VisionModelClient implementation backed by the Groq API
    (free developer tier).

    Text-only: reasons over already-extracted evidence rather than the
    raw screenshot image. See module docstring for why.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        if not GROQ_AVAILABLE:
            raise GroqVisionModelError(
                "The 'groq' package is not installed. "
                "Run: pip install groq"
            )

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")

        if not resolved_key:
            raise GroqVisionModelError(
                "No Groq API key found. Set the GROQ_API_KEY "
                "environment variable (get one free, no card, at "
                "console.groq.com), or pass api_key= explicitly."
            )

        self._client = Groq(api_key=resolved_key)
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

        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise GroqVisionModelError(
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
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                )
                break

            except Exception as exc:

                is_last_attempt = attempt == MAX_RETRY_ATTEMPTS

                if not _is_transient_error(exc) or is_last_attempt:
                    raise GroqVisionModelError(
                        f"Groq API request failed: {exc}"
                    ) from exc

                wait_seconds = (
                    RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                )

                print(
                    f"Groq API temporarily unavailable "
                    f"(attempt {attempt}/{MAX_RETRY_ATTEMPTS}), "
                    f"retrying in {wait_seconds:.0f}s: {exc}"
                )

                time.sleep(wait_seconds)

        response_text = (
            response.choices[0].message.content
            if response and response.choices
            else None
        )

        if not response_text:
            raise GroqVisionModelError(
                "Groq API response contained no text content."
            )

        return self._parse_response(response_text)