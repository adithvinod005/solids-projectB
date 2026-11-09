"""
vision/solution_generator.py

Generates a suggested fix for an already-identified SOLIDWORKS problem.

DESIGN NOTES:
- This is a DELIBERATELY SEPARATE step from vision/analyzer.py's
  identification stage. It only runs after a diagnosis already exists
  (VisionAnalysisResult.detected_problem is not None) - it never
  re-identifies the problem itself, only reasons about how to fix the
  problem that's already been identified.
- Inputs: the confirmed diagnosis (VisionAnalysisResult), the
  screenshot-side problem_evidence, and solidworks_api_data (feature
  error codes, causes, dependencies from the C# analyzer) - the same
  data already gathered earlier in the pipeline, nothing new to
  extract.
- Uses the Groq API (free developer tier, no credit card), same
  provider and model as vision/groq_client.py, with the same
  retry-on-transient-error handling.
- Returns models.output_schema.SolutionResult - the one place in the
  schema solution/fix content is allowed (see output_schema.py).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from models.output_schema import (
    ProblemEvidence,
    SolutionResult,
    VisionAnalysisResult,
)

try:
    from groq import Groq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# Same model as vision/groq_client.py - llama-3.3-70b-versatile and
# llama-3.1-8b-instant were deprecated by Groq in June 2026;
# openai/gpt-oss-120b is the current recommended free-tier model.
# Free tier as of this writing: 30 RPM, 1,000 RPD, 200,000 TPD.
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


def _is_transient_error(exc: Exception) -> bool:
    text = str(exc)
    return any(marker in text for marker in _TRANSIENT_ERROR_MARKERS)


SYSTEM_PROMPT = """You are suggesting a fix for a SOLIDWORKS CAD problem that has ALREADY been identified.

You will be given:
- The confirmed diagnosis (what problem was identified, and the evidence for it)
- Problem evidence extracted from the screenshot (feature names, related features, visible parameters)
- SolidWorks API data, if available (feature names, types, error codes, and parent/child dependencies)

Your job is to suggest a concrete, actionable fix for THIS SPECIFIC problem - not generic SOLIDWORKS advice. Reference the actual feature names and parameter values given to you where relevant.

Respond with ONLY a JSON object in this exact shape, with no other text, no markdown code fences:

{
  "suggested_fix": "<one to three sentences describing a concrete fix, or null if no confident fix can be suggested>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": ["<short reason this fix should work>", "<short reason>", ...]
}
"""


class SolutionGeneratorError(Exception):
    """Raised when the solution generator can't run or respond."""


class GroqSolutionGenerator:
    """
    Generates a suggested fix using the Groq API (free developer tier),
    given an already-confirmed diagnosis.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        if not GROQ_AVAILABLE:
            raise SolutionGeneratorError(
                "The 'groq' package is not installed. "
                "Run: pip install groq"
            )

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")

        if not resolved_key:
            raise SolutionGeneratorError(
                "No Groq API key found. Set the GROQ_API_KEY "
                "environment variable, or pass api_key= explicitly."
            )

        self._client = Groq(api_key=resolved_key)
        self._model = model

    # ----------------------------------------------------------------
    # PROMPT BUILDING
    # ----------------------------------------------------------------

    def _build_user_prompt(
        self,
        diagnosis: VisionAnalysisResult,
        problem_evidence: ProblemEvidence,
        solidworks_api_data: Optional[dict[str, Any]],
    ) -> str:

        sections = [
            "CONFIRMED DIAGNOSIS:",
            json.dumps(diagnosis.to_dict(), indent=2),
            "",
            "PROBLEM EVIDENCE (from the screenshot):",
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

    def _parse_response(self, raw_text: str) -> SolutionResult:
        cleaned = raw_text.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise SolutionGeneratorError(
                f"Model response was not valid JSON: {exc}\n"
                f"Raw response: {raw_text}"
            ) from exc

        suggested_fix = parsed.get("suggested_fix")
        confidence = float(parsed.get("confidence", 0.0))
        reasoning = parsed.get("reasoning", [])

        if not isinstance(reasoning, list):
            reasoning = [str(reasoning)]

        return SolutionResult(
            suggested_fix=suggested_fix,
            confidence=max(0.0, min(1.0, confidence)),
            reasoning=[str(item) for item in reasoning],
            model_used=self._model,
        )

    # ----------------------------------------------------------------
    # PUBLIC INTERFACE
    # ----------------------------------------------------------------

    def generate(
        self,
        diagnosis: VisionAnalysisResult,
        problem_evidence: ProblemEvidence,
        solidworks_api_data: Optional[dict[str, Any]],
    ) -> SolutionResult:

        user_prompt = self._build_user_prompt(
            diagnosis, problem_evidence, solidworks_api_data
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
                    raise SolutionGeneratorError(
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
            raise SolutionGeneratorError(
                "Groq API response contained no text content."
            )

        return self._parse_response(response_text)


def generate_solution(
    diagnosis: Optional[VisionAnalysisResult],
    problem_evidence: ProblemEvidence,
    solidworks_api_data: Optional[dict[str, Any]],
    generator: Optional[GroqSolutionGenerator] = None,
) -> Optional[SolutionResult]:
    """
    Generate a suggested fix, given an already-confirmed diagnosis.

    Returns None (rather than an empty SolutionResult) if there's no
    diagnosis to act on yet - no point suggesting a fix for a problem
    that wasn't actually identified. Callers should check for this
    before printing/saving a "no solution" message.
    """

    if diagnosis is None or not diagnosis.detected_problem:
        return None

    active_generator = generator or GroqSolutionGenerator()

    return active_generator.generate(
        diagnosis, problem_evidence, solidworks_api_data
    )