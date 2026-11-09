"""
report_generator.py

Generates a self-contained HTML diagnostic report for a single
screenshot analysis, and opens it in the default browser.

DESIGN NOTES:
- Fully self-contained except for two Google Fonts (IBM Plex Sans /
  IBM Plex Mono) loaded over the network at view time; falls back to
  system fonts if offline, so the report still works.
- The screenshot itself is embedded as a base64 PNG data URI, with
  detected regions drawn as absolutely-positioned overlay boxes sized
  as percentages of the image dimensions - this keeps the overlay
  aligned regardless of how the browser scales the image.
- All model-generated and OCR-derived text is HTML-escaped before
  being inserted into the page, since it comes from an AI model and
  OCR output, not code we control.
"""

from __future__ import annotations

import base64
import html
import json
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import OUTPUT_DIR
from models.output_schema import Region, ScreenshotAnalysis

REPORT_FILENAME = "report.html"

# Region colors are semantic, not decorative - each matches a distinct
# part of the SOLIDWORKS UI so the overlay reads at a glance.
_REGION_COLORS = {
    "left_ui_panel": "#3B6EA5",
    "graphics_area": "#C97A2B",
    "status_bar": "#3E8E7E",
    "dialog": "#B23A3A",
}
_DEFAULT_REGION_COLOR = "#6B7280"


# ============================================================
# SMALL HTML-BUILDING HELPERS
# ============================================================

def _encode_image_base64(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("ascii")


def _region_overlay_html(
    regions: list[Region],
    width: int,
    height: int,
) -> str:
    if width <= 0 or height <= 0:
        return ""

    boxes = []

    for region in regions:
        bbox = region.bbox

        left_pct = bbox.x_min / width * 100
        top_pct = bbox.y_min / height * 100
        width_pct = (bbox.x_max - bbox.x_min) / width * 100
        height_pct = (bbox.y_max - bbox.y_min) / height * 100

        color = _REGION_COLORS.get(
            region.name, _DEFAULT_REGION_COLOR
        )
        label = html.escape(region.name.replace("_", " "))

        boxes.append(
            f'<div class="region-box" style="'
            f"left:{left_pct:.2f}%; top:{top_pct:.2f}%; "
            f"width:{width_pct:.2f}%; height:{height_pct:.2f}%; "
            f'border-color:{color};">'
            f'<span class="region-label" style="background:{color};">'
            f"{label}</span></div>"
        )

    return "\n".join(boxes)


def _confidence_tier_color(confidence: float) -> str:
    if confidence >= 0.75:
        return "#3E8E63"
    if confidence >= 0.4:
        return "#C97A2B"
    return "#B23A3A"


def _gauge_html(confidence: float) -> str:
    pct = max(0.0, min(1.0, confidence)) * 100
    color = _confidence_tier_color(confidence)

    ticks = "".join(
        f'<span style="left:{t}%;"></span>'
        for t in (0, 25, 50, 75, 100)
    )

    return (
        '<div class="gauge">'
        f'<div class="gauge-fill" style="width:{pct:.1f}%; '
        f'background:{color};"></div>'
        f'<div class="gauge-ticks">{ticks}</div>'
        "</div>"
    )


def _list_html(items: list, empty_message: str) -> str:
    if not items:
        return f'<p class="muted">{html.escape(empty_message)}</p>'

    rows = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in items
    )
    return f'<ul class="item-list">{rows}</ul>'


def _field_list_html(rows: list[tuple[str, str]]) -> str:
    parts = "".join(
        f"<dt>{html.escape(str(label))}</dt>"
        f"<dd>{html.escape(str(value))}</dd>"
        for label, value in rows
    )
    return f'<dl class="field-list">{parts}</dl>'


def _api_summary_html(api_data: Optional[dict]) -> str:
    if api_data is None:
        return '<p class="muted">Not available for this run.</p>'

    rows = [
        ("Document", api_data.get("DocumentTitle", "Unknown")),
        (
            "Features analyzed",
            len(api_data.get("Features", [])),
        ),
        ("Sketches", len(api_data.get("Sketches", {}))),
        ("Extrusions", len(api_data.get("Extrusions", {}))),
        (
            "Problems flagged",
            len(api_data.get("Problems", [])),
        ),
    ]

    return _field_list_html(rows)


# ============================================================
# PAGE TEMPLATE
# ============================================================

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{page_title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --paper: #F5F6F3;
    --panel: #FFFFFF;
    --ink: #1C232B;
    --ink-muted: #5B6672;
    --line: #D7DBD6;
    --accent: #C97A2B;
  }}

  * {{ box-sizing: border-box; }}

  body {{
    margin: 0;
    padding: 48px 24px 80px;
    background: var(--paper);
    color: var(--ink);
    font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;
    font-size: 15px;
    line-height: 1.55;
  }}

  .report {{
    max-width: 760px;
    margin: 0 auto;
  }}

  .mono {{
    font-family: "IBM Plex Mono", Consolas, "Courier New", monospace;
  }}

  header.masthead {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    border-bottom: 1px solid var(--line);
    padding-bottom: 16px;
    margin-bottom: 32px;
  }}

  header.masthead h1 {{
    font-size: 20px;
    font-weight: 600;
    margin: 0;
  }}

  header.masthead .meta {{
    font-size: 12px;
    color: var(--ink-muted);
    text-align: right;
  }}

  .capture-frame {{
    position: relative;
    width: 100%;
    border: 1px solid var(--line);
    background: #111;
    margin-bottom: 12px;
    line-height: 0;
  }}

  .capture-frame img {{
    width: 100%;
    display: block;
  }}

  .region-box {{
    position: absolute;
    border: 2px solid;
    pointer-events: none;
  }}

  .region-label {{
    position: absolute;
    top: -1px;
    left: -2px;
    transform: translateY(-100%);
    color: #fff;
    font-family: "IBM Plex Mono", monospace;
    font-size: 11px;
    padding: 2px 6px;
    white-space: nowrap;
  }}

  .capture-legend {{
    font-size: 12px;
    color: var(--ink-muted);
    margin-bottom: 40px;
  }}

  .verdict {{
    border-bottom: 1px solid var(--line);
    padding-bottom: 32px;
    margin-bottom: 32px;
  }}

  .verdict-figure {{
    display: flex;
    align-items: flex-end;
    gap: 16px;
    margin-bottom: 8px;
  }}

  .verdict-number {{
    font-family: "IBM Plex Mono", monospace;
    font-size: 56px;
    font-weight: 600;
    line-height: 1;
  }}

  .verdict-number .unit {{
    font-size: 24px;
    font-weight: 500;
    color: var(--ink-muted);
  }}

  .verdict-label {{
    font-size: 13px;
    color: var(--ink-muted);
    padding-bottom: 10px;
  }}

  .gauge {{
    position: relative;
    height: 8px;
    background: #E7E9E4;
    border: 1px solid var(--line);
    margin: 16px 0 24px;
  }}

  .gauge-fill {{
    position: absolute;
    top: 0; left: 0; bottom: 0;
  }}

  .gauge-ticks span {{
    position: absolute;
    top: -3px;
    width: 1px;
    height: 14px;
    background: var(--line);
  }}

  .verdict-statement {{
    font-size: 17px;
    margin: 0 0 16px;
    max-width: 62ch;
  }}

  .item-list {{
    margin: 0;
    padding-left: 20px;
  }}

  .item-list li {{
    margin-bottom: 4px;
  }}

  .muted {{
    color: var(--ink-muted);
    margin: 0;
  }}

  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 40px;
    margin-bottom: 32px;
  }}

  .two-col h2 {{
    font-size: 14px;
    font-weight: 600;
    margin: 0 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--line);
  }}

  .field-list {{
    margin: 0;
    display: grid;
    grid-template-columns: auto 1fr;
    column-gap: 12px;
    row-gap: 6px;
    font-size: 14px;
  }}

  .field-list dt {{
    color: var(--ink-muted);
    white-space: nowrap;
  }}

  .field-list dd {{
    margin: 0;
  }}

  .fix-statement {{
    font-size: 15px;
    margin: 0 0 12px;
  }}

  .fix-confidence {{
    font-family: "IBM Plex Mono", monospace;
    font-size: 12px;
    color: var(--ink-muted);
    margin-bottom: 12px;
  }}

  details.appendix {{
    border-top: 1px solid var(--line);
    padding: 16px 0;
  }}

  details.appendix summary {{
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    color: var(--ink-muted);
  }}

  details.appendix[open] summary {{
    margin-bottom: 16px;
  }}

  pre.raw-json {{
    background: var(--panel);
    border: 1px solid var(--line);
    padding: 16px;
    font-size: 12px;
    overflow-x: auto;
    max-height: 480px;
  }}

  footer {{
    margin-top: 40px;
    font-size: 12px;
    color: var(--ink-muted);
  }}

  @media (max-width: 640px) {{
    .two-col {{ grid-template-columns: 1fr; gap: 24px; }}
    header.masthead {{ flex-direction: column; gap: 8px; }}
    header.masthead .meta {{ text-align: left; }}
  }}
</style>
</head>
<body>
<div class="report">

  <header class="masthead">
    <h1>{document_title}</h1>
    <div class="meta mono">
      {screenshot_name}<br>
      generated {generated_timestamp}
    </div>
  </header>

  <div class="capture-frame">
    <img src="data:image/png;base64,{image_base64}" alt="Analyzed screenshot">
    {region_overlay}
  </div>
  <p class="capture-legend">Detected UI regions overlaid above.</p>

  <section class="verdict">
    <div class="verdict-figure">
      <div class="verdict-number">{confidence_pct:.0f}<span class="unit">%</span></div>
      <div class="verdict-label">confidence &mdash; problem identified<br>({vision_model_used})</div>
    </div>
    {gauge_html}
    <p class="verdict-statement">{detected_problem}</p>
    {evidence_html}
  </section>

  <section class="two-col">
    <div>
      <h2>Diagnosis detail</h2>
      {diagnosis_field_list}
    </div>
    <div>
      <h2>Suggested fix</h2>
      <p class="fix-statement">{suggested_fix}</p>
      <div class="fix-confidence mono">confidence {solution_confidence_pct:.0f}% &middot; {solution_model_used}</div>
      {reasoning_html}
    </div>
  </section>

  <details class="appendix">
    <summary>SolidWorks API data</summary>
    {api_summary_html}
  </details>

  <details class="appendix">
    <summary>Raw analysis JSON</summary>
    <pre class="raw-json mono">{raw_json}</pre>
  </details>

  <footer>SolidWorks screenshot pipeline &middot; report generated {generated_timestamp}</footer>

</div>
</body>
</html>
"""


# ============================================================
# BUILD + WRITE
# ============================================================

def build_report_html(
    analysis: ScreenshotAnalysis,
    screenshot_path: Path,
    image: np.ndarray,
) -> str:

    vision = analysis.vision_analysis
    solution = analysis.solution

    has_diagnosis = bool(
        vision and vision.detected_problem
    )

    confidence = vision.confidence if vision else 0.0

    detected_problem = (
        vision.detected_problem
        if has_diagnosis
        else "No confirmed problem for this run."
    )

    vision_model_used = (
        vision.model_used if vision else "none"
    )

    evidence_html = _list_html(
        vision.evidence if vision else [],
        "No evidence recorded.",
    )

    document_title = (
        analysis.solidworks_api_data.get(
            "DocumentTitle", screenshot_path.stem
        )
        if analysis.solidworks_api_data
        else screenshot_path.stem
    )

    evidence = analysis.problem_evidence

    diagnosis_field_list = _field_list_html(
        [
            ("Feature", evidence.primary_feature or "-"),
            ("Problem type", evidence.problem_type or "-"),
            (
                "Related features",
                ", ".join(evidence.related_features) or "-",
            ),
            (
                "Visible parameters",
                ", ".join(evidence.visible_parameter_values)
                or "-",
            ),
        ]
    )

    suggested_fix = (
        solution.suggested_fix
        if solution and solution.suggested_fix
        else "No suggested fix available for this run."
    )

    solution_confidence_pct = (
        (solution.confidence * 100) if solution else 0.0
    )

    solution_model_used = (
        solution.model_used if solution else "none"
    )

    reasoning_html = _list_html(
        solution.reasoning if solution else [],
        "No reasoning recorded.",
    )

    region_overlay = _region_overlay_html(
        analysis.regions,
        analysis.image.width,
        analysis.image.height,
    )

    raw_json = html.escape(
        json.dumps(analysis.to_dict(), indent=2)
    )

    return _TEMPLATE.format(
        page_title=html.escape(
            f"SOLIDWORKS diagnostic report - {screenshot_path.name}"
        ),
        document_title=html.escape(str(document_title)),
        screenshot_name=html.escape(screenshot_path.name),
        generated_timestamp=datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        image_base64=_encode_image_base64(image),
        region_overlay=region_overlay,
        confidence_pct=confidence * 100,
        vision_model_used=html.escape(vision_model_used),
        gauge_html=_gauge_html(confidence),
        detected_problem=html.escape(str(detected_problem)),
        evidence_html=evidence_html,
        diagnosis_field_list=diagnosis_field_list,
        suggested_fix=html.escape(str(suggested_fix)),
        solution_confidence_pct=solution_confidence_pct,
        solution_model_used=html.escape(solution_model_used),
        reasoning_html=reasoning_html,
        api_summary_html=_api_summary_html(
            analysis.solidworks_api_data
        ),
        raw_json=raw_json,
    )


def generate_html_report(
    analysis: ScreenshotAnalysis,
    screenshot_path: Path,
    image: np.ndarray,
    open_in_browser: bool = True,
) -> Path:
    """
    Build the HTML report and write it to OUTPUT_DIR/report.html,
    optionally opening it in the default browser.
    """

    html_content = build_report_html(
        analysis, screenshot_path, image
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / REPORT_FILENAME
    report_path.write_text(html_content, encoding="utf-8")

    if open_in_browser:
        webbrowser.open(report_path.resolve().as_uri())

    return report_path