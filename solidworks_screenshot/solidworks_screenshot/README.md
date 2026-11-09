# SOLIDWORKS Screenshot Analysis (Python) — V1 Foundation

This is the **screenshot-analysis half** of a two-part SOLIDWORKS
troubleshooting assistant. It takes a screenshot of the SOLIDWORKS
interface and produces structured JSON describing what's visible:
detected UI regions (feature tree, graphics area, PropertyManager, status
bar, dialogs), OCR'd text, and a set of high-level visual observations.

It does **not** talk to SOLIDWORKS directly — that's the job of the
separate C# application using the SOLIDWORKS API. This project is
designed to eventually receive that C# app's JSON output alongside its
own analysis, both feeding into an AI/vision model for final problem
diagnosis. See [Architecture](#architecture) below.

## What this project does NOT do (by design)

- It does not call SOLIDWORKS or its API.
- It does not run a chatbot or web UI.
- It does not call any AI/vision provider yet — `vision/analyzer.py`
  defines the interface a future model will plug into, but ships with a
  no-op stub (`NullVisionModel`) so the pipeline runs end-to-end today.
- It does not claim to reliably detect everything a real vision model or
  the SOLIDWORKS API could tell you (see [Limitations](#limitations)).
- **It does not generate solutions, fixes, or corrective actions —
  identification only.** See [Scope boundary](#scope-boundary-identification-only-never-solutions) below.

## Scope boundary: identification only, never solutions

This pipeline's job stops at describing **what** is visible and **what
problem** it points to. It must never produce **how to fix it**.

- The output schema has no `solution`, `suggested_action`, `fix`,
  `recommendation`, or `next_steps` field, and `models/output_schema.py`
  actively raises `SolutionFieldNotAllowedError` at serialization time if
  one is ever accidentally added.
- `vision/analyzer.py`'s `VisionModelClient` interface is documented to
  return identification only (`detected_problem`, `confidence`,
  `evidence`) — when a real model is plugged in later, prompt it to
  identify the problem and its visual evidence, and explicitly instruct
  it not to propose a fix.
- Deciding what to do about a detected problem is left to a downstream
  system or a human — out of scope for this project entirely.

## Installation

**Requirements:** Python 3.10+ (uses modern type hint syntax like `list[str]`).

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux (if testing outside Windows)
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR** (separate from the `pytesseract` Python
   package — `pytesseract` is just a wrapper around the actual Tesseract
   program).

   - **Windows:** Download and run the installer from the
     [UB-Mannheim Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki).
     During install, note the install path (default is usually
     `C:\Program Files\Tesseract-OCR\tesseract.exe`).
   - If Tesseract isn't on your system `PATH` after installing, point
     `pytesseract` at it explicitly by adding this near the top of
     `ocr/extractor.py`:

     ```python
     pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
     ```

   OCR is optional: if Tesseract isn't installed, the pipeline still runs
   and produces JSON — it just skips the `ocr` section and reports OCR as
   unavailable in the console summary. You can also explicitly skip it
   with `--no-ocr`.

## Usage

```bash
python main.py path/to/screenshot.png
```

This will:
1. Load and validate the screenshot.
2. Preprocess it (resize if needed, enhance for OCR).
3. Detect likely UI regions.
4. Run OCR on those regions (if Tesseract is available).
5. Write a structured JSON file to `output/analysis.json`.
6. Print a human-readable summary to the console.

**Options:**

```bash
python main.py screenshot.png --output custom/path.json   # choose output path
python main.py screenshot.png --no-ocr                     # skip OCR entirely
```

### Example console output

```
--- Screenshot Analysis Summary ---
Image size: 1920x1080

Detected regions:
  - feature_tree        confidence=0.80  bbox=[0, 54, 346, 1026]
  - property_manager     confidence=0.35  bbox=[0, 54, 346, 1026]
  - graphics_area        confidence=0.70  bbox=[346, 54, 1920, 972]
  - status_bar           confidence=0.50  bbox=[0, 1026, 1920, 1080]

OCR text found (4 items):
  - [feature_tree] "Extrude1" (confidence=91)
  - [status_bar] "Rebuild error due to geometric conditions" (confidence=95)
  ...

Visual observations:
  - error_message_visible: True
  - warning_message_visible: False
  - feature_tree_visible: True
  - property_manager_visible: False
  - graphics_area_visible: True
  - status_bar_visible: True
  - dialog_visible: False

Vision/AI analysis: (stub - no model connected yet)
  - detected_problem: None
------------------------------------

Full JSON output written to: ...\output\analysis.json
```

## Architecture

```
SOLIDWORKS
  │
  ├── Screenshot ───────────────┐
  │                             ▼
  │                    Python screenshot pipeline   <-- THIS PROJECT
  │                             │
  └── C# SOLIDWORKS API ──► JSON│
                             │  │
                             ▼  ▼
                     AI / Vision model
                             │
                             ▼
                   Problem identification
```

**This project covers only the "Python screenshot pipeline" box.**
Internally, it's split into four stages, each in its own module so they
can be improved or swapped independently:

```
screenshot.png
     │
     ▼
image/loader.py         → load + validate the file
     │
     ▼
image/preprocessing.py  → resize, grayscale, contrast/sharpen for OCR
     │
     ▼
image/regions.py        → heuristic detection of feature tree, graphics
     │                     area, PropertyManager, status bar, dialogs
     ▼
ocr/extractor.py         → Tesseract OCR, run per detected region
     │
     ▼
vision/analyzer.py       → clean interface for a future AI/vision model
     │                     (ships with a no-op stub for now)
     ▼
models/output_schema.py  → assembles everything into the final JSON
```

### Why regions are heuristic, not "smart"

SOLIDWORKS' panels can be docked, undocked, resized, or hidden, and we
only have pixels to work with — no access to the application's internal
window layout. `image/regions.py` uses simple, transparent heuristics
(expected screen position + edge-density checks + a brightness-based
dialog check), and every detected region carries a **confidence score**
rather than being treated as ground truth. This is intentional: the
project brief calls for a system an AI/vision model can improve upon
later, not a fragile hand-tuned CV pipeline pretending to be certain.

### The vision model interface

`vision/analyzer.py` defines:

```python
def analyze_screenshot(image, ocr_data, regions, model=None) -> VisionAnalysisResult:
    ...
```

Today, `model` defaults to `NullVisionModel`, which returns an empty,
zero-confidence result — explicitly signaling "no AI analysis happened
yet" rather than guessing. To plug in a real model later (OpenAI,
Anthropic, a local vision model, etc.), implement the `VisionModelClient`
protocol in a new file (e.g. `vision/openai_client.py`) and pass an
instance of it into `analyze_screenshot()`. Nothing else in the pipeline
needs to change.

### Adding the future C# API JSON

`models/output_schema.py`'s `ScreenshotAnalysis` already has a
`solidworks_api_data` field, left as `None` for now. When the C# app's
JSON becomes available, loading it and attaching it here (before handing
everything to the vision model) is a small, isolated change — the schema
was designed with this in mind from the start.

## Output schema

```jsonc
{
  "image": {
    "width": 1920,
    "height": 1080,
    "original_path": "C:/.../screenshot.png",
    "was_resized": false
  },
  "regions": [
    { "name": "feature_tree", "confidence": 0.8, "bbox": [0, 54, 346, 1026], "detection_method": "heuristic" }
  ],
  "ocr": [
    { "text": "Rebuild error due to geometric conditions", "confidence": 95.0, "bbox": [500, 700, 1100, 760], "region": "status_bar" }
  ],
  "visual_observations": {
    "error_message_visible": true,
    "warning_message_visible": false,
    "feature_tree_visible": true,
    "property_manager_visible": false,
    "graphics_area_visible": true,
    "status_bar_visible": true,
    "dialog_visible": false
  },
  "vision_analysis": {
    "detected_problem": null,
    "confidence": 0.0,
    "evidence": [],
    "model_used": "none"
  },
  "solidworks_api_data": null
}
```

## Limitations

A screenshot alone cannot reliably reveal everything about SOLIDWORKS'
internal state. This pipeline **cannot** tell you:

- exact feature error codes,
- complete feature dependency chains,
- exact sketch constraint state,
- hidden feature parameters.

That information is expected to come from the separate C# SOLIDWORKS API
application, and is designed to be merged in later via the
`solidworks_api_data` field — this Python project complements that data
rather than replacing it.

Similarly, region detection here is heuristic and openly imperfect;
low-confidence detections should be treated as hints for a future vision
model to confirm or correct, not as facts.

## Project structure

```
solidworks_screenshot/
├── main.py                    # CLI entry point
├── config.py                  # paths + tunable thresholds
├── image/
│   ├── loader.py               # load & validate screenshots
│   ├── preprocessing.py        # resize, grayscale, OCR enhancement
│   └── regions.py              # heuristic UI region detection
├── ocr/
│   └── extractor.py            # Tesseract-based text extraction
├── vision/
│   └── analyzer.py             # pluggable AI/vision model interface (stub)
├── models/
│   └── output_schema.py        # dataclasses defining the JSON output
├── output/                     # analysis.json lands here
├── screenshots/                # put test screenshots here
├── requirements.txt
└── README.md
```

## Roadmap (not part of this V1)

- Implement a concrete `VisionModelClient` (OpenAI/Anthropic/local) in
  `vision/`.
- Load and merge the C# SOLIDWORKS API JSON into `solidworks_api_data`.
- Combine screenshot + API JSON into one prompt for the vision model to
  produce a final `detected_problem`.
