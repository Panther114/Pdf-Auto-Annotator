# PDF Auto-Annotator

Automatically highlight and comment any PDF using free HuggingFace LLMs.
No paid API needed — just a free HuggingFace account.

---

## How it works

The tool uses a **two-phase** approach:

| Phase | What happens |
|-------|-------------|
| **1 — Document Intelligence** | The LLM reads a sample of your PDF and decides *which features are most worth highlighting* for your stated goal. |
| **2 — Per-page Annotation** | Every page is highlighted in colour-coded categories and annotated with short freetext comments. |

Colours are assigned deterministically: colour 1 → feature 1, colour 2 → feature 2, etc.
You can customise the colours (and their count) in `config.json`.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a free API key

Sign up at <https://huggingface.co> → Settings → Access Tokens → New token (read access is enough).

### 3. Run the GUI

```bash
python gui.py
```

**Step 1** — Pick your PDF, set your annotation goal and document type, then click **"Step 1: Discover Categories"**.
The app will show you the colour-coded features it found.

**Step 2** — Click **"Step 2: Confirm & Annotate"**.
The annotated PDF is saved automatically next to your input file in an `annotated/` folder.

---

## GUI walkthrough

| Field | What to enter |
|-------|---------------|
| **Input PDF** | The PDF you want to annotate |
| **Output PDF** | Auto-filled — saved to `<same folder>/annotated/<name>_annotated.pdf` |
| **Annotation goal** | What you want to get out of reading this document, e.g. *"study for a bar exam"* |
| **Document type** | Optional hint, e.g. *"legal brief"*, *"scientific paper"*, *"novel"* |
| **Page range** | Annotate only a subset of pages (leave "to" at 0 for all pages) |
| **Custom notes** | Extra instructions appended to every prompt |
| **API Key** | Your HuggingFace token |
| **Model** | LLM to use (pick from the dropdown or type any HF model ID) |
| **Config file** | Optional JSON config that overrides any of the above |

After **Step 1**, the discovered categories are shown with colour swatches.
Use **Re-run Discovery** if you want to try again with different settings.

Clicking **Cancel** at any time stops the current operation cleanly.

---

## Command-line usage

```bash
# Basic — output goes to annotated/<name>_annotated.pdf automatically
python annotator.py my_paper.pdf

# Explicit output path
python annotator.py my_paper.pdf my_paper_highlighted.pdf

# Set goal and document type
python annotator.py my_paper.pdf --goal "prepare for a PhD viva" --document-type "scientific paper"

# Annotate only pages 5-20
python annotator.py my_paper.pdf --pages 5-20

# Dry-run: discover categories only, print them, then exit
python annotator.py my_paper.pdf --dry-run

# Use a custom config file
python annotator.py my_paper.pdf --config config.json

# Full help
python annotator.py --help
```

---

## Configuration

Copy `config.example.json` to `config.json` and edit as needed.

```json
{
  "api_key": "hf_YOUR_KEY_HERE",
  "annotation_goal": "study for a law school exam",
  "document_type": "legal brief",

  "annotation_colors": [
    [1,   1,   0  ],
    [0.6, 1,   0.6],
    [0.6, 0.8, 1  ]
  ],

  "num_comments": 2,
  "num_highlights": { "min": 5, "max": 7 },
  "append_legend": true
}
```

### Colour customisation

`annotation_colors` is a list of `[R, G, B]` arrays where each value is **0–1**.
- The **number of colours** controls how many distinct features Phase 1 will discover.
- You may have **1–6 colours**.
- Phase 1 assigns features to colours in order (colour[0] → feature 1, etc.).

| RGB value | Approximate colour |
|-----------|--------------------|
| `[1, 1, 0]` | Yellow |
| `[0.6, 1, 0.6]` | Light green |
| `[0.6, 0.8, 1]` | Light blue |
| `[1, 0.7, 0.5]` | Peach/orange |
| `[1, 0.6, 0.8]` | Pink |

### `num_comments`

Integer 0–4. Controls how many freetext comment boxes appear on each page.

### `append_legend`

`true` (default) — appends a legend page at the end of the annotated PDF that
lists every category and its colour.

---

## Output

- **Annotated PDF** — saved to `annotated/<original_name>_annotated.pdf` next to the input file.
- **Text log** — `annotation_log_<timestamp>.txt` saved alongside the PDF.
- **JSON log** — `annotation_log_<timestamp>.json` saved alongside the PDF — useful for scripting or debugging.

---

## Supported models

Any instruction-tuned chat model available on HuggingFace Inference API works.
Recommended free models:

- `meta-llama/Meta-Llama-3-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `google/gemma-7b-it`

---

## Requirements

- Python 3.10+
- `pymupdf >= 1.23.0`
- `huggingface_hub >= 0.20.0`
- `tkinter` (bundled with CPython on Windows/macOS; on Linux: `sudo apt install python3-tk`)
