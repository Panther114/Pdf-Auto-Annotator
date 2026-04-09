# PDF Auto-Annotator

Automatically highlight and annotate PDF documents using free LLMs — no paid API required.

> **API Key notice:** The HuggingFace API key shipped in this repository (`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`) is a **test/demo key included intentionally for ease of setup**. Leaking it poses no security risk. You are encouraged to replace it with your own free key from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## What it does

Given any PDF, the annotator sends each page to a language model and asks it to:

1. **Highlight** important sentences in colour-coded categories you define (e.g. theme, character development, conflict).
2. **Add short freetext comments** at the top and bottom of each page summarising key ideas.

The annotated PDF is saved as a new file — the original is never modified.

---

## Quick start

```bash
# Install dependencies
pip install pymupdf huggingface_hub
```

### GUI (recommended)

```bash
python gui.py
```

1. Click **Browse...** next to *Input PDF* and select your file — the output path is filled in automatically.
2. Adjust the *Document context*, *Custom instructions*, API key, and model as needed.
3. Click **Start Annotation**.
4. Watch the progress bar and live log update in real time. When finished, a confirmation dialog appears and the session log is automatically saved next to the output PDF as `annotation_log_YYYYMMDD_HHMMSS.txt`.

### Command line

```bash
# Annotate with defaults (generic literary novel categories)
python annotator.py mybook.pdf mybook_annotated.pdf

# Use your own config (recommended)
cp config.example.json config.json
# edit config.json to match your book and categories...
python annotator.py mybook.pdf mybook_annotated.pdf --config config.json
```

---

## Command-line options

```
python annotator.py INPUT OUTPUT [options]

Positional arguments:
  input             Input PDF file path
  output            Output PDF file path

Options:
  --config FILE     Path to a JSON config file (see config.example.json)
  --api-key KEY     HuggingFace API key (overrides config)
  --model ID        HuggingFace model ID (overrides config)
  --context TEXT    Description of the document, e.g. "Hamlet by Shakespeare"
  --pages START-END Only annotate a page range, e.g. "1-10"
  --verbose, -v     Enable DEBUG-level logging
```

---

## Configuration

Copy `config.example.json` to `config.json` and customise it.  
Key fields:

| Field | Description |
|---|---|
| `api_key` | HuggingFace API key |
| `model` | Any HuggingFace Inference API chat model |
| `context` | Freeform description of the document fed to the LLM |
| `categories` | Dictionary of highlight categories with `label` and RGB `color` |
| `num_highlights` | `min`/`max` sentences to highlight per page |
| `num_comments` | Number of freetext comments per page (0–2) |
| `max_retries` | Retry attempts per page on LLM/parse failures |
| `sleep_between_pages` | Seconds to wait between pages (respects API rate limits) |
| `min_text_length` | Skip pages with fewer characters than this |

### Example: Catcher in the Rye

```json
{
  "context": "The Catcher in the Rye by J.D. Salinger",
  "categories": {
    "bildungsroman_quotes": { "color": [1, 1, 0],       "label": "Coming-of-age / Bildungsroman" },
    "expectations_quotes":  { "color": [0.6, 1, 0.6],   "label": "Social expectations and conformity" },
    "barriers_quotes":      { "color": [0.6, 0.8, 1],   "label": "Barriers to connection and isolation" }
  }
}
```

---

## Supported models

Any model available through the HuggingFace Inference API works. Recommended free options:

| Model | Notes |
|---|---|
| `meta-llama/Meta-Llama-3-8B-Instruct` | Default; fast and accurate |
| `mistralai/Mistral-7B-Instruct-v0.3` | Good alternative |
| `google/gemma-7b-it` | Lighter weight |

---

## Project structure

```
gui.py                GUI entry-point (tkinter)
annotator.py          Core logic + CLI entry-point
config.example.json   Annotated example configuration
annotator v0.3.py     Original prototype (kept for reference)
```

---

## Roadmap / future expansions

Ideas for improving the tool further:

- **Multi-PDF batch mode** — annotate an entire folder in one command
- **Additional annotation types** — margin notes, strikethrough, underline, bookmarks
- **OpenAI / Anthropic / Ollama backend** — swap the LLM provider via config
- **Fine-tuned annotation models** — small, fast models trained specifically for annotation tasks
- **Annotation templates** — shareable JSON presets for common curricula (AP Lit, IB, etc.)
- **Export to CSV/JSON** — extract all highlights and comments as structured data
- **Confidence scoring** — flag pages where the model was uncertain
- **Parallel page processing** — process multiple pages concurrently to speed up large books

---

## License

[MIT](LICENSE)
