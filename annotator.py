"""
PDF Auto-Annotator — powered by free HuggingFace LLMs.

Usage:
    python annotator.py input.pdf output.pdf [--config config.json] [options]

Run `python annotator.py --help` for the full option list.
"""

import fitz  # PyMuPDF
import json
import ast
import time
import re
import argparse
import logging
import sys
from pathlib import Path
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — used when no config file is supplied
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict = {
    "api_key": "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",
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "max_tokens": 800,
    "max_retries": 3,
    "sleep_between_pages": 3,
    "min_text_length": 200,
    "context": "a literary novel",
    "categories": {
        "theme_quotes": {
            "color": [1, 1, 0],
            "label": "Central theme",
        },
        "character_quotes": {
            "color": [0.6, 1, 0.6],
            "label": "Character development",
        },
        "conflict_quotes": {
            "color": [0.6, 0.8, 1],
            "label": "Conflict or tension",
        },
    },
    "num_highlights": {"min": 5, "max": 7},
    "num_comments": 2,
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str | None) -> dict:
    """Merge user config file over DEFAULT_CONFIG."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy via JSON
    if config_path:
        path = Path(config_path)
        if not path.exists():
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)
        with open(path, encoding="utf-8") as fh:
            user_config = json.load(fh)
        _deep_update(config, user_config)
    return config


def _deep_update(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def extract_and_parse_json(text: str) -> dict:
    """Safely extract JSON even if the model garbles quotes or commas."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")

    clean_text = match.group(0)

    # Normalise embedded newlines so both parsers see clean string literals
    clean_text = clean_text.replace("\n", "\\n")

    # Strict JSON first
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass

    # Fallback: ast is more forgiving with Python-style dicts
    try:
        return ast.literal_eval(clean_text)
    except Exception as exc:
        raise ValueError(f"Failed to parse model output as JSON/dict: {exc}") from exc


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(text: str, config: dict) -> str:
    categories: dict = config["categories"]
    num_highlights: dict = config["num_highlights"]
    num_comments: int = config["num_comments"]
    context: str = config["context"]

    categories_desc = "\n".join(
        f"  - {key}: sentences related to {val['label']}"
        for key, val in categories.items()
    )

    json_template: dict = {key: ["exact sentence here"] for key in categories}
    if num_comments >= 1:
        json_template["comment_top"] = "short choppy note here"
    if num_comments >= 2:
        json_template["comment_bottom"] = "another short choppy note"

    base = f"""You are an experienced reader annotating a page from {context}.

Task 1 — Highlights:
Extract {num_highlights['min']} to {num_highlights['max']} important sentences from the text below.
Categorize them into the JSON arrays described here:
{categories_desc}

CRITICAL RULES FOR QUOTES:
- Quotes must be EXACT, word-for-word substrings copied from the page text.
- Do NOT use literal double-quote characters inside string values; replace them with single quotes (').

Task 2 — Comments:
Write {num_comments} short, academic, note-style comment(s) about this page.
- Maximum 10–15 words each; incomplete sentences are fine.
- Make each comment specific to this page's events, dialogue, or ideas.
"""

    extra_instructions = config.get("_custom_prompt", "").strip()
    if extra_instructions:
        base += f"\nAdditional instructions from the user:\n{extra_instructions}\n"

    base += f"""
Respond ONLY with a valid JSON object in exactly this format:
{json.dumps(json_template, indent=4)}

Page Text:
{text}
"""
    return base


# ---------------------------------------------------------------------------
# Core annotation logic
# ---------------------------------------------------------------------------

def annotate_page(page: fitz.Page, data: dict, config: dict) -> None:
    """Apply highlights and freetext comments to a single page."""
    categories: dict = config["categories"]
    num_comments: int = config["num_comments"]

    # 1. Categorised highlights
    for category, cat_cfg in categories.items():
        color = tuple(cat_cfg["color"])
        quotes = data.get(category, [])
        if isinstance(quotes, str):
            quotes = [quotes]

        for quote in quotes:
            if not quote or len(quote) < 5:
                continue
            clean_quote = quote.replace("\n", " ").strip()
            for inst in page.search_for(clean_quote):
                annot = page.add_highlight_annot(inst)
                annot.set_colors(stroke=color)
                annot.update()

    # 2. Top comment
    if num_comments >= 1:
        top_comment = data.get("comment_top", "")
        if top_comment:
            rect = fitz.Rect(40, 20, 300, 60)
            annot = page.add_freetext_annot(
                rect, top_comment, fontsize=11,
                text_color=(0, 0, 0), fill_color=(1, 1, 0.8),
            )
            annot.update()

    # 3. Bottom comment
    if num_comments >= 2:
        bottom_comment = data.get("comment_bottom", "")
        if bottom_comment:
            h = page.rect.height
            rect = fitz.Rect(300, h - 60, 560, h - 20)
            annot = page.add_freetext_annot(
                rect, bottom_comment, fontsize=11,
                text_color=(0, 0, 0), fill_color=(0.8, 1, 1),
            )
            annot.update()


def annotate_pdf(input_pdf: str, output_pdf: str, config: dict,
                 *, progress_cb=None, log_cb=None) -> None:
    """Annotate *input_pdf* and write to *output_pdf*.

    Optional keyword arguments
    --------------------------
    progress_cb : callable(current: int, total: int) | None
        Called after each page attempt with the 1-based page number and
        total page count so callers can drive a progress bar.
    log_cb : callable(level: str, message: str) | None
        Called alongside the standard logger for every info/warning/error
        message.  *level* is one of ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
    """

    def _log(level: str, msg: str) -> None:
        getattr(logger, level.lower(), logger.info)(msg)
        if log_cb:
            log_cb(level.upper(), msg)

    client = InferenceClient(api_key=config["api_key"])
    doc = fitz.open(input_pdf)
    total = len(doc)

    page_range = config.get("_page_range", range(total))

    for page_num in page_range:
        if progress_cb:
            progress_cb(page_num + 1, total)
        _log("INFO", f"Page {page_num + 1}/{total} …")
        page = doc[page_num]
        text = page.get_text()

        if len(text.strip()) < config["min_text_length"]:
            _log("INFO", "  Skipping — not enough text.")
            continue

        prompt = build_prompt(text, config)
        success = False

        for attempt in range(config["max_retries"]):
            try:
                response = client.chat_completion(
                    model=config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config["max_tokens"],
                )
                raw_text = response.choices[0].message.content
                data = extract_and_parse_json(raw_text)
                annotate_page(page, data, config)
                _log("INFO", "  ✅ Annotated successfully.")
                time.sleep(config["sleep_between_pages"])
                success = True
                break

            except Exception as exc:
                _log(
                    "WARNING",
                    f"  ⚠️  Attempt {attempt + 1}/{config['max_retries']} failed: {exc}",
                )
                time.sleep(2)

        if not success:
            _log(
                "ERROR",
                f"  ❌ All {config['max_retries']} attempts failed for page "
                f"{page_num + 1}. Skipping.",
            )

    doc.save(output_pdf)
    if progress_cb:
        progress_cb(total, total)
    _log("INFO", f"Done! Saved annotated PDF as: {output_pdf}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="annotator",
        description="PDF Auto-Annotator — highlight and comment PDFs with free LLMs.",
    )
    parser.add_argument("input", help="Input PDF file path")
    parser.add_argument("output", help="Output PDF file path")
    parser.add_argument(
        "--config", metavar="FILE",
        help="Path to a JSON config file (see config.example.json)",
    )
    parser.add_argument("--api-key", help="HuggingFace API key (overrides config)")
    parser.add_argument("--model", help="HuggingFace model ID (overrides config)")
    parser.add_argument(
        "--context",
        help='Description of the document, e.g. "The Catcher in the Rye" (overrides config)',
    )
    parser.add_argument(
        "--pages", metavar="START-END",
        help='Only annotate a page range, e.g. "1-10" (1-indexed, inclusive)',
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser


def parse_page_range(spec: str, total: int) -> range:
    """Parse a '1-10' style range string into a Python range."""
    try:
        parts = spec.split("-")
        start = int(parts[0]) - 1  # convert to 0-indexed
        end = int(parts[1]) if len(parts) > 1 else start + 1
        return range(max(0, start), min(end, total))
    except (ValueError, IndexError):
        logger.error(f"Invalid --pages value: '{spec}'. Expected format: START-END (e.g. 1-10).")
        sys.exit(1)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)

    # CLI overrides
    if args.api_key:
        config["api_key"] = args.api_key
    if args.model:
        config["model"] = args.model
    if args.context:
        config["context"] = args.context

    # Page-range support
    if args.pages:
        doc_probe = fitz.open(args.input)
        page_range = parse_page_range(args.pages, len(doc_probe))
        doc_probe.close()
        config["_page_range"] = page_range

    annotate_pdf(args.input, args.output, config)


if __name__ == "__main__":
    main()
