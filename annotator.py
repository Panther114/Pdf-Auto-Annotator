"""
PDF Auto-Annotator - powered by free HuggingFace LLMs.

Two-phase process
-----------------
Phase 1 - Document Intelligence:
    The LLM analyses a representative sample of your PDF and decides which
    features are most worth annotating.  The number of features equals the
    number of colours in ``annotation_colors`` (default: 3).

Phase 2 - Per-page Annotation:
    Every page is highlighted and commented using the categories discovered in
    Phase 1.  Colours are assigned deterministically (colour[0] -> feature 1,
    colour[1] -> feature 2, ...).

Usage:
    python annotator.py input.pdf [output.pdf] [--config config.json] [options]

Run ``python annotator.py --help`` for the full option list.
"""

import argparse
import ast
import json
import logging
import re
import sys
import threading
import time
import types
from pathlib import Path

import fitz  # PyMuPDF
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

# Persistent error log written next to the running script (or CWD as fallback).
_LOG_FILE = Path(__file__).parent / "log.txt"
try:
    _file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    _file_handler.setLevel(logging.WARNING)
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.getLogger().addHandler(_file_handler)
except OSError:
    pass  # If the log file cannot be created, continue without it

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: dict = {
    "api_key": "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",
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "max_tokens": 4000,
    "max_retries": 3,
    "sleep_between_pages": 3,
    "min_text_length": 200,
    # Number of pages sampled during Phase 1 category discovery.
    # 0 = all pages. Values larger than the document page count also sample all pages.
    # Defaulting to 3 to keep the prompt short and avoid "Bad request" errors from the API.
    "discovery_sample_pages": 3,
    # What you want to get out of the annotation.
    "annotation_goal": "",
    # Optional hint about the document type - helps the LLM choose better categories.
    "document_type": "",
    # Vocabulary sophistication of annotations: 1 (simplest) to 5 (most advanced).
    "language_depth": 2,
    # Highlight colours (RGB, each value 0-1).
    # Phase 1 assigns discovered features to these colours in order.
    # Add, remove, or change colours here; the count also controls how many
    # distinct features the LLM will discover.
    "annotation_colors": [
        [1, 1, 0],      # Yellow  - feature 1
        [0.6, 1, 0.6],  # Green   - feature 2
        [0.6, 0.8, 1],  # Blue    - feature 3
    ],
    "num_highlights": {"min": 5, "max": 7},
    # Number of freetext comment boxes added per page (0-4).
    "num_comments": 2,
    # Append a human-readable legend page at the end of the annotated PDF.
    "append_legend": True,
}

# Language depth instructions injected into prompts (1 = simplest, 5 = most advanced).
_LANGUAGE_DEPTH_INSTRUCTIONS = types.MappingProxyType({
    1: "Use very simple, everyday language. Short sentences. No jargon whatsoever.",
    2: "Use clear, accessible language suitable for a general audience. Moderate vocabulary.",
    3: "Use standard professional language with appropriate field-specific terminology.",
    4: "Use sophisticated vocabulary and nuanced phrasing suitable for subject-matter specialists.",
    5: "Use advanced, expert-level vocabulary with precise technical terminology and complex analytical language.",
})

_MAX_COLORS = 6
_MAX_COMMENTS = 4

# Phase 1 sampling parameters (overridable via config; 0 = all pages)
_DISCOVERY_SAMPLE_PAGES = 0
_DISCOVERY_CHARS_PER_PAGE = 5000


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Merge a user JSON config file over DEFAULT_CONFIG."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if config_path:
        path = Path(config_path)
        if not path.exists():
            logger.error("Config file not found: %s", config_path)
            sys.exit(1)
        with open(path, encoding="utf-8") as fh:
            user_config = json.load(fh)
        _deep_update(config, user_config)
    return config


def _deep_update(base, override):
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def validate_config(config):
    """Raise ValueError with a clear message if the config is invalid."""
    if not config.get("api_key"):
        raise ValueError(
            "'api_key' is required. "
            "Get a free key at https://huggingface.co/settings/tokens"
        )

    colors = config.get("annotation_colors", [])
    if not isinstance(colors, list) or len(colors) < 1:
        raise ValueError("'annotation_colors' must be a non-empty list of RGB arrays.")
    if len(colors) > _MAX_COLORS:
        raise ValueError(
            "'annotation_colors' may have at most {} entries. Got {}.".format(
                _MAX_COLORS, len(colors)
            )
        )
    for i, c in enumerate(colors):
        if not (
            isinstance(c, (list, tuple))
            and len(c) == 3
            and all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0 for v in c)
        ):
            raise ValueError(
                "annotation_colors[{}] must be [R, G, B] with values in 0-1. Got: {}".format(
                    i, c
                )
            )

    nc = config.get("num_comments", 2)
    if not isinstance(nc, int) or nc < 0 or nc > _MAX_COMMENTS:
        raise ValueError(
            "'num_comments' must be an integer 0-{}. Got: {}".format(_MAX_COMMENTS, nc)
        )

    nh = config.get("num_highlights", {})
    if not isinstance(nh, dict) or "min" not in nh or "max" not in nh:
        raise ValueError("'num_highlights' must be an object with 'min' and 'max'.")
    if nh["min"] < 1 or nh["max"] < nh["min"]:
        raise ValueError(
            "'num_highlights.min' must be >= 1 and <= 'num_highlights.max'."
        )

    for field in ("max_retries", "sleep_between_pages", "min_text_length", "max_tokens"):
        val = config.get(field)
        if val is not None and (not isinstance(val, (int, float)) or float(val) < 0):
            raise ValueError("'{}' must be a non-negative number. Got: {}".format(field, val))

    depth = config.get("language_depth", 2)
    if not isinstance(depth, int) or depth < 1 or depth > 5:
        raise ValueError("'language_depth' must be an integer from 1 to 5. Got: {}".format(depth))


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _find_outer_json(text):
    """Return the first outermost JSON object or array in *text*.

    Uses bracket-counting so it correctly handles nested structures.
    String content (including escaped characters) is skipped so that
    braces/brackets inside strings are not counted.
    """
    # Strip markdown code fences before searching.
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    first_brace = text.find("{")
    first_bracket = text.find("[")

    if first_brace == -1 and first_bracket == -1:
        return None

    if first_brace == -1:
        start_idx = first_bracket
    elif first_bracket == -1:
        start_idx = first_brace
    else:
        start_idx = min(first_brace, first_bracket)

    open_char = text[start_idx]
    close_char = "}" if open_char == "{" else "]"

    depth = 0
    in_string = False
    i = start_idx
    while i < len(text):
        c = text[i]
        if c == "\\" and in_string:
            if i + 1 < len(text):
                i += 2  # skip the escaped character
            else:
                break
            continue
        if c == '"':
            in_string = not in_string
        elif not in_string:
            if c == open_char:
                depth += 1
            elif c == close_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx : i + 1]
        i += 1

    return None


def _repair_json(raw):
    """Apply heuristic repairs to fix common JSON formatting issues from LLMs.

    Fixes applied (in order):
    1. Strip markdown fences.
    2. Remove trailing commas before ``}`` or ``]``.
    3. Insert missing commas between adjacent string/value tokens.
    """
    # 1. Strip markdown fences.
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()

    # 2. Remove trailing commas before } or ]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    # 3. Insert missing commas between adjacent values that sit on consecutive
    #    lines without a comma.  Pattern: end of a value token followed
    #    immediately (with only whitespace) by the start of a new value token.
    #    Value-end tokens: ", ], }, digits, true, false, null.
    #    Value-start tokens: ", [, {.
    raw = re.sub(
        r'("(?:[^"\\]|\\.)*"|\d[\d.eE+\-]*|true|false|null|\]|\})'
        r'(\s+)'
        r'("|\[|\{)',
        lambda m: m.group(1) + "," + m.group(2) + m.group(3),
        raw,
    )

    return raw


def extract_and_parse_json(text):
    """Extract and parse the first JSON object or array from *text*.

    Tries multiple strategies in order:
    1. Direct ``json.loads`` of the full (cleaned) text.
    2. Bracket-counted extraction of the outermost JSON block.
    3. Heuristic repairs (trailing commas, missing commas).
    4. ``ast.literal_eval`` for Python-style dict/list literals.

    Raises ``ValueError`` if all strategies fail.
    """
    # Clean markdown fences.
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()

    # Strategy 1: parse cleaned text directly.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: bracket-counted extraction.
    raw = _find_outer_json(cleaned)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Strategy 3: heuristic repairs on the extracted block.
        repaired = _repair_json(raw)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # Strategy 4: ast.literal_eval handles Python-style dicts/lists
        # (single quotes, trailing commas, etc.).
        try:
            result = ast.literal_eval(repaired)
            if isinstance(result, (dict, list)):
                return result
        except (ValueError, SyntaxError):
            pass

    # All strategies exhausted — raise with the original error message.
    try:
        json.loads(raw or cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Failed to parse model output as JSON: {}".format(exc)) from exc
    raise ValueError("No JSON object or array found in model response.")


# ---------------------------------------------------------------------------
# Phase 1 - Document Intelligence
# ---------------------------------------------------------------------------

def build_discovery_prompt(sample_text, config):
    """Return the prompt used in Phase 1 to discover annotation categories."""
    num_features = len(config["annotation_colors"])
    goal = (config.get("annotation_goal") or "").strip() or "identify the most important ideas"
    doc_type = (config.get("document_type") or "").strip() or "general document"
    depth = int(config.get("language_depth", 2))
    lang_instruction = _LANGUAGE_DEPTH_INSTRUCTIONS.get(depth, _LANGUAGE_DEPTH_INSTRUCTIONS[2])

    example_items = "\n".join(
        '    {{"key": "feature_{i}", "label": "One-sentence description of what to highlight"}}'.format(i=i)
        for i in range(1, num_features + 1)
    )

    return (
        "You are an expert document analyst preparing to annotate a PDF.\n\n"
        "Document type: {doc_type}\n"
        "Annotation goal: {goal}\n"
        "Language style for labels: {lang_instruction}\n\n"
        "Below is a representative sample of the document text:\n"
        "---\n"
        "{sample_text}\n"
        "---\n\n"
        "Task: Identify exactly {n} distinct, non-overlapping features that are\n"
        "most worth highlighting throughout this document, given the annotation goal above.\n\n"
        "Rules:\n"
        "- Each feature must be clearly different from the others (no overlap).\n"
        "- Use plain language that a first-time reader can understand.\n"
        "- The 'key' must be a short snake_case identifier (letters, digits, underscores only).\n"
        "- The 'label' is one sentence describing what kind of text to highlight.\n"
        "- Write labels at the language style indicated above.\n\n"
        "Respond ONLY with a valid JSON array of exactly {n} objects, like this:\n"
        "[\n"
        "{example_items}\n"
        "]"
    ).format(
        doc_type=doc_type,
        goal=goal,
        lang_instruction=lang_instruction,
        sample_text=sample_text,
        n=num_features,
        example_items=example_items,
    )


def discover_categories(doc, config, client, log_cb=None):
    """Phase 1 - ask the LLM to choose annotation features for this document.

    Returns a list like:
        [{"key": "main_argument", "label": "The author's central claim..."}, ...]

    List length equals len(config["annotation_colors"]).
    """

    def _log(level, msg):
        getattr(logger, level.lower(), logger.info)(msg)
        if log_cb:
            log_cb(level.upper(), msg)

    num_features = len(config["annotation_colors"])

    sample_limit = config.get("discovery_sample_pages", _DISCOVERY_SAMPLE_PAGES)
    page_count = len(doc)
    if sample_limit == 0 or sample_limit >= page_count:
        pages_to_sample = page_count
    else:
        pages_to_sample = sample_limit

    parts = []
    for page_num in range(pages_to_sample):
        text = doc[page_num].get_text().strip()
        if text:
            parts.append(text[:_DISCOVERY_CHARS_PER_PAGE])

    if not parts:
        _log("WARNING", "No text found for discovery - using generic categories.")
        return _fallback_categories(num_features)

    sample_text = "\n\n[next page]\n\n".join(parts)
    prompt = build_discovery_prompt(sample_text, config)
    _log(
        "INFO",
        "Discovering annotation categories from {} sample page(s) "
        "({} chars, prompt ~{} chars)...".format(
            len(parts), len(sample_text), len(prompt)
        ),
    )

    for attempt in range(config["max_retries"]):
        try:
            _log(
                "INFO",
                "Discovery attempt {}/{} - sending request to model '{}'...".format(
                    attempt + 1, config["max_retries"], config["model"]
                ),
            )
            response = client.chat_completion(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
            )
            raw = response.choices[0].message.content
            _log("INFO", "Discovery raw response ({} chars): {}".format(len(raw), raw[:300]))
            parsed = extract_and_parse_json(raw)

            if isinstance(parsed, list) and len(parsed) == num_features:
                categories = []
                for entry in parsed:
                    raw_key = str(entry.get("key", "")).lower()
                    key = re.sub(r"[^a-z0-9_]", "_", raw_key).strip("_") or "feature_{}".format(len(categories) + 1)
                    label = str(entry.get("label", "")).strip() or key
                    categories.append({"key": key, "label": label})
                _log(
                    "INFO",
                    "Discovered categories: " + ", ".join(c["key"] for c in categories),
                )
                return categories

            _log(
                "WARNING",
                "Attempt {}: unexpected response shape "
                "(expected list of {}, got {} - {}). Retrying...".format(
                    attempt + 1, num_features,
                    type(parsed).__name__,
                    repr(parsed)[:200],
                ),
            )

        except Exception as exc:
            exc_type = type(exc).__name__
            exc_msg = str(exc)
            _log(
                "WARNING",
                "Discovery attempt {} failed [{}]: {}".format(attempt + 1, exc_type, exc_msg),
            )
            # Extra hint for common failure modes
            if "400" in exc_msg or "bad request" in exc_msg.lower():
                _log(
                    "WARNING",
                    "  Hint: 'Bad request' often means the prompt is too long "
                    "or the model does not accept this input. "
                    "Try reducing 'discovery_sample_pages' or switching model.",
                )
            elif "401" in exc_msg or "unauthorized" in exc_msg.lower():
                _log("WARNING", "  Hint: API key may be invalid or expired.")
            elif "429" in exc_msg or "rate limit" in exc_msg.lower():
                _log("WARNING", "  Hint: Rate limited - will wait before retrying.")
            if attempt < config["max_retries"] - 1:
                wait = 2 * (attempt + 1)
                _log("INFO", "  Waiting {}s before next attempt...".format(wait))
                time.sleep(wait)

    _log("WARNING", "Discovery failed - falling back to generic categories.")
    return _fallback_categories(num_features)


def _fallback_categories(num_features):
    defaults = [
        {"key": "main_ideas",         "label": "Core ideas and central arguments"},
        {"key": "supporting_details", "label": "Evidence, examples, and supporting detail"},
        {"key": "key_terms",          "label": "Important terms, definitions, and concepts"},
        {"key": "conclusions",        "label": "Conclusions and implications"},
        {"key": "methodology",        "label": "Methods, processes, and procedures"},
        {"key": "notable_quotes",     "label": "Notable or quotable passages"},
    ]
    return defaults[:num_features]


# ---------------------------------------------------------------------------
# Phase 2 - Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(text, config, categories):
    """Build the per-page annotation prompt using categories from Phase 1."""
    num_highlights = config["num_highlights"]
    num_comments = config["num_comments"]
    goal = (config.get("annotation_goal") or "").strip() or "identify the most important ideas"
    doc_type = (config.get("document_type") or "").strip() or "document"
    depth = int(config.get("language_depth", 2))
    lang_instruction = _LANGUAGE_DEPTH_INSTRUCTIONS.get(depth, _LANGUAGE_DEPTH_INSTRUCTIONS[2])

    categories_desc = "\n".join(
        "  - {key}: sentences related to {label}".format(**cat)
        for cat in categories
    )

    json_template = {cat["key"]: ["exact sentence here"] for cat in categories}
    if num_comments >= 1:
        json_template["comment_top"] = "short choppy note here"
    if num_comments >= 2:
        json_template["comment_bottom"] = "another short choppy note"
    for i in range(3, num_comments + 1):
        json_template["comment_{}".format(i)] = "short note {}".format(i)

    prompt = (
        "You are an expert analyst annotating a page from a {doc_type}.\n"
        "Annotation goal: {goal}\n"
        "Language style: {lang_instruction}\n\n"
        "Task 1 - Highlights:\n"
        "Extract {nh_min} to {nh_max} important sentences from the text below.\n"
        "Categorize each sentence into one of these JSON arrays:\n"
        "{categories_desc}\n\n"
        "CRITICAL RULES FOR QUOTES:\n"
        "- Each quote must be an EXACT, word-for-word substring of the page text below.\n"
        "- Do NOT use double-quote characters inside string values; use single quotes (') instead.\n\n"
        "Task 2 - Comments:\n"
        "Write {nc} short, note-style comment(s) about this page.\n"
        "- Maximum 10-15 words each; incomplete sentences are fine.\n"
        "- Make each comment specific to this page's content.\n"
        "- Write comments at the language style indicated above.\n"
    ).format(
        doc_type=doc_type,
        goal=goal,
        lang_instruction=lang_instruction,
        nh_min=num_highlights["min"],
        nh_max=num_highlights["max"],
        categories_desc=categories_desc,
        nc=num_comments,
    )

    extra = (config.get("_custom_prompt") or "").strip()
    if extra:
        prompt += "\nAdditional instructions:\n{}\n".format(extra)

    retry_hint = (config.get("_retry_hint") or "").strip()
    if retry_hint:
        prompt += "\n{}\n".format(retry_hint)

    prompt += (
        "\nRespond ONLY with a valid JSON object in exactly this format:\n"
        + json.dumps(json_template, indent=4)
        + "\n\nPage Text:\n"
        + text
        + "\n"
    )
    return prompt


# ---------------------------------------------------------------------------
# Core annotation logic
# ---------------------------------------------------------------------------

def annotate_page(page, data, config, categories):
    """Apply highlights and freetext comments to a page."""
    colors = config["annotation_colors"]
    num_comments = config["num_comments"]

    # Track already-highlighted spans to avoid duplicates
    seen_spans = set()

    # Highlights
    for idx, cat in enumerate(categories):
        color = tuple(colors[idx % len(colors)])
        quotes = data.get(cat["key"], [])
        if isinstance(quotes, str):
            quotes = [quotes]

        for quote in quotes:
            if not quote or len(quote) < 5:
                continue
            clean = quote.replace("\n", " ").strip()
            for inst in page.search_for(clean):
                span = (
                    round(inst.x0, 1), round(inst.y0, 1),
                    round(inst.x1, 1), round(inst.y1, 1),
                )
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                annot = page.add_highlight_annot(inst)
                annot.set_colors(stroke=color)
                annot.update()

    # Top comment
    if num_comments >= 1:
        text = data.get("comment_top", "")
        if text:
            page.add_freetext_annot(
                fitz.Rect(40, 20, 300, 60), text,
                fontsize=11, text_color=(0, 0, 0), fill_color=(1, 1, 0.8),
            ).update()

    # Bottom comment
    if num_comments >= 2:
        text = data.get("comment_bottom", "")
        if text:
            h = page.rect.height
            page.add_freetext_annot(
                fitz.Rect(300, h - 60, 560, h - 20), text,
                fontsize=11, text_color=(0, 0, 0), fill_color=(0.8, 1, 1),
            ).update()

    # Extra comments (3+)
    for i in range(3, num_comments + 1):
        text = data.get("comment_{}".format(i), "")
        if text:
            h = page.rect.height
            offset = (i - 2) * 50
            page.add_freetext_annot(
                fitz.Rect(40, h - 60 - offset, 300, h - 20 - offset), text,
                fontsize=11, text_color=(0, 0, 0), fill_color=(0.9, 0.9, 1.0),
            ).update()


def append_legend_page(doc, categories, config):
    """Append a human-readable legend page listing features and their colours."""
    colors = config["annotation_colors"]
    goal = (config.get("annotation_goal") or "").strip() or "-"
    doc_type = (config.get("document_type") or "").strip() or "-"

    page = doc.new_page(-1)
    h = page.rect.height
    grey = (0.35, 0.35, 0.35)

    page.insert_text((50, 70),  "Annotation Legend", fontsize=22, color=(0, 0, 0))
    page.insert_text((50, 105), "Document type:   {}".format(doc_type), fontsize=11, color=grey)
    page.insert_text((50, 123), "Annotation goal: {}".format(goal),     fontsize=11, color=grey)

    y = 163
    page.insert_text((50, y), "Highlight Categories:", fontsize=13, color=(0, 0, 0))
    y += 28

    for idx, cat in enumerate(categories):
        rgb = tuple(colors[idx % len(colors)])
        page.draw_rect(fitz.Rect(50, y - 12, 78, y + 4), color=rgb, fill=rgb)
        page.insert_text(
            (88, y),
            "{}  -  {}".format(cat["key"], cat["label"]),
            fontsize=11, color=(0, 0, 0),
        )
        y += 30

    nc = config.get("num_comments", 2)
    y += 10
    page.insert_text(
        (50, y),
        "Each page also has {} freetext comment(s) summarising key ideas.".format(nc),
        fontsize=11, color=grey,
    )
    page.insert_text(
        (50, h - 40),
        "Generated by PDF Auto-Annotator - github.com/Panther114/Pdf-Auto-Annotator",
        fontsize=9, color=(0.65, 0.65, 0.65),
    )


def compute_output_path(input_pdf):
    """Return default output path: <input_dir>/annotated/<stem>_annotated.pdf"""
    p = Path(input_pdf)
    out_dir = p.parent / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / (p.stem + "_annotated.pdf")


# ---------------------------------------------------------------------------
# Main annotation pipeline
# ---------------------------------------------------------------------------

def annotate_pdf(
    input_pdf,
    output_pdf,
    config,
    categories=None,
    progress_cb=None,
    log_cb=None,
    cancel_event=None,
):
    """Annotate *input_pdf* and write the result to *output_pdf*.

    If *categories* is None, Phase 1 (discovery) runs automatically.
    Pass pre-computed categories to skip Phase 1.

    Returns a list of per-page result dicts for structured logging.

    Parameters
    ----------
    progress_cb  : callable(current: int, total: int)
    log_cb       : callable(level: str, message: str)
    cancel_event : threading.Event  - set to request cancellation
    """

    def _log(level, msg):
        getattr(logger, level.lower(), logger.info)(msg)
        if log_cb:
            log_cb(level.upper(), msg)

    def _cancelled():
        return cancel_event is not None and cancel_event.is_set()

    validate_config(config)

    client = InferenceClient(api_key=config["api_key"])

    # API-key sanity check - fast fail before processing any pages
    _log("INFO", "Validating API key...")
    try:
        client.chat_completion(
            model=config["model"],
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        _log("INFO", "API key is valid.")
    except Exception as exc:
        raise RuntimeError(
            "API key validation failed: {}\n"
            "Check your key at https://huggingface.co/settings/tokens".format(exc)
        ) from exc

    doc = fitz.open(input_pdf)
    total = len(doc)

    # Phase 1 - discover categories if not already provided
    if categories is None:
        categories = discover_categories(doc, config, client, log_cb=log_cb)

    if _cancelled():
        doc.close()
        return []

    page_range = config.get("_page_range", range(total))
    page_results = []

    page_list = list(page_range)
    pages_to_annotate = len(page_list)
    for idx, page_num in enumerate(page_list):
        if _cancelled():
            _log("INFO", "Annotation cancelled by user.")
            break

        if progress_cb:
            progress_cb(idx + 1, pages_to_annotate)

        _log("INFO", "Page {}/{} (PDF page {}) ...".format(idx + 1, pages_to_annotate, page_num + 1))
        page = doc[page_num]
        text = page.get_text()

        if len(text.strip()) < config["min_text_length"]:
            _log("INFO", "  Skipping - not enough text.")
            page_results.append({"page": page_num + 1, "status": "skipped", "reason": "not enough text"})
            continue

        # Token budget guard (~4 chars per token; keep headroom for template)
        max_chars = int(config["max_tokens"] * 0.75 * 4)
        if len(text) > max_chars:
            _log("WARNING", "  Page text truncated from {} to {} chars.".format(len(text), max_chars))
            text = text[:max_chars]

        page_start = time.monotonic()
        success = False
        is_last_page = (idx == len(page_list) - 1)

        for attempt in range(config["max_retries"]):
            if _cancelled():
                break

            # Smarter retry hint on subsequent attempts
            if attempt > 0:
                config["_retry_hint"] = (
                    "Your previous response was not valid JSON. "
                    "Respond ONLY with the JSON object - no explanation, no markdown."
                )
            else:
                config.pop("_retry_hint", None)

            try:
                prompt = build_prompt(text, config, categories)
                response = client.chat_completion(
                    model=config["model"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config["max_tokens"],
                )
                raw_text = response.choices[0].message.content
                data = extract_and_parse_json(raw_text)
                annotate_page(page, data, config, categories)
                _log("INFO", "  Annotated successfully.")
                elapsed = round(time.monotonic() - page_start, 2)
                page_results.append({"page": page_num + 1, "status": "ok", "elapsed_s": elapsed})
                # Skip the inter-page sleep on the last page to avoid unnecessary delay.
                if not is_last_page:
                    time.sleep(config["sleep_between_pages"])
                success = True
                break

            except Exception as exc:
                err_str = str(exc)
                # Adaptive sleep for rate-limit errors
                if "429" in err_str or "rate limit" in err_str.lower():
                    backoff = 5 * (2 ** attempt)
                    _log("WARNING", "  Rate limited - backing off {}s...".format(backoff))
                    time.sleep(backoff)
                else:
                    _log(
                        "WARNING",
                        "  Attempt {}/{} failed: {}".format(attempt + 1, config["max_retries"], exc),
                    )
                    time.sleep(2)

        if not success and not _cancelled():
            _log(
                "ERROR",
                "  All {} attempts failed for page {}. Skipping.".format(
                    config["max_retries"], page_num + 1
                ),
            )
            page_results.append({"page": page_num + 1, "status": "failed",
                                  "attempts": config["max_retries"]})

    # Append legend page
    if config.get("append_legend", True) and not _cancelled():
        append_legend_page(doc, categories, config)
        _log("INFO", "Appended annotation legend page.")

    doc.save(output_pdf)
    if progress_cb:
        progress_cb(pages_to_annotate, pages_to_annotate)
    _log("INFO", "Done! Saved annotated PDF as: {}".format(output_pdf))

    return page_results


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="annotator",
        description="PDF Auto-Annotator - highlight and comment PDFs with free LLMs.",
    )
    parser.add_argument("input", help="Input PDF file path")
    parser.add_argument(
        "output", nargs="?", default=None,
        help=(
            "Output PDF path (optional). "
            "Defaults to <input_dir>/annotated/<name>_annotated.pdf"
        ),
    )
    parser.add_argument(
        "--config", metavar="FILE",
        help="Path to a JSON config file (see config.example.json)",
    )
    parser.add_argument("--api-key", help="HuggingFace API key (overrides config)")
    parser.add_argument("--model",   help="HuggingFace model ID (overrides config)")
    parser.add_argument(
        "--goal", dest="annotation_goal",
        help='What you want to get from annotation, e.g. "study for a law school exam"',
    )
    parser.add_argument(
        "--document-type", dest="document_type",
        help='Type of document, e.g. "scientific paper", "legal brief", "novel"',
    )
    parser.add_argument(
        "--language-depth", dest="language_depth", type=int, choices=range(1, 6),
        metavar="LEVEL",
        help=(
            "Vocabulary sophistication of annotations: 1 (simplest) to 5 (most advanced). "
            "Default: 2"
        ),
    )
    parser.add_argument(
        "--pages", metavar="START-END",
        help='Annotate only a page range, e.g. "1-10" (1-indexed, inclusive)',
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run Phase 1 (category discovery) only and print results, then exit.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser


def parse_page_range(spec, total):
    """Parse a '1-10' style range string into a Python range."""
    try:
        parts = spec.split("-")
        start = int(parts[0]) - 1
        end = int(parts[1]) if len(parts) > 1 else start + 1
        return range(max(0, start), min(end, total))
    except (ValueError, IndexError):
        logger.error("Invalid --pages value: '%s'. Expected format: START-END (e.g. 1-10).", spec)
        sys.exit(1)


def main():
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
    if args.annotation_goal:
        config["annotation_goal"] = args.annotation_goal
    if args.document_type:
        config["document_type"] = args.document_type
    if args.language_depth is not None:
        config["language_depth"] = args.language_depth

    try:
        validate_config(config)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    output = args.output or str(compute_output_path(args.input))

    if args.pages:
        doc_probe = fitz.open(args.input)
        config["_page_range"] = parse_page_range(args.pages, len(doc_probe))
        doc_probe.close()

    # Dry-run: discover categories and exit
    if args.dry_run:
        client = InferenceClient(api_key=config["api_key"])
        doc = fitz.open(args.input)
        cats = discover_categories(doc, config, client)
        doc.close()
        print("\nDiscovered annotation categories:")
        for i, cat in enumerate(cats, 1):
            color = config["annotation_colors"][(i - 1) % len(config["annotation_colors"])]
            color_str = "RGB({:.2f}, {:.2f}, {:.2f})".format(*color)
            print("  {}. [{}]  {}  ({})".format(i, cat["key"], cat["label"], color_str))
        return

    annotate_pdf(args.input, output, config)


if __name__ == "__main__":
    main()
