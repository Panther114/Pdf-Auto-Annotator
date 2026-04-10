"""
PDF Auto-Annotator - Graphical User Interface

Run with:
    python gui.py

Requirements: same as annotator.py (pymupdf, huggingface_hub).
tkinter is part of the Python standard library.

Two-phase flow
--------------
1. Fill in the fields and click "Discover Categories".
   The app samples your PDF with the LLM and shows you what it found.
2. Review the colour-coded category list, then click "Confirm & Annotate"
   to annotate every page using those categories.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from annotator import (
    DEFAULT_CONFIG,
    annotate_pdf,
    compute_output_path,
    discover_categories,
    load_config,
    validate_config,
)

import fitz
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRESET_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-7b-it",
]

LOG_TAG_COLORS = {
    "ERROR":   "#cc0000",
    "WARNING": "#b35900",
    "SUCCESS": "#007700",
    "INFO":    "#1a1a1a",
}

# Swatch colours shown next to each discovered category in the preview panel.
# These must be valid Tk colour strings and should visually match the RGB
# values in DEFAULT_CONFIG["annotation_colors"].
SWATCH_TK_COLORS = ["#ffff00", "#99ff99", "#99ccff", "#ffcc99", "#ff99cc", "#ccccff"]

WINDOW_MIN_WIDTH  = 640
WINDOW_MIN_HEIGHT = 520

# Persistent error log written next to the running script.
LOG_FILE = Path(__file__).parent / "log.txt"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _rgb_to_tk(rgb):
    """Convert a [R, G, B] list (0-1 floats) to a '#rrggbb' Tk colour string."""
    r, g, b = (int(v * 255) for v in rgb)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class AnnotatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Auto-Annotator")
        self.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resizable(True, True)

        self._msg_queue = queue.Queue()
        self._worker_thread = None
        self._cancel_event = threading.Event()
        self._log_lines = []          # plain-text lines
        self._log_records = []        # structured dicts for JSON log
        self._discovered_categories = None
        self._output_path = None      # computed after input selected
        self._pdf_page_count = 0      # cached page count of the loaded PDF

        self._build_ui()
        self._poll_queue()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer_pad = {"padx": 8, "pady": 2}

        # -- Files -------------------------------------------------------
        file_frame = ttk.LabelFrame(self, text="Files", padding=6)
        file_frame.pack(fill="x", **outer_pad)
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="Input PDF:").grid(row=0, column=0, sticky="w", pady=2)
        self._input_var = tk.StringVar()
        self._input_var.trace_add("write", self._on_input_changed)
        ttk.Entry(file_frame, textvariable=self._input_var).grid(
            row=0, column=1, sticky="ew", padx=6)
        ttk.Button(file_frame, text="Browse...",
                   command=self._browse_input).grid(row=0, column=2)

        # Output path label (read-only, auto-computed)
        ttk.Label(file_frame, text="Output PDF:").grid(row=1, column=0, sticky="w", pady=2)
        self._output_label_var = tk.StringVar(value="(auto - chosen after input selected)")
        ttk.Label(
            file_frame, textvariable=self._output_label_var,
            foreground="grey", anchor="w",
        ).grid(row=1, column=1, columnspan=2, sticky="ew", padx=6)

        # -- Settings ----------------------------------------------------
        settings_frame = ttk.LabelFrame(self, text="Settings", padding=6)
        settings_frame.pack(fill="x", **outer_pad)
        settings_frame.columnconfigure(1, weight=1)

        row = 0

        ttk.Label(settings_frame, text="Annotation goal:").grid(
            row=row, column=0, sticky="w", pady=2)
        self._goal_var = tk.StringVar(value=DEFAULT_CONFIG.get("annotation_goal", ""))
        ttk.Entry(settings_frame, textvariable=self._goal_var).grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=6)
        row += 1

        ttk.Label(settings_frame, text="Document type:").grid(
            row=row, column=0, sticky="w", pady=2)
        self._doctype_var = tk.StringVar(value=DEFAULT_CONFIG.get("document_type", ""))
        ttk.Entry(settings_frame, textvariable=self._doctype_var).grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=6)
        row += 1

        # Page range
        ttk.Label(settings_frame, text="Page range:").grid(
            row=row, column=0, sticky="w", pady=2)
        page_range_frame = ttk.Frame(settings_frame)
        page_range_frame.grid(row=row, column=1, columnspan=2, sticky="w", padx=6)
        ttk.Label(page_range_frame, text="From").pack(side="left")
        self._page_from = tk.IntVar(value=1)
        ttk.Spinbox(
            page_range_frame, from_=1, to=9999, width=6,
            textvariable=self._page_from,
        ).pack(side="left", padx=(4, 8))
        ttk.Label(page_range_frame, text="to").pack(side="left")
        self._page_to = tk.IntVar(value=0)
        ttk.Spinbox(
            page_range_frame, from_=0, to=9999, width=6,
            textvariable=self._page_to,
        ).pack(side="left", padx=(4, 0))
        ttk.Label(page_range_frame, text="  (0 = all)", foreground="grey").pack(side="left")
        row += 1

        ttk.Label(settings_frame, text="Custom notes:").grid(
            row=row, column=0, sticky="nw", pady=2)
        self._prompt_text = tk.Text(settings_frame, height=2, wrap="word")
        self._prompt_text.grid(row=row, column=1, columnspan=2, sticky="ew", padx=6, pady=2)
        row += 1

        ttk.Label(settings_frame, text="API Key:").grid(
            row=row, column=0, sticky="w", pady=2)
        self._apikey_var = tk.StringVar(value=DEFAULT_CONFIG["api_key"])
        ttk.Entry(settings_frame, textvariable=self._apikey_var, show="*").grid(
            row=row, column=1, columnspan=2, sticky="ew", padx=6)
        row += 1

        ttk.Label(settings_frame, text="Model:").grid(
            row=row, column=0, sticky="w", pady=2)
        self._model_var = tk.StringVar(value=DEFAULT_CONFIG["model"])
        ttk.Combobox(
            settings_frame, textvariable=self._model_var, values=PRESET_MODELS,
        ).grid(row=row, column=1, columnspan=2, sticky="ew", padx=6)
        row += 1

        ttk.Label(settings_frame, text="Config file\n(optional):").grid(
            row=row, column=0, sticky="w", pady=2)
        self._config_var = tk.StringVar()
        ttk.Entry(settings_frame, textvariable=self._config_var).grid(
            row=row, column=1, sticky="ew", padx=6)
        ttk.Button(settings_frame, text="Browse...",
                   command=self._browse_config).grid(row=row, column=2)

        # -- Phase 1 button ----------------------------------------------
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=(6, 2))

        self._discover_btn = ttk.Button(
            btn_frame, text="Step 1: Discover Categories",
            command=self._start_discovery,
        )
        self._discover_btn.grid(row=0, column=0, padx=6)

        self._cancel_btn = ttk.Button(
            btn_frame, text="Cancel",
            command=self._cancel, state="disabled",
        )
        self._cancel_btn.grid(row=0, column=1, padx=6)

        # -- Category preview panel --------------------------------------
        cat_frame = ttk.LabelFrame(self, text="Discovered Categories (Phase 1 result)", padding=6)
        cat_frame.pack(fill="x", **outer_pad)
        self._cat_inner = ttk.Frame(cat_frame)
        self._cat_inner.pack(fill="x")
        self._cat_placeholder = ttk.Label(
            self._cat_inner,
            text="Run Step 1 to discover annotation categories for your document.",
            foreground="grey",
        )
        self._cat_placeholder.pack(anchor="w")

        rerun_frame = ttk.Frame(cat_frame)
        rerun_frame.pack(fill="x", pady=(2, 0))
        self._rerun_btn = ttk.Button(
            rerun_frame, text="Re-run Discovery",
            command=self._start_discovery, state="disabled",
        )
        self._rerun_btn.pack(side="left")

        # -- Phase 2 button ----------------------------------------------
        self._annotate_btn = ttk.Button(
            self, text="Step 2: Confirm & Annotate",
            command=self._start_annotation, state="disabled",
        )
        self._annotate_btn.pack(pady=(2, 2))

        # -- Progress ----------------------------------------------------
        progress_frame = ttk.LabelFrame(self, text="Progress", padding=6)
        progress_frame.pack(fill="x", **outer_pad)
        progress_frame.columnconfigure(0, weight=1)

        eta_row = ttk.Frame(progress_frame)
        eta_row.grid(row=0, column=0, sticky="ew")
        self._progress_label = ttk.Label(eta_row, text="Idle")
        self._progress_label.pack(side="left")
        self._eta_label = ttk.Label(eta_row, text="", foreground="grey")
        self._eta_label.pack(side="right")

        self._progress_bar = ttk.Progressbar(
            progress_frame, mode="determinate", maximum=100)
        self._progress_bar.grid(row=1, column=0, sticky="ew", pady=2)

        # -- Session log -------------------------------------------------
        log_frame = ttk.LabelFrame(self, text="Session Log", padding=6)
        log_frame.pack(fill="both", expand=True, **outer_pad)

        self._log_box = scrolledtext.ScrolledText(
            log_frame,
            state="disabled",
            wrap="word",
            height=8,
            font=("Courier", 9),
        )
        self._log_box.pack(fill="both", expand=True)

        for tag, color in LOG_TAG_COLORS.items():
            self._log_box.tag_config(tag, foreground=color)

        # Internal timing state for ETA
        self._annotation_start = None
        self._last_page_num = 0
        self._total_pages = 0

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title="Select input PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self._input_var.set(path)

    def _browse_config(self):
        path = filedialog.askopenfilename(
            title="Select config file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self._config_var.set(path)

    # ------------------------------------------------------------------
    # Input-change hook
    # ------------------------------------------------------------------

    def _on_input_changed(self, *_):
        input_pdf = self._input_var.get().strip()
        if not input_pdf or not Path(input_pdf).exists():
            self._output_label_var.set("(auto - chosen after input selected)")
            self._output_path = None
            self._pdf_page_count = 0
            return

        # Compute output path
        out = compute_output_path(input_pdf)
        self._output_path = str(out)
        self._output_label_var.set(self._output_path)

        # Cache page count and update page_to spinbox in a single open
        try:
            doc = fitz.open(input_pdf)
            self._pdf_page_count = len(doc)
            doc.close()
            self._page_to.set(self._pdf_page_count)
        except Exception:
            self._pdf_page_count = 0

    # ------------------------------------------------------------------
    # Build config from UI
    # ------------------------------------------------------------------

    def _build_config(self):
        config = load_config(self._config_var.get().strip() or None)

        api_key = self._apikey_var.get().strip()
        if api_key:
            config["api_key"] = api_key
        model = self._model_var.get().strip()
        if model:
            config["model"] = model
        goal = self._goal_var.get().strip()
        if goal:
            config["annotation_goal"] = goal
        doc_type = self._doctype_var.get().strip()
        if doc_type:
            config["document_type"] = doc_type
        custom_prompt = self._prompt_text.get("1.0", "end-1c").strip()
        if custom_prompt:
            config["_custom_prompt"] = custom_prompt

        # Page range — use cached page count to avoid re-opening the PDF
        p_from = self._page_from.get()
        p_to = self._page_to.get()
        total = self._pdf_page_count
        if total > 0 and p_to > 0 and (p_from > 1 or p_to < total):
            start = max(0, p_from - 1)
            end = min(p_to, total)
            config["_page_range"] = range(start, end)

        return config

    # ------------------------------------------------------------------
    # Phase 1 - Category discovery
    # ------------------------------------------------------------------

    def _start_discovery(self):
        input_pdf = self._input_var.get().strip()
        if not input_pdf:
            messagebox.showerror("Missing input", "Please select an input PDF first.")
            return
        if not Path(input_pdf).exists():
            messagebox.showerror("File not found",
                                 "Input file not found:\n{}".format(input_pdf))
            return

        config = self._build_config()
        try:
            validate_config(config)
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self._clear_log()
        self._discovered_categories = None
        self._annotate_btn.config(state="disabled")
        self._discover_btn.config(state="disabled")
        self._rerun_btn.config(state="disabled")
        self._cancel_event.clear()
        self._cancel_btn.config(state="normal")
        self._progress_label.config(text="Discovering categories...")
        self._progress_bar["value"] = 0

        self._worker_thread = threading.Thread(
            target=self._discovery_worker,
            args=(input_pdf, config),
            daemon=True,
        )
        self._worker_thread.start()

    def _discovery_worker(self, input_pdf, config):
        try:
            client = InferenceClient(api_key=config["api_key"])
            doc = fitz.open(input_pdf)
            cats = discover_categories(doc, config, client, log_cb=self._on_log)
            doc.close()
            self._msg_queue.put(("CATEGORIES", cats))
        except Exception as exc:
            self._msg_queue.put(("FATAL", str(exc)))

    # ------------------------------------------------------------------
    # Phase 2 - Annotation
    # ------------------------------------------------------------------

    def _start_annotation(self):
        if not self._discovered_categories:
            messagebox.showerror("No categories", "Please run Step 1 first.")
            return

        input_pdf = self._input_var.get().strip()
        if not input_pdf or not Path(input_pdf).exists():
            messagebox.showerror("File not found",
                                 "Input file not found:\n{}".format(input_pdf))
            return

        output_pdf = self._output_path
        if not output_pdf:
            output_pdf = str(compute_output_path(input_pdf))
            self._output_path = output_pdf

        config = self._build_config()
        try:
            validate_config(config)
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self._clear_log()
        self._progress_bar["value"] = 0
        self._progress_label.config(text="Starting annotation...")
        self._eta_label.config(text="")
        self._annotate_btn.config(state="disabled")
        self._discover_btn.config(state="disabled")
        self._rerun_btn.config(state="disabled")
        self._cancel_event.clear()
        self._cancel_btn.config(state="normal")
        self._annotation_start = time.monotonic()
        self._last_page_num = 0

        self._worker_thread = threading.Thread(
            target=self._annotation_worker,
            args=(input_pdf, output_pdf, config, self._discovered_categories),
            daemon=True,
        )
        self._worker_thread.start()

    def _annotation_worker(self, input_pdf, output_pdf, config, categories):
        try:
            results = annotate_pdf(
                input_pdf,
                output_pdf,
                config,
                categories=categories,
                progress_cb=self._on_progress,
                log_cb=self._on_log,
                cancel_event=self._cancel_event,
            )
            self._msg_queue.put(("DONE", output_pdf, results))
        except Exception as exc:
            self._msg_queue.put(("FATAL", str(exc)))

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def _cancel(self):
        self._cancel_event.set()
        self._cancel_btn.config(state="disabled")
        self._progress_label.config(text="Cancelling...")

    # ------------------------------------------------------------------
    # Callbacks from worker threads (thread-safe via queue)
    # ------------------------------------------------------------------

    def _on_progress(self, current, total):
        self._msg_queue.put(("PROGRESS", current, total))

    def _on_log(self, level, message):
        self._msg_queue.put(("LOG", level, message))

    # ------------------------------------------------------------------
    # Queue polling
    # ------------------------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                item = self._msg_queue.get_nowait()
                kind = item[0]

                if kind == "PROGRESS":
                    _, current, total = item
                    self._last_page_num = current
                    self._total_pages = total
                    pct = int((current / total) * 100) if total else 0
                    self._progress_bar["value"] = pct
                    self._progress_label.config(
                        text="Page {} of {}  ({}%)".format(current, total, pct))

                    # ETA
                    if self._annotation_start and current > 1:
                        elapsed = time.monotonic() - self._annotation_start
                        rate = elapsed / current
                        remaining = rate * (total - current)
                        self._eta_label.config(
                            text="ETA: ~{}s remaining".format(int(remaining)))

                elif kind == "LOG":
                    _, level, message = item
                    self._append_log(level, message)

                elif kind == "CATEGORIES":
                    _, cats = item
                    self._discovered_categories = cats
                    self._show_categories(cats)
                    self._annotate_btn.config(state="normal")
                    self._discover_btn.config(state="normal")
                    self._rerun_btn.config(state="normal")
                    self._cancel_btn.config(state="disabled")
                    self._progress_label.config(text="Categories ready - click Step 2 to annotate")

                elif kind == "DONE":
                    _, out_path, results = item
                    self._progress_bar["value"] = 100
                    self._progress_label.config(text="Done")
                    self._eta_label.config(text="")
                    self._append_log("SUCCESS",
                                     "Saved annotated PDF -> {}".format(out_path))
                    self._save_log(out_path, results)
                    self._discover_btn.config(state="normal")
                    self._rerun_btn.config(state="normal")
                    self._annotate_btn.config(state="normal")
                    self._cancel_btn.config(state="disabled")

                    # Completion dialog with "Open file" button
                    self._show_done_dialog(out_path)

                elif kind == "FATAL":
                    _, msg = item
                    self._append_log("ERROR", "Fatal error: {}".format(msg))
                    self._progress_label.config(text="Failed")
                    self._discover_btn.config(state="normal")
                    self._rerun_btn.config(state="normal")
                    self._annotate_btn.config(state="normal")
                    self._cancel_btn.config(state="disabled")
                    messagebox.showerror("Error",
                                         "Operation failed:\n\n{}".format(msg))

        except queue.Empty:
            pass

        self.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    # Category preview panel
    # ------------------------------------------------------------------

    def _show_categories(self, categories):
        # Clear old widgets
        for w in self._cat_inner.winfo_children():
            w.destroy()

        colors = DEFAULT_CONFIG["annotation_colors"]
        for idx, cat in enumerate(categories):
            row_frame = ttk.Frame(self._cat_inner)
            row_frame.pack(fill="x", pady=2)

            # Colour swatch
            tk_color = _rgb_to_tk(colors[idx % len(colors)])
            swatch = tk.Label(
                row_frame, text="   ", bg=tk_color,
                relief="solid", borderwidth=1,
            )
            swatch.pack(side="left", padx=(0, 8))

            label_text = "{}  -  {}".format(cat["key"], cat["label"])
            ttk.Label(row_frame, text=label_text, anchor="w").pack(side="left", fill="x")

    # ------------------------------------------------------------------
    # Completion dialog
    # ------------------------------------------------------------------

    def _show_done_dialog(self, out_path):
        dialog = tk.Toplevel(self)
        dialog.title("Annotation Complete")
        dialog.resizable(False, False)
        dialog.grab_set()

        ttk.Label(dialog, text="Annotation complete!", font=("", 12, "bold")).pack(
            pady=(16, 4), padx=20)
        ttk.Label(dialog, text="Saved to:", foreground="grey").pack()
        ttk.Label(dialog, text=out_path, wraplength=400).pack(padx=20, pady=(0, 12))

        btn_row = ttk.Frame(dialog)
        btn_row.pack(pady=(0, 16))

        def open_file():
            try:
                if sys.platform == "win32":
                    os.startfile(out_path)
                elif sys.platform == "darwin":
                    subprocess.call(["open", out_path])
                else:
                    subprocess.call(["xdg-open", out_path])
            except Exception as exc:
                messagebox.showwarning("Could not open file", str(exc))

        ttk.Button(btn_row, text="Open Annotated PDF", command=open_file).grid(
            row=0, column=0, padx=6)
        ttk.Button(btn_row, text="Close", command=dialog.destroy).grid(
            row=0, column=1, padx=6)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def _append_log(self, level, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = "{} [{}] {}\n".format(timestamp, level, message)
        self._log_lines.append(line)
        self._log_records.append({
            "time": timestamp,
            "level": level,
            "message": message,
        })

        self._log_box.config(state="normal")
        tag = level if level in LOG_TAG_COLORS else "INFO"
        self._log_box.insert("end", line, tag)
        self._log_box.see("end")
        self._log_box.config(state="disabled")

        # Persist errors and warnings to the log file immediately.
        if level in ("ERROR", "WARNING"):
            try:
                with open(LOG_FILE, "a", encoding="utf-8") as fh:
                    fh.write(datetime.now().strftime("%Y-%m-%d ") + line)
            except OSError:
                pass

    def _clear_log(self):
        self._log_lines = []
        self._log_records = []
        self._log_box.config(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.config(state="disabled")

    def _save_log(self, output_pdf, page_results=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path(output_pdf).with_name("annotation_log_{}".format(timestamp))

        # Plain-text log
        txt_path = base.with_suffix(".txt")
        try:
            txt_path.write_text("".join(self._log_lines), encoding="utf-8")
            self._append_log("INFO", "Text log saved -> {}".format(txt_path))
        except Exception as exc:
            self._append_log("WARNING", "Could not save text log: {}".format(exc))

        # Structured JSON log
        json_path = base.with_suffix(".json")
        try:
            payload = {
                "session_timestamp": timestamp,
                "output_pdf": output_pdf,
                "log_entries": self._log_records,
                "page_results": page_results or [],
            }
            json_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self._append_log("INFO", "JSON log saved  -> {}".format(json_path))
        except Exception as exc:
            self._append_log("WARNING", "Could not save JSON log: {}".format(exc))


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    app = AnnotatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
