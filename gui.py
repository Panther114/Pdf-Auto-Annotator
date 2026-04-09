"""
PDF Auto-Annotator - Graphical User Interface

Run with:
    python gui.py

Requirements: same as annotator.py (pymupdf, huggingface_hub).
tkinter is part of the Python standard library.
"""

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from datetime import datetime
from pathlib import Path

from annotator import DEFAULT_CONFIG, annotate_pdf, load_config

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

WINDOW_MIN_WIDTH  = 660
WINDOW_MIN_HEIGHT = 640


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class AnnotatorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PDF Auto-Annotator")
        self.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.resizable(True, True)

        self._msg_queue: queue.Queue = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._log_lines: list[str] = []

        self._build_ui()
        self._poll_queue()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer_pad = {"padx": 10, "pady": 4}

        # -- Files -------------------------------------------------------
        file_frame = ttk.LabelFrame(self, text="Files", padding=8)
        file_frame.pack(fill="x", **outer_pad)
        file_frame.columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="Input PDF:").grid(
            row=0, column=0, sticky="w", pady=2)
        self._input_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self._input_var).grid(
            row=0, column=1, sticky="ew", padx=6)
        ttk.Button(file_frame, text="Browse...",
                   command=self._browse_input).grid(row=0, column=2)

        ttk.Label(file_frame, text="Output PDF:").grid(
            row=1, column=0, sticky="w", pady=2)
        self._output_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self._output_var).grid(
            row=1, column=1, sticky="ew", padx=6)
        ttk.Button(file_frame, text="Browse...",
                   command=self._browse_output).grid(row=1, column=2)

        # -- Settings ----------------------------------------------------
        settings_frame = ttk.LabelFrame(self, text="Settings", padding=8)
        settings_frame.pack(fill="x", **outer_pad)
        settings_frame.columnconfigure(1, weight=1)

        ttk.Label(settings_frame, text="Document context:").grid(
            row=0, column=0, sticky="w", pady=2)
        self._context_var = tk.StringVar(value=DEFAULT_CONFIG["context"])
        ttk.Entry(settings_frame, textvariable=self._context_var).grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=6)

        ttk.Label(settings_frame, text="Custom instructions\n(optional):").grid(
            row=1, column=0, sticky="nw", pady=2)
        self._prompt_text = tk.Text(settings_frame, height=3, wrap="word")
        self._prompt_text.grid(
            row=1, column=1, columnspan=2, sticky="ew", padx=6, pady=2)
        ttk.Label(
            settings_frame,
            text="Extra guidance appended to the auto-generated prompt.",
            foreground="grey",
        ).grid(row=2, column=1, columnspan=2, sticky="w", padx=6)

        ttk.Label(settings_frame, text="API Key:").grid(
            row=3, column=0, sticky="w", pady=2)
        self._apikey_var = tk.StringVar(value=DEFAULT_CONFIG["api_key"])
        ttk.Entry(settings_frame, textvariable=self._apikey_var, show="*").grid(
            row=3, column=1, columnspan=2, sticky="ew", padx=6)

        ttk.Label(settings_frame, text="Model:").grid(
            row=4, column=0, sticky="w", pady=2)
        self._model_var = tk.StringVar(value=DEFAULT_CONFIG["model"])
        ttk.Combobox(
            settings_frame, textvariable=self._model_var, values=PRESET_MODELS,
        ).grid(row=4, column=1, columnspan=2, sticky="ew", padx=6)

        ttk.Label(settings_frame, text="Config file:").grid(
            row=5, column=0, sticky="w", pady=2)
        self._config_var = tk.StringVar()
        ttk.Entry(settings_frame, textvariable=self._config_var).grid(
            row=5, column=1, sticky="ew", padx=6)
        ttk.Button(settings_frame, text="Browse...",
                   command=self._browse_config).grid(row=5, column=2)

        # -- Run button --------------------------------------------------
        self._run_btn = ttk.Button(
            self, text="Start Annotation", command=self._start)
        self._run_btn.pack(pady=(8, 2))

        # -- Progress ----------------------------------------------------
        progress_frame = ttk.LabelFrame(self, text="Progress", padding=8)
        progress_frame.pack(fill="x", **outer_pad)
        progress_frame.columnconfigure(0, weight=1)

        self._progress_label = ttk.Label(progress_frame, text="Idle")
        self._progress_label.grid(row=0, column=0, sticky="w")

        self._progress_bar = ttk.Progressbar(
            progress_frame, mode="determinate", maximum=100)
        self._progress_bar.grid(row=1, column=0, sticky="ew", pady=4)

        # -- Session log -------------------------------------------------
        log_frame = ttk.LabelFrame(self, text="Session Log", padding=8)
        log_frame.pack(fill="both", expand=True, **outer_pad)

        self._log_box = scrolledtext.ScrolledText(
            log_frame,
            state="disabled",
            wrap="word",
            height=10,
            font=("Courier", 9),
        )
        self._log_box.pack(fill="both", expand=True)

        for tag, color in LOG_TAG_COLORS.items():
            self._log_box.tag_config(tag, foreground=color)

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Select input PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self._input_var.set(path)
            if not self._output_var.get():
                p = Path(path)
                self._output_var.set(str(p.with_stem(p.stem + "_annotated")))

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save annotated PDF as",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self._output_var.set(path)

    def _browse_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Select config file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self._config_var.set(path)

    # ------------------------------------------------------------------
    # Start annotation
    # ------------------------------------------------------------------

    def _start(self) -> None:
        input_pdf  = self._input_var.get().strip()
        output_pdf = self._output_var.get().strip()

        if not input_pdf:
            messagebox.showerror("Missing input", "Please select an input PDF.")
            return
        if not output_pdf:
            messagebox.showerror("Missing output",
                                 "Please specify an output PDF path.")
            return
        if not Path(input_pdf).exists():
            messagebox.showerror("File not found",
                                 f"Input file not found:\n{input_pdf}")
            return

        config = load_config(self._config_var.get().strip() or None)

        api_key = self._apikey_var.get().strip()
        if api_key:
            config["api_key"] = api_key
        model = self._model_var.get().strip()
        if model:
            config["model"] = model
        context = self._context_var.get().strip()
        if context:
            config["context"] = context
        custom_prompt = self._prompt_text.get("1.0", "end-1c").strip()
        if custom_prompt:
            config["_custom_prompt"] = custom_prompt

        self._clear_log()
        self._progress_bar["value"] = 0
        self._progress_label.config(text="Starting...")
        self._run_btn.config(state="disabled")

        self._worker_thread = threading.Thread(
            target=self._run_worker,
            args=(input_pdf, output_pdf, config),
            daemon=True,
        )
        self._worker_thread.start()

    def _run_worker(self, input_pdf: str, output_pdf: str, config: dict) -> None:
        """Runs in the background thread; communicates via _msg_queue."""
        try:
            annotate_pdf(
                input_pdf,
                output_pdf,
                config,
                progress_cb=self._on_progress,
                log_cb=self._on_log,
            )
            self._msg_queue.put(("DONE", output_pdf))
        except Exception as exc:
            self._msg_queue.put(("FATAL", str(exc)))

    # ------------------------------------------------------------------
    # Callbacks from the annotation thread (thread-safe via queue)
    # ------------------------------------------------------------------

    def _on_progress(self, current: int, total: int) -> None:
        self._msg_queue.put(("PROGRESS", current, total))

    def _on_log(self, level: str, message: str) -> None:
        self._msg_queue.put(("LOG", level, message))

    # ------------------------------------------------------------------
    # Queue polling (runs on main thread via after())
    # ------------------------------------------------------------------

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self._msg_queue.get_nowait()
                kind = item[0]

                if kind == "PROGRESS":
                    _, current, total = item
                    pct = int((current / total) * 100) if total else 0
                    self._progress_bar["value"] = pct
                    self._progress_label.config(
                        text=f"Page {current} of {total}  ({pct}%)")

                elif kind == "LOG":
                    _, level, message = item
                    self._append_log(level, message)

                elif kind == "DONE":
                    _, out_path = item
                    self._progress_bar["value"] = 100
                    self._progress_label.config(text="Done")
                    self._append_log("SUCCESS",
                                     f"Saved annotated PDF -> {out_path}")
                    self._save_log(out_path)
                    self._run_btn.config(state="normal")
                    messagebox.showinfo(
                        "Done",
                        f"Annotation complete!\n\nSaved to:\n{out_path}",
                    )

                elif kind == "FATAL":
                    _, msg = item
                    self._append_log("ERROR", f"Fatal error: {msg}")
                    self._progress_label.config(text="Failed")
                    self._run_btn.config(state="normal")
                    messagebox.showerror("Error",
                                         f"Annotation failed:\n\n{msg}")

        except queue.Empty:
            pass

        self.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def _append_log(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"{timestamp} [{level}] {message}\n"
        self._log_lines.append(line)

        self._log_box.config(state="normal")
        tag = level if level in LOG_TAG_COLORS else "INFO"
        self._log_box.insert("end", line, tag)
        self._log_box.see("end")
        self._log_box.config(state="disabled")

    def _clear_log(self) -> None:
        self._log_lines = []
        self._log_box.config(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.config(state="disabled")

    def _save_log(self, output_pdf: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(output_pdf).with_name(
            f"annotation_log_{timestamp}.txt")
        try:
            log_path.write_text("".join(self._log_lines), encoding="utf-8")
            self._append_log("INFO", f"Log saved -> {log_path}")
        except Exception as exc:
            self._append_log("WARNING", f"Could not save log: {exc}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    app = AnnotatorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
