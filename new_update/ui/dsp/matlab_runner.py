"""
SPANDHAN — MATLAB DSP Pipeline Runner
======================================
Launches MATLAB as a fully headless background subprocess to execute
runAudioPipeline() or runImagePipeline() and returns the result via a .mat file.

Architecture:
    Python UI                   MATLAB subprocess (headless, no GUI)
    ─────────────────────       ─────────────────────────────────────────
    MATLABDSPRunner.run()  →    matlab -nosplash -nodesktop -batch "..."
                           ←    .mat written to temp dir
    load_mat_file(out.mat)       → DSPStateManager updated
    UI rendered

Critical design notes
─────────────────────
• -nosplash and -nodesktop MUST come BEFORE -batch.
  MATLAB stops processing command-line flags once it encounters -batch.
  If those flags come after -batch, MATLAB opens partially in GUI mode
  and any licensing or error dialogs pop up as GUI windows (Error 5020).

• All paths passed to MATLAB strings use forward slashes.
  Windows backslashes inside MATLAB single-quoted strings work, but
  using forward slashes is safer and avoids any escape issues.

• The MATLAB batch command is wrapped in try/catch.
  Errors are written to stderr via fprintf(2, ...) so Python captures
  them without GUI popups, and exit(1) signals failure to the runner.

No DSP math is performed in Python. MATLAB is the only DSP engine.
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import QObject, QThread, Signal


# ---------------------------------------------------------------------------
# Locate the SPANDHAN project root and matlab/ directory
# ---------------------------------------------------------------------------

_HERE         = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]          # new_update/
_MATLAB_ROOT  = _PROJECT_ROOT / "matlab"  # new_update/matlab/


def _find_matlab_exe() -> Optional[str]:
    """
    Locate the matlab executable.
    Searches PATH first, then common Windows installation locations
    (newest release first).
    """
    # 1. On PATH
    which = shutil.which("matlab")
    if which:
        return which

    # 2. Common Windows paths — scan newest release first
    candidate_roots = [
        r"C:\Program Files\MATLAB",
        r"C:\Program Files (x86)\MATLAB",
    ]
    for root_str in candidate_roots:
        root = Path(root_str)
        if root.exists():
            for release_dir in sorted(root.iterdir(), reverse=True):
                exe = release_dir / "bin" / "matlab.exe"
                if exe.exists():
                    return str(exe)

    return None


# ---------------------------------------------------------------------------
# MATLAB addpath snippet — all paths use forward slashes
# ---------------------------------------------------------------------------

def _build_matlab_addpath(matlab_root: Path) -> str:
    """
    Returns a MATLAB addpath() call covering every relevant sub-directory.
    All Windows backslashes are converted to forward slashes so they are
    safe inside MATLAB single-quoted string literals.
    """
    subdirs = [
        matlab_root,
        matlab_root / "dsp",
        matlab_root / "dsp" / "analysis",
        matlab_root / "dsp" / "transforms",
        matlab_root / "dsp" / "filters",
        matlab_root / "dsp" / "systems",
        matlab_root / "dsp" / "image",
        matlab_root / "dsp" / "visualization",
        matlab_root / "preprocessing",
        matlab_root / "io",
        matlab_root / "ml_interface",
        matlab_root / "visualization",
        matlab_root / "config",
    ]

    # Forward-slashed, single-quoted paths
    quoted = [
        f"'{str(d).replace(chr(92), '/')}'"
        for d in subdirs
        if d.exists()
    ]
    return "addpath(" + ", ".join(quoted) + ");"


# ---------------------------------------------------------------------------
# Preprocessing Step Information (reflected from MATLAB contracts)
# ---------------------------------------------------------------------------

AUDIO_PREPROCESSING_STEPS = [
    ("1 → Mono Conversion",           "Multi-channel audio averaged to mono column vector"),
    ("2 → NaN / Inf Removal",         "Corrupted samples replaced with 0 before any processing"),
    ("3 → DC Offset Removal",         "Signal mean subtracted to eliminate recording pipeline artifact"),
    ("4 → Resample → 16 kHz",         "Polyphase anti-aliasing resampler (MATLAB built-in) applied"),
    ("5 → Peak Amplitude Normalize",  "Signal divided by peak amplitude → range [−1, +1]"),
    ("6 → Class-Aware Length Std",    "Class-specific crop / zero-pad to exactly 32 000 samples (2 s)"),
    ("7 → Final Normalization",       "Post-pad peak normalization guard pass"),
]

IMAGE_PREPROCESSING_STEPS = [
    ("1 → Grayscale Conversion",      "RGB/RGBA images converted to single-channel via rgb2gray"),
    ("2 → Resize → 128 × 128",        "Bicubic interpolation (MATLAB imresize) to canonical resolution"),
    ("3 → Intensity Normalize [0,1]", "Pixel values mapped to floating-point [0, 1] range"),
]


# ---------------------------------------------------------------------------
# Background Worker
# ---------------------------------------------------------------------------

class MATLABRunnerWorker(QObject):
    """
    Worker that executes a MATLAB pipeline command in a background thread.

    Signals:
        stage_changed(str)   – progress text for status bar
        finished(str)        – path to the output .mat file on success
        failed(str)          – error message on failure
    """

    stage_changed = Signal(str)
    finished      = Signal(str)   # mat file path
    failed        = Signal(str)   # error message

    def __init__(
        self,
        pipeline_fn: str,          # "runAudioPipeline" | "runImagePipeline"
        input_path:  str,          # path to input file
        out_mat:     str,          # path where MATLAB should save result.mat
        matlab_root: Path,
        matlab_exe:  str,
    ):
        super().__init__()
        self.pipeline_fn = pipeline_fn
        self.input_path  = input_path
        self.out_mat     = out_mat
        self.matlab_root = matlab_root
        self.matlab_exe  = matlab_exe

    def run(self):
        try:
            self.stage_changed.emit("Setting up MATLAB environment…")

            addpath_cmd = _build_matlab_addpath(self.matlab_root)

            # Use forward slashes in all paths passed to MATLAB strings
            escaped_input = str(self.input_path).replace("\\", "/")
            escaped_out   = str(self.out_mat).replace("\\", "/")

            # ----------------------------------------------------------------
            # MATLAB batch script
            # Wrapped in try/catch so MATLAB errors go to stderr (captured
            # by Python) instead of opening a GUI dialog window.
            # exit(1) on failure lets Python detect non-zero exit code.
            # ----------------------------------------------------------------
            matlab_script = (
                f"try; "
                f"{addpath_cmd} "
                f"result = {self.pipeline_fn}('{escaped_input}'); "
                f"save('{escaped_out}', 'result'); "
                f"catch ME; "
                f"fprintf(2, 'SPANDHAN_MATLAB_ERROR: %s\\n', ME.message); "
                f"exit(1); "
                f"end"
            )

            # ----------------------------------------------------------------
            # CRITICAL: -nosplash and -nodesktop MUST come BEFORE -batch.
            # MATLAB stops processing flags when it hits -batch, so anything
            # after -batch is treated as part of the batch script, NOT as
            # startup flags.  Putting them after causes MATLAB to start with
            # partial GUI support, which triggers GUI licensing dialogs
            # (MathWorks Licensing Error 5020).
            # ----------------------------------------------------------------
            cmd = [
                self.matlab_exe,
                "-nosplash",
                "-nodesktop",
                "-batch", matlab_script,
            ]

            self.stage_changed.emit(f"Running MATLAB {self.pipeline_fn}…")

            # Pass the full environment so MATLAB can find its license.
            # Set cwd to project root so relative paths resolve correctly.
            env = os.environ.copy()

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,       # 5-minute timeout
                cwd=str(_PROJECT_ROOT),
                env=env,
            )

            if proc.returncode != 0:
                # Prefer stderr; fall back to stdout; then a generic message
                raw_err = (proc.stderr or proc.stdout or "").strip()

                # Strip MATLAB's verbose startup lines from the error report
                lines = [
                    ln for ln in raw_err.splitlines()
                    if not ln.strip().startswith("< M A T L A B")
                    and "MATLAB is starting" not in ln
                ]
                clean_err = "\n".join(lines).strip() or "Unknown MATLAB error"
                self.failed.emit(
                    f"MATLAB exited with code {proc.returncode}.\n\n{clean_err}"
                )
                return

            if not Path(self.out_mat).exists():
                self.failed.emit(
                    "MATLAB finished but did not write the output file.\n"
                    f"Expected: {self.out_mat}\n\n"
                    f"MATLAB output:\n{(proc.stdout or '').strip()}"
                )
                return

            self.stage_changed.emit("MATLAB DSP complete — loading result…")
            self.finished.emit(self.out_mat)

        except subprocess.TimeoutExpired:
            self.failed.emit(
                "MATLAB process timed out (> 300 s).\n"
                "The pipeline may be stuck. Try with a shorter/simpler signal."
            )
        except FileNotFoundError:
            self.failed.emit(
                f"MATLAB executable not found:\n  '{self.matlab_exe}'\n\n"
                "Ensure MATLAB R2019b or later is installed.\n"
                "Alternatively, use 'Import MATLAB .mat Result' to load an existing file."
            )
        except Exception as exc:
            self.failed.emit(f"Unexpected error launching MATLAB:\n{exc}")


# ---------------------------------------------------------------------------
# Public Controller
# ---------------------------------------------------------------------------

class MATLABDSPRunner(QObject):
    """
    High-level controller that manages the MATLAB subprocess lifecycle,
    writes a temp .mat output, and reports progress back to the UI.

    Usage:
        runner = MATLABDSPRunner(parent=self)
        runner.stage_changed.connect(...)
        runner.finished.connect(...)
        runner.failed.connect(...)
        runner.run_audio_pipeline("/path/to/signal.wav")
    """

    stage_changed = Signal(str)
    finished      = Signal(str)   # path to .mat
    failed        = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[MATLABRunnerWorker] = None
        self._temp_dir: Optional[str] = None

        self._matlab_exe = _find_matlab_exe()

    @property
    def matlab_available(self) -> bool:
        return self._matlab_exe is not None

    @property
    def matlab_path(self) -> str:
        return self._matlab_exe or "Not found"

    def run_audio_pipeline(self, input_path: str):
        self._launch("runAudioPipeline", input_path)

    def run_image_pipeline(self, input_path: str):
        self._launch("runImagePipeline", input_path)

    def _launch(self, pipeline_fn: str, input_path: str):
        if self._thread and self._thread.isRunning():
            self.failed.emit("Another MATLAB process is already running. Please wait.")
            return

        if not self._matlab_exe:
            self.failed.emit(
                "MATLAB executable not found.\n\n"
                "Please ensure MATLAB R2019b+ is installed and either:\n"
                "  • Added to your system PATH, or\n"
                "  • Located at  C:\\Program Files\\MATLAB\\<version>\\bin\\matlab.exe\n\n"
                "Alternatively, use 'Import MATLAB .mat Result' to load an existing result."
            )
            return

        if not input_path or not Path(input_path).exists():
            self.failed.emit(
                f"Input file not found:\n  {input_path}\n\n"
                "Please select a valid audio or image file."
            )
            return

        # Create a unique temp directory for this run's .mat output
        self._temp_dir = tempfile.mkdtemp(prefix="spandhan_dsp_")
        out_mat = str(Path(self._temp_dir) / "dsp_result.mat")

        self._thread = QThread()
        self._worker = MATLABRunnerWorker(
            pipeline_fn = pipeline_fn,
            input_path  = input_path,
            out_mat     = out_mat,
            matlab_root = _MATLAB_ROOT,
            matlab_exe  = self._matlab_exe,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.stage_changed.connect(self.stage_changed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_worker_finished(self, mat_path: str):
        self.finished.emit(mat_path)

    def _on_worker_failed(self, error: str):
        self.failed.emit(error)

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.isRunning())
