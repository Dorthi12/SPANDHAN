"""
SPANDHAN — Master Application Launcher
=======================================
Single entry point for the entire SPANDHAN system.

    python run.py

This replaces the old MATLAB-first workflow (startSPANDHAN.m).
Python is now the application controller. MATLAB runs headlessly
as a background processing engine when DSP analysis is triggered.

Architecture
────────────
  VS Code  →  run.py  →  PySide6 UI
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
               Python ML          MATLAB subprocess
               (inference)        (preprocessing + DSP)

MATLAB is never opened as a GUI. It runs via `matlab -batch` when
the Audio DSP or Image DSP pages are used inside the application.
"""

from __future__ import annotations

import sys
import os
import shutil
import subprocess
from pathlib import Path

# ── UTF-8 safe stdout for Windows console ────────────────────────────────
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Project root is the directory that contains this file ─────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Minimum Python version ─────────────────────────────────────────────────
MIN_PYTHON = (3, 9)

# ── Required Python packages: (import_name, pip_name, display_label) ──────
REQUIRED_PACKAGES = [
    ("numpy",      "numpy",           "NumPy"),
    ("scipy",      "scipy",           "SciPy"),
    ("torch",      "torch",           "PyTorch"),
    ("sklearn",    "scikit-learn",    "scikit-learn"),
    ("joblib",     "joblib",          "joblib"),
    ("PIL",        "Pillow",          "Pillow"),
    ("cv2",        "opencv-python",   "OpenCV"),
    ("librosa",    "librosa",         "librosa"),
    ("matplotlib", "matplotlib",      "matplotlib"),
    ("soundfile",  "soundfile",       "soundfile"),
    ("PySide6",    "PySide6",         "PySide6"),
]

# ── Required trained model files ───────────────────────────────────────────
REQUIRED_MODELS = [
    (PROJECT_ROOT / "models" / "audio" / "audio_signal_classifier.pkl", "Audio classifier"),
    (PROJECT_ROOT / "models" / "audio" / "audio_scaler.pkl",            "Audio scaler"),
    (PROJECT_ROOT / "models" / "image" / "image_signal_classifier.pkl", "Image classifier"),
    (PROJECT_ROOT / "models" / "image" / "image_scaler.pkl",            "Image scaler"),
]


# ══════════════════════════════════════════════════════════════════════════ #
#  ANSI helpers (works on Windows 10+ with modern PowerShell / VS Code)
# ══════════════════════════════════════════════════════════════════════════ #

_USE_COLOR = sys.stdout.isatty() or os.environ.get("TERM_PROGRAM") == "vscode"

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def green(t):  return _c(t, "32")
def red(t):    return _c(t, "31")
def yellow(t): return _c(t, "33")
def cyan(t):   return _c(t, "36")
def bold(t):   return _c(t, "1")


# ══════════════════════════════════════════════════════════════════════════ #
#  Banner
# ══════════════════════════════════════════════════════════════════════════ #

def _print_banner():
    print()
    print(bold("═" * 60))
    print(bold(cyan("              SPANDHAN")))
    print(bold(cyan("     SIGNAL PROCESSING & ANALYSIS SYSTEM")))
    print(bold("═" * 60))
    print()


def _print_section(title: str):
    print(bold(f"\n{title}"))
    print("─" * 50)


def _ok(label: str, detail: str = ""):
    suffix = f"  {detail}" if detail else ""
    print(f"  {green('✓')}  {label:<28}{suffix}")


def _warn(label: str, detail: str = ""):
    suffix = f"  {detail}" if detail else ""
    print(f"  {yellow('⚠')}  {label:<28}{detail}")


def _fail(label: str, detail: str = ""):
    suffix = f"  {detail}" if detail else ""
    print(f"  {red('✗')}  {label:<28}{detail}")


# ══════════════════════════════════════════════════════════════════════════ #
#  Step 1 — Python version
# ══════════════════════════════════════════════════════════════════════════ #

def check_python_version():
    _print_section("[1/5] Python Runtime")
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"
    exe = sys.executable
    if (v.major, v.minor) < MIN_PYTHON:
        _fail("Python version", f"{ver_str}  (need ≥ {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")
        _abort(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required. Found {ver_str}.")
    _ok("Python version", ver_str)
    _ok("Executable", exe)


# ══════════════════════════════════════════════════════════════════════════ #
#  Step 2 — Python packages
# ══════════════════════════════════════════════════════════════════════════ #

def check_packages():
    _print_section("[2/5] Python Packages")
    import importlib
    missing = []
    for import_name, pip_name, label in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "")
            _ok(label, ver)
        except ImportError:
            _fail(label, "MISSING")
            missing.append(pip_name)

    if missing:
        print()
        print(red("  ✗  Missing packages detected."))
        print("     Install them with:")
        print(f"       {cyan(sys.executable)} -m pip install {' '.join(missing)}")
        print("     Or install everything:")
        print(f"       {cyan(sys.executable)} -m pip install -r requirements.txt")
        _abort("One or more required Python packages are missing.")


# ══════════════════════════════════════════════════════════════════════════ #
#  Step 3 — Trained model files
# ══════════════════════════════════════════════════════════════════════════ #

def check_models():
    _print_section("[3/5] Trained ML Models")
    missing = []
    for path, label in REQUIRED_MODELS:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            _ok(label, f"{size_kb:.1f} KB")
        else:
            _fail(label, f"NOT FOUND  ({path})")
            missing.append(str(path))

    if missing:
        print()
        print(red("  ✗  One or more model files are missing."))
        print("     Run the SPANDHAN training pipeline to regenerate them.")
        _abort("Required model files not found.")


# ══════════════════════════════════════════════════════════════════════════ #
#  Step 4 — MATLAB (optional but reported)
# ══════════════════════════════════════════════════════════════════════════ #

def check_matlab() -> str | None:
    """
    Locate the MATLAB executable.  MATLAB is OPTIONAL for startup — the app
    will open and you can use Audio ML / Image ML without MATLAB.  DSP pages
    will report MATLAB unavailable if it cannot be found.

    Returns the path string if found, None otherwise.
    """
    _print_section("[4/5] MATLAB Engine (DSP Backend)")

    # 1. Check PATH
    matlab_exe = shutil.which("matlab")

    # 2. Scan common Windows installation roots
    if not matlab_exe:
        candidate_roots = [
            Path(r"C:\Program Files\MATLAB"),
            Path(r"C:\Program Files (x86)\MATLAB"),
        ]
        for root in candidate_roots:
            if root.exists():
                releases = sorted(root.iterdir(), reverse=True)
                for release in releases:
                    exe = release / "bin" / "matlab.exe"
                    if exe.exists():
                        matlab_exe = str(exe)
                        break
            if matlab_exe:
                break

    if matlab_exe:
        _ok("MATLAB executable", matlab_exe)
        _ok("DSP backend", "Will run headlessly via  matlab -batch")
        _ok("Preprocessing", "MATLAB handles preprocessing automatically")
        _ok("DSP analysis",  "FFT / STFT / Wavelet / FIR / IIR via MATLAB")
        return matlab_exe
    else:
        _warn("MATLAB executable", "NOT FOUND on PATH")
        print()
        print(f"  {yellow('ℹ')}  Audio ML and Image ML will work normally.")
        print(f"  {yellow('ℹ')}  Audio DSP and Image DSP pages require MATLAB.")
        print(f"  {yellow('ℹ')}  Add MATLAB to PATH or install it to enable DSP features.")
        return None


# ══════════════════════════════════════════════════════════════════════════ #
#  Step 5 — MATLAB source tree
# ══════════════════════════════════════════════════════════════════════════ #

def check_matlab_source():
    _print_section("[5/5] MATLAB Source Modules")
    matlab_root = PROJECT_ROOT / "matlab"

    checks = [
        (matlab_root / "runAudioPipeline.m", "runAudioPipeline"),
        (matlab_root / "runImagePipeline.m", "runImagePipeline"),
        (matlab_root / "preprocessing",      "preprocessing/"),
        (matlab_root / "dsp",                "dsp/"),
        (matlab_root / "ml_interface",       "ml_interface/"),
        (matlab_root / "io",                 "io/"),
    ]

    for path, label in checks:
        if path.exists():
            _ok(label)
        else:
            _warn(label, "NOT FOUND — DSP may be affected")


# ══════════════════════════════════════════════════════════════════════════ #
#  Launch PySide6 UI
# ══════════════════════════════════════════════════════════════════════════ #

def launch_ui():
    """
    Start the SPANDHAN PySide6 application in-process.
    This is NOT a subprocess — the UI runs inside this Python process,
    which means VS Code's debugger, breakpoints, and output panel all work.
    """
    print()
    print(bold("═" * 60))
    print(bold(cyan("          SPANDHAN INITIALIZATION COMPLETE")))
    print(bold("═" * 60))
    print()
    print(f"  {green('▶')}  Launching SPANDHAN UI...")
    print(f"  {cyan('ℹ')}  MATLAB will start headlessly when DSP is triggered.")
    print(f"  {cyan('ℹ')}  You do NOT need to open MATLAB manually.")
    print()

    # Ensure the project root is on sys.path so all imports resolve
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # Import and run the application
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication
    from ui.app import SpandhanMainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("SPANDHAN Signal Intelligence")
    app.setOrganizationName("SPANDHAN")

    # Set a clean system font
    app.setFont(QFont("Segoe UI", 10))

    window = SpandhanMainWindow()
    window.show()

    sys.exit(app.exec())


# ══════════════════════════════════════════════════════════════════════════ #
#  Abort helper
# ══════════════════════════════════════════════════════════════════════════ #

def _abort(reason: str):
    print()
    print(bold(red("═" * 60)))
    print(bold(red("  SPANDHAN STARTUP FAILED")))
    print(bold(red("═" * 60)))
    print(f"\n  {red('Reason:')} {reason}\n")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════ #
#  Main
# ══════════════════════════════════════════════════════════════════════════ #

def main():
    _print_banner()

    check_python_version()
    check_packages()
    check_models()
    check_matlab()        # optional — warns but does not abort
    check_matlab_source() # informational

    if "--check-only" in sys.argv or "--check" in sys.argv:
        print()
        print(bold(green("  ✓  All environment checks passed successfully!")))
        print("     System is ready to run SPANDHAN.")
        print()
        sys.exit(0)

    launch_ui()


if __name__ == "__main__":
    main()
