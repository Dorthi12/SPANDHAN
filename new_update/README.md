# SPANDHAN: Signal Processing & Machine Learning Framework

SPANDHAN is an integrated hybrid platform for Digital Signal Processing (DSP) and Machine Learning (ML) analysis of audio and image signals.

---

## Architecture: Unified VS Code Execution

You do **not** need to open MATLAB manually. The entire project runs from VS Code through a single master entry point:

```
                    VS CODE
                       │
                 python run.py  (or press F5)
                       │
                       ▼
            ┌─────────────────────┐
            │   PySide6 Desktop   │
            │     Workstation     │
            └──────────┬──────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
    Python ML               MATLAB Engine (Headless)
    - Audio Classification   - Audio Preprocessing (16 kHz, DC removal, etc.)
    - Image CNN              - Image Preprocessing (128x128 grayscale)
    - Feature Extraction     - Audio DSP (FFT, STFT, Wavelet, FIR/IIR)
                             - Image DSP (2D FFT, Wavelet2D, Filtering)
```

Both preprocessing and DSP run automatically inside the pipeline. You never have to manually run separate programs.

---

## How to Run from VS Code

### Option 1: Direct F5 Run (Recommended)
1. Open the project folder `C:\Users\d12ra\spandhan\new_update` in VS Code.
2. Press **`F5`** (or go to the **Run & Debug** panel on the left and click **`▶ Run SPANDHAN`**).

### Option 2: Integrated Terminal
Open PowerShell or VS Code terminal in the project directory:
```powershell
python run.py
```

To run a quick health check of all packages, models, and MATLAB without opening the GUI:
```powershell
python run.py --check-only
```

---

## Project Structure

- `run.py`: **Master application entry point** (checks dependencies, verifies models, checks MATLAB, and starts workstation).
- `.vscode/launch.json`: VS Code debug and run configurations (`▶ Run SPANDHAN`, `⚡ Launch UI Only`, `🐛 Debug SPANDHAN`, `🧪 Run Tests`).
- `ui/`: PySide6 graphical user interface with side navigation and integrated analysis views.
  - `pages/audio_ml/`: 1D Audio ML inspection, waveform viewer, and classification.
  - `pages/image_ml/`: 2D Image ML inspection and CNN classification.
  - `pages/audio_dsp/`: Audio DSP analysis with automated MATLAB pipeline execution.
  - `pages/image_dsp/`: Image DSP analysis with automated MATLAB pipeline execution.
- `ml/`: Python ML inference models and feature extraction pipelines.
  - `audio/`: SVM / scikit-learn audio classifier.
  - `image/`: PyTorch SignalCNN classifier.
- `matlab/`: MATLAB processing engine (executed headlessly via `matlab -batch`).
  - `preprocessing/`: Preprocessing functions (`preprocessAudio.m`, `preprocessImage.m`).
  - `dsp/`: Advanced DSP transforms and filters (`analyzeAudio.m`, `analyzeImage.m`).
  - `runAudioPipeline.m`: Unified end-to-end audio pipeline.
  - `runImagePipeline.m`: Unified end-to-end image pipeline.
- `models/`: Pre-trained ML classifiers and scalers.
- `datasets/`: Dataset categories (impulse, sinusoidal, white noise, step, chirp).
