# SPANDHAN — Digital Signal & Image Processing (DSP) UI Module

The **SPANDHAN DSP Analysis Workstation** is a scientific signal-processing visualization interface built with PySide6 and Matplotlib. It serves as the dynamic presentation layer for the unified MATLAB DSP pipeline.

---

## 1. System Architecture

```text
                  INPUT SIGNAL / IMAGE
                           ↓
               MATLAB PREPROCESSING
           (Normalization, DC Removal, Resampling)
                           ↓
               PYTHON ML CLASSIFICATION
      (SignalCNN / Random Forest Multiclass Prediction)
                           ↓
           MATLAB CLASS-AWARE DSP ANALYSIS
     (FFT, STFT, Wavelet, FIR, IIR, 2D FFT, Filter, etc.)
                           ↓
                  DSP RESULT STRUCTURE
                           ↓
               PYTHON/PySide6 DSP UI
```

### Architectural Contract:
- **MATLAB is the source of truth for all DSP computations.**
- **The Python UI never reimplements DSP algorithms** (no FFT, STFT, wavelets, FIR, IIR, convolution, or deconvolution calculations are performed in GUI code).
- The UI dynamically inspects `result.dsp` and renders rich interactive scientific visualizations strictly for modules that actually executed.

---

## 2. Supported Signal Classes & Class-Aware Routing

SPANDHAN operates on exactly five canonical signal classes for both 1D Audio and 2D Image modalities:

| Canonical Class | Audio Expected Analyses | Image Expected Analyses |
| :--- | :--- | :--- |
| **Impulse** | FFT, Wavelet (db4), FIR, (Convolution, Deconvolution) | 2D FFT, 2D Wavelet (db4), Image Filter |
| **Sinusoidal** | FFT, FIR, IIR | 2D FFT, Image Filter, (Convolution) |
| **White Noise** | FFT, FIR, IIR | 2D FFT, Image Filter |
| **Step** | FFT, Wavelet (db4), FIR, IIR, (Convolution) | 2D FFT, 2D Wavelet (db4), Image Filter |
| **Chirp** | FFT, STFT Spectrogram, Wavelet (db4), (Deconvolution) | 2D FFT, 2D Wavelet, Image Filter |

---

## 3. Directory Structure

All DSP UI implementation files reside strictly inside `new_update/ui/dsp/`:

```text
new_update/ui/dsp/
├── __init__.py               # Package exports
├── dsp_theme.py              # Visual design tokens, scientific color palettes, Matplotlib rcParams styling
├── dsp_result_adapter.py     # MATLAB struct, cell array, and .mat unwrap and normalization into typed dataclasses
├── dsp_plot_widget.py        # Interactive Matplotlib canvas with custom dark toolbar & cursor coordinate readout
├── dsp_cards.py              # Glassmorphic scientific cards, metadata chips, diagnostic drawer & fullscreen dialog
├── audio_dsp_view.py         # Dynamic 1D audio visualization layout (waveform, FFT, STFT, wavelets, filters)
├── image_dsp_view.py         # Dynamic 2D image visualization layout (spatial slice, 2D FFT, 2D wavelets, filtering)
├── dsp_state.py              # Singleton reactive state store (DSPStateManager)
├── dsp_controller.py         # Test case verification engine & MATLAB .mat file importer
├── dsp_page.py               # Master workstation page, pipeline indicator, technical summary & modal switcher
└── README.md                 # Module documentation and usage reference
```

---

## 4. Key Components

### `dsp_plot_widget.py` — `DSPScientificPlotWidget`
- Embeds high-DPI `FigureCanvasQTAgg` with a dark scientific theme (`#081426` canvas, `#1E293B` grid).
- Micro-toolbar controls:
  - `↺` Reset View / Autoscale (Home)
  - `✥` Pan / Drag Axes
  - `🔍` Box Zoom Mode
  - `◫` Toggle Scientific Grid
  - `💾` Export High-Res PNG / Vector PDF (300 DPI)
  - `⛶` Fullscreen Modal Viewer
- Live cursor readout: displays exact physical units (e.g. `Time: 1.250 s | Freq: 2,450 Hz` or `X: 64 px | Intensity: 0.842`).

### `dsp_result_adapter.py`
- Handles `.mat` files loaded through `scipy.io.loadmat` (handling `mat_struct`, 1D/2D matrix squeezing, nested cells).
- Exposes strongly typed dataclasses: `AudioDSPResult` and `ImageDSPResult`.

### `dsp_cards.py` — `ScientificVisualizationCard`
- Glassmorphic card styling with title, subtitle, module tag pill, and collapse/expand toggle.
- Bottom metadata strip displaying exact MATLAB-calculated metrics (peak frequency, -3dB bandwidth, spectral centroid, spectral flatness, energy, RMS, wavelet entropy, etc.).
- `DiagnosticSection` for skipped analyses (with technical reason) and failed analyses (with MATLAB error details).

### `dsp_state.py` — `DSPStateManager`
- Centralized reactive state store using Qt signals (`audio_result_changed`, `image_result_changed`, `modality_changed`).
- Supports cross-workstation session persistence.

---

## 5. Sidebar & Shell Integration

The DSP interface is directly accessible from the persistent sidebar:
- Clicking **Audio DSP** loads `AudioDSPPage` with 1D acoustic signal visualizations.
- Clicking **Image DSP** loads `ImageDSPPage` with 2D spatial frequency visualizations.
- Shared `DSPPage` architecture allows rapid switching between modalities without duplicating layout logic.
