"""
SPANDHAN — Digital Signal & Image Processing (DSP) UI Package
=============================================================
Provides the scientific visualization interface for MATLAB-generated DSP results:
- Audio DSP (Waveform, FFT spectrum, STFT spectrogram, Wavelets, FIR, IIR, Convolution, Deconvolution)
- Image DSP (Spatial image, 2D FFT, 2D Wavelets, Image Filter, 2D Convolution, 2D Deconvolution)
- Reactive state management and MATLAB .mat / struct adapter
"""

from ui.dsp.dsp_page import DSPPage, AudioDSPPage, ImageDSPPage
from ui.dsp.dsp_state import DSPStateManager, get_dsp_state
from ui.dsp.dsp_controller import DSPController
from ui.dsp.dsp_result_adapter import (
    AudioDSPResult,
    ImageDSPResult,
    parse_audio_dsp_result,
    parse_image_dsp_result,
    detect_modality_and_parse,
    canonical_class_name,
)
from ui.dsp.dsp_plot_widget import DSPScientificPlotWidget
from ui.dsp.dsp_cards import ScientificVisualizationCard, DiagnosticSection, FullscreenPlotDialog
from ui.dsp.dsp_theme import DSP_THEME, apply_scientific_plot_style

__all__ = [
    "DSPPage",
    "AudioDSPPage",
    "ImageDSPPage",
    "DSPStateManager",
    "get_dsp_state",
    "DSPController",
    "AudioDSPResult",
    "ImageDSPResult",
    "parse_audio_dsp_result",
    "parse_image_dsp_result",
    "detect_modality_and_parse",
    "canonical_class_name",
    "DSPScientificPlotWidget",
    "ScientificVisualizationCard",
    "DiagnosticSection",
    "FullscreenPlotDialog",
    "DSP_THEME",
    "apply_scientific_plot_style",
]
