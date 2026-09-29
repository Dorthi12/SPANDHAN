"""
SPANDHAN — DSP Result Data Adapter & Normalizer
================================================
Normalizes and parses MATLAB DSP result structures into strongly-typed,
Python-native scientific dataclasses for the PySide6 UI.

Supports:
- Direct Python dictionaries (e.g. from Python-MATLAB bridges or JSON)
- MATLAB .mat files loaded via scipy.io.loadmat (handles mat_struct, cell arrays, squeezing)
- 5 Canonical Signal Classes: Impulse, Sinusoidal, White Noise, Step, Chirp
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# ---------------------------------------------------------------------------
# Canonical Class Normalization
# ---------------------------------------------------------------------------

CANONICAL_CLASSES = {
    "impulse": "Impulse",
    "sinusoidal": "Sinusoidal",
    "white_noise": "White Noise",
    "whitenoise": "White Noise",
    "white noise": "White Noise",
    "step": "Step",
    "chirp": "Chirp",
}


def canonical_class_name(raw_name: Any) -> str:
    """Safely normalizes any string representation to one of the 5 canonical names."""
    if raw_name is None:
        return "Unknown"
    s = str(raw_name).strip().lower().replace("-", "_").replace(" ", "_")
    return CANONICAL_CLASSES.get(s, str(raw_name).title())


# ---------------------------------------------------------------------------
# Low-level MATLAB struct / NumPy unwrappers
# ---------------------------------------------------------------------------

def safe_str(val: Any, default: str = "") -> str:
    """Unwraps MATLAB char/string/bytes or scalar to Python str."""
    if val is None:
        return default
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return default
        if val.dtype.kind in ("U", "S", "O"):
            item = val.flat[0]
            if isinstance(item, (bytes, bytearray)):
                return item.decode("utf-8", errors="replace")
            return str(item).strip()
        val = val.squeeze()
    return str(val).strip()


def safe_float(val: Any, default: float = 0.0) -> float:
    """Unwraps scalar numerical value safely."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return default
        try:
            return float(val.flat[0])
        except (ValueError, TypeError):
            return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    """Unwraps scalar integer safely."""
    return int(round(safe_float(val, float(default))))


def safe_bool(val: Any, default: bool = False) -> bool:
    """Unwraps scalar boolean safely."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return default
        return bool(val.flat[0])
    try:
        return bool(int(val))
    except (ValueError, TypeError):
        return default


def safe_1d_array(val: Any) -> np.ndarray:
    """Extracts a clean 1D float64 numpy array."""
    if val is None:
        return np.array([], dtype=np.float64)
    if not isinstance(val, np.ndarray):
        try:
            val = np.asarray(val, dtype=np.float64)
        except Exception:
            return np.array([], dtype=np.float64)
    val = np.squeeze(val)
    if val.ndim == 0:
        return np.array([float(val)], dtype=np.float64) if val.size > 0 else np.array([], dtype=np.float64)
    if val.ndim > 1:
        val = val.ravel()
    return val.astype(np.float64, copy=False)


def safe_2d_array(val: Any) -> np.ndarray:
    """Extracts a clean 2D float64 numpy array."""
    if val is None:
        return np.zeros((0, 0), dtype=np.float64)
    if not isinstance(val, np.ndarray):
        try:
            val = np.asarray(val, dtype=np.float64)
        except Exception:
            return np.zeros((0, 0), dtype=np.float64)
    val = np.squeeze(val)
    if val.ndim == 1:
        val = val[:, np.newaxis]
    elif val.ndim > 2:
        val = val[:, :, 0]
    return val.astype(np.float64, copy=False)


def safe_list_of_1d_arrays(val: Any) -> list[np.ndarray]:
    """Unwraps MATLAB cell array of 1D arrays (e.g. wavelet details)."""
    if val is None:
        return []
    result = []
    if isinstance(val, (list, tuple)):
        for item in val:
            result.append(safe_1d_array(item))
    elif isinstance(val, np.ndarray):
        if val.dtype == object:
            for item in val.flat:
                result.append(safe_1d_array(item))
        else:
            result.append(safe_1d_array(val))
    return result


def safe_list_of_2d_arrays(val: Any) -> list[np.ndarray]:
    """Unwraps MATLAB cell array of 2D arrays (e.g. 2D wavelet details)."""
    if val is None:
        return []
    result = []
    if isinstance(val, (list, tuple)):
        for item in val:
            result.append(safe_2d_array(item))
    elif isinstance(val, np.ndarray):
        if val.dtype == object:
            for item in val.flat:
                result.append(safe_2d_array(item))
        else:
            result.append(safe_2d_array(val))
    return result


def get_field(obj: Any, field_name: str, default: Any = None) -> Any:
    """Retrieves a field whether obj is a dict, mat_struct, or object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        if field_name in obj:
            return obj[field_name]
        # Check lower/upper case variations
        for k, v in obj.items():
            if k.lower() == field_name.lower():
                return v
        return default
    if hasattr(obj, field_name):
        return getattr(obj, field_name)
    if hasattr(obj, "_fieldnames") and field_name in obj._fieldnames:
        return getattr(obj, field_name)
    return default


def is_field_present_and_nonempty(obj: Any, field_name: str) -> bool:
    """Checks if a field exists and has meaningful data."""
    val = get_field(obj, field_name)
    if val is None:
        return False
    if isinstance(val, np.ndarray) and val.size == 0:
        return False
    if isinstance(val, (list, tuple, dict)) and len(val) == 0:
        return False
    return True


# ---------------------------------------------------------------------------
# Audio DSP Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AudioFFTData:
    frequency: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    magnitude: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    magnitude_db: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    phase: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    power: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    nfft: int = 4096
    sampling_frequency: float = 16000.0
    dc_magnitude: float = 0.0
    peak_frequency: float = 0.0
    peak_magnitude: float = 0.0
    spectral_centroid: float = 0.0
    spectral_bandwidth: float = 0.0
    bandwidth_3db: float = 0.0
    lower_3db_freq: float = 0.0
    upper_3db_freq: float = 0.0
    spectral_flatness: float = 0.0
    energy: float = 0.0
    rms: float = 0.0


@dataclass
class AudioSTFTData:
    frequency: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    time: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    magnitude_db: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    dominant_frequency: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    dominant_magnitude: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    spectral_centroid: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    spectral_bandwidth: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    frame_energy: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    total_energy: float = 0.0
    time_resolution: float = 0.0
    frequency_resolution: float = 0.0
    window_length: int = 1024
    overlap: int = 512
    hop_size: int = 512
    nfft: int = 2048
    sampling_frequency: float = 16000.0


@dataclass
class AudioWaveletData:
    wavelet_name: str = "db4"
    decomposition_level: int = 5
    approximation: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    details: list[np.ndarray] = field(default_factory=list)
    reconstructed_approximation: Optional[np.ndarray] = None
    reconstructed_details: list[np.ndarray] = field(default_factory=list)
    detail_energy: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    detail_energy_ratio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    dominant_level: int = 1
    wavelet_entropy: float = 0.0
    reconstruction_rmse: float = 0.0
    approximation_band: Tuple[float, float] = (0.0, 0.0)
    detail_bands: list[Tuple[float, float]] = field(default_factory=list)


@dataclass
class AudioFIRData:
    order: int = 100
    filter_type: str = "low"
    cutoff_frequency: float = 3000.0
    window: str = "hamming"
    frequency: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    magnitude_db: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    phase: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    group_delay: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    impulse_response: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    impulse_time: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    estimated_3db_cutoff: float = 3000.0
    sampling_frequency: float = 16000.0


@dataclass
class AudioIIRData:
    order: int = 6
    filter_family: str = "butter"
    filter_type: str = "low"
    passband_frequency: float = 3000.0
    stopband_frequency: float = 4000.0
    passband_ripple: float = 1.0
    stopband_attenuation: float = 60.0
    frequency: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    magnitude_db: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    phase: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    group_delay: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    impulse_response: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    impulse_time: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    stable: bool = True
    peak_frequency: float = 0.0
    estimated_3db_cutoff: float = 3000.0
    sampling_frequency: float = 16000.0


@dataclass
class AudioConvolutionData:
    input_signal: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    impulse_response: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    output_signal: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    method: str = "direct"
    output_mode: str = "full"
    input_energy: float = 0.0
    impulse_response_energy: float = 0.0
    output_energy: float = 0.0


@dataclass
class AudioDeconvolutionData:
    input_signal: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    reconstructed_signal: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    method: str = "regularized"
    lambda_param: float = 0.01


@dataclass
class AudioDSPResult:
    signal_class: str = "Unknown"
    status: str = "Completed"
    sampling_frequency: float = 16000.0
    signal_length: int = 32000
    duration: float = 2.0
    completed_analyses: list[str] = field(default_factory=list)
    failed_analyses: list[str] = field(default_factory=list)
    execution_log: list[str] = field(default_factory=list)
    input_file: str = ""
    raw_signal: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    analysis_signal: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)

    # Sub-modules
    fft: Optional[AudioFFTData] = None
    stft: Optional[AudioSTFTData] = None
    wavelet: Optional[AudioWaveletData] = None
    fir: Optional[AudioFIRData] = None
    iir: Optional[AudioIIRData] = None
    convolution: Optional[AudioConvolutionData] = None
    deconvolution: Optional[AudioDeconvolutionData] = None


# ---------------------------------------------------------------------------
# Image DSP Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ImageFFT2Data:
    gray_image: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    magnitude_db: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    power_spectrum: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    frequency_x: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    frequency_y: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    frequency_radius: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    spectral_centroid_x: float = 0.0
    spectral_centroid_y: float = 0.0
    spectral_bandwidth: float = 0.0
    total_energy: float = 0.0
    dominant_frequency_x: float = 0.0
    dominant_frequency_y: float = 0.0
    dominant_spatial_frequency: float = 0.0


@dataclass
class ImageWavelet2DData:
    wavelet_name: str = "db4"
    decomposition_level: int = 3
    approximation: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    horizontal_details: list[np.ndarray] = field(default_factory=list)
    vertical_details: list[np.ndarray] = field(default_factory=list)
    diagonal_details: list[np.ndarray] = field(default_factory=list)
    detail_energy: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    energy_ratio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    wavelet_entropy: float = 0.0
    reconstructed_signal: Optional[np.ndarray] = None
    reconstruction_rmse: float = 0.0


@dataclass
class ImageFilterData:
    input_image: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    filtered_image: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    difference_image: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    filter_type: str = "gaussian"
    kernel: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    kernel_size: Tuple[int, int] = (5, 5)
    sigma: float = 1.0
    boundary: str = "symmetric"
    operation: str = "convolution"
    input_mean: float = 0.0
    output_mean: float = 0.0
    input_std: float = 0.0
    output_std: float = 0.0
    input_energy: float = 0.0
    output_energy: float = 0.0
    difference_energy: float = 0.0
    computation_time: float = 0.0


@dataclass
class ImageConvolutionData:
    input_image: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    kernel: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    output_image: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    mode: str = "same"


@dataclass
class ImageDeconvolutionData:
    input_image: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    psf: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    restored_image: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    method: str = "wiener"
    nsr: float = 0.01


@dataclass
class ImageDSPResult:
    signal_class: str = "Unknown"
    status: str = "Completed"
    image_size: Tuple[int, int] = (128, 128)
    completed_analyses: list[str] = field(default_factory=list)
    failed_analyses: list[str] = field(default_factory=list)
    execution_log: list[str] = field(default_factory=list)
    input_file: str = ""
    raw_image: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    analysis_image: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)

    # Sub-modules
    fft2: Optional[ImageFFT2Data] = None
    wavelet2d: Optional[ImageWavelet2DData] = None
    filter: Optional[ImageFilterData] = None
    convolution: Optional[ImageConvolutionData] = None
    deconvolution: Optional[ImageDeconvolutionData] = None


# ---------------------------------------------------------------------------
# High-Level Parsers
# ---------------------------------------------------------------------------

def parse_audio_dsp_result(raw_result: Any) -> AudioDSPResult:
    """
    Parses a MATLAB audio pipeline result struct into an AudioDSPResult.
    """
    dsp = get_field(raw_result, "dsp", raw_result)
    ml = get_field(raw_result, "ml", {})
    inp = get_field(raw_result, "input", {})
    prep = get_field(raw_result, "preprocessing", {})

    # Signal Class & Metadata
    raw_class = get_field(dsp, "signalClass") or get_field(ml, "class", "chirp")
    sig_class = canonical_class_name(raw_class)

    status = safe_str(get_field(dsp, "status", "Completed"), "Completed")
    fs = safe_float(get_field(dsp, "samplingFrequency") or get_field(prep, "processedFs", 16000.0), 16000.0)
    sig_len = safe_int(get_field(dsp, "signalLength") or get_field(prep, "processedSamples", 32000), 32000)
    duration = safe_float(get_field(dsp, "duration", sig_len / fs), sig_len / fs)

    # Completed & Failed Analyses
    completed_raw = get_field(dsp, "completedAnalyses", [])
    if isinstance(completed_raw, (np.ndarray, list)):
        completed_analyses = [safe_str(x) for x in list(completed_raw) if safe_str(x)]
    else:
        completed_analyses = [safe_str(completed_raw)] if safe_str(completed_raw) else []

    failed_raw = get_field(dsp, "failedAnalyses", [])
    if isinstance(failed_raw, (np.ndarray, list)):
        failed_analyses = [safe_str(x) for x in list(failed_raw) if safe_str(x)]
    else:
        failed_analyses = [safe_str(failed_raw)] if safe_str(failed_raw) else []

    exec_log_raw = get_field(dsp, "executionLog", [])
    if isinstance(exec_log_raw, (np.ndarray, list)):
        execution_log = [safe_str(x) for x in list(exec_log_raw) if safe_str(x)]
    else:
        execution_log = [safe_str(exec_log_raw)] if safe_str(exec_log_raw) else []

    # Signals
    raw_sig = safe_1d_array(get_field(dsp, "input") or get_field(raw_result, "signal"))
    analysis_sig = safe_1d_array(get_field(dsp, "analysisSignal") or raw_sig)

    # ML Metadata
    confidence = safe_float(get_field(ml, "confidence", 0.0), 0.0)
    probs_raw = get_field(ml, "probabilities", {})
    probabilities = {}
    if isinstance(probs_raw, dict):
        for k, v in probs_raw.items():
            probabilities[str(k).lower()] = safe_float(v)

    # 1. FFT
    fft_data = None
    if is_field_present_and_nonempty(dsp, "fft"):
        f_obj = get_field(dsp, "fft")
        fft_data = AudioFFTData(
            frequency=safe_1d_array(get_field(f_obj, "frequency")),
            magnitude=safe_1d_array(get_field(f_obj, "magnitude")),
            magnitude_db=safe_1d_array(get_field(f_obj, "magnitudeDB")),
            phase=safe_1d_array(get_field(f_obj, "phase")),
            power=safe_1d_array(get_field(f_obj, "power")),
            nfft=safe_int(get_field(f_obj, "nfft", 4096), 4096),
            sampling_frequency=safe_float(get_field(f_obj, "samplingFrequency", fs), fs),
            dc_magnitude=safe_float(get_field(f_obj, "dcMagnitude", 0.0)),
            peak_frequency=safe_float(get_field(f_obj, "peakFrequency", 0.0)),
            peak_magnitude=safe_float(get_field(f_obj, "peakMagnitude", 0.0)),
            spectral_centroid=safe_float(get_field(f_obj, "spectralCentroid", 0.0)),
            spectral_bandwidth=safe_float(get_field(f_obj, "spectralBandwidth", 0.0)),
            bandwidth_3db=safe_float(get_field(f_obj, "bandwidth3dB", 0.0)),
            lower_3db_freq=safe_float(get_field(f_obj, "lower3dBFrequency", 0.0)),
            upper_3db_freq=safe_float(get_field(f_obj, "upper3dBFrequency", 0.0)),
            spectral_flatness=safe_float(get_field(f_obj, "spectralFlatness", 0.0)),
            energy=safe_float(get_field(f_obj, "energy", 0.0)),
            rms=safe_float(get_field(f_obj, "rms", 0.0)),
        )
        if "FFT" not in completed_analyses:
            completed_analyses.append("FFT")

    # 2. STFT
    stft_data = None
    if is_field_present_and_nonempty(dsp, "stft"):
        s_obj = get_field(dsp, "stft")
        stft_data = AudioSTFTData(
            frequency=safe_1d_array(get_field(s_obj, "frequency")),
            time=safe_1d_array(get_field(s_obj, "time")),
            magnitude_db=safe_2d_array(get_field(s_obj, "magnitudeDB")),
            dominant_frequency=safe_1d_array(get_field(s_obj, "dominantFrequency")),
            dominant_magnitude=safe_1d_array(get_field(s_obj, "dominantMagnitude")),
            spectral_centroid=safe_1d_array(get_field(s_obj, "spectralCentroid")),
            spectral_bandwidth=safe_1d_array(get_field(s_obj, "spectralBandwidth")),
            frame_energy=safe_1d_array(get_field(s_obj, "frameEnergy")),
            total_energy=safe_float(get_field(s_obj, "totalEnergy", 0.0)),
            time_resolution=safe_float(get_field(s_obj, "timeResolution", 0.0)),
            frequency_resolution=safe_float(get_field(s_obj, "frequencyResolution", 0.0)),
            window_length=safe_int(get_field(s_obj, "windowLength", 1024), 1024),
            overlap=safe_int(get_field(s_obj, "overlap", 512), 512),
            hop_size=safe_int(get_field(s_obj, "hopSize", 512), 512),
            nfft=safe_int(get_field(s_obj, "nfft", 2048), 2048),
            sampling_frequency=safe_float(get_field(s_obj, "samplingFrequency", fs), fs),
        )
        if "STFT" not in completed_analyses:
            completed_analyses.append("STFT")

    # 3. Wavelet
    wavelet_data = None
    if is_field_present_and_nonempty(dsp, "wavelet"):
        w_obj = get_field(dsp, "wavelet")
        wavelet_data = AudioWaveletData(
            wavelet_name=safe_str(get_field(w_obj, "waveletName", "db4"), "db4"),
            decomposition_level=safe_int(get_field(w_obj, "decompositionLevel", 5), 5),
            approximation=safe_1d_array(get_field(w_obj, "approximation")),
            details=safe_list_of_1d_arrays(get_field(w_obj, "details")),
            reconstructed_approximation=safe_1d_array(get_field(w_obj, "reconstructedApproximation")),
            reconstructed_details=safe_list_of_1d_arrays(get_field(w_obj, "reconstructedDetails")),
            detail_energy=safe_1d_array(get_field(w_obj, "detailEnergy")),
            detail_energy_ratio=safe_1d_array(get_field(w_obj, "detailEnergyRatio")),
            dominant_level=safe_int(get_field(w_obj, "dominantLevel", 1), 1),
            wavelet_entropy=safe_float(get_field(w_obj, "waveletEntropy", 0.0)),
            reconstruction_rmse=safe_float(get_field(w_obj, "reconstructionRMSE", 0.0)),
        )
        if "Wavelet" not in completed_analyses:
            completed_analyses.append("Wavelet")

    # 4. FIR
    fir_data = None
    if is_field_present_and_nonempty(dsp, "fir"):
        fir_obj = get_field(dsp, "fir")
        fir_data = AudioFIRData(
            order=safe_int(get_field(fir_obj, "order", 100), 100),
            filter_type=safe_str(get_field(fir_obj, "filterType", "low"), "low"),
            cutoff_frequency=safe_float(get_field(fir_obj, "cutoffFrequency", 3000.0), 3000.0),
            window=safe_str(get_field(fir_obj, "window", "hamming"), "hamming"),
            frequency=safe_1d_array(get_field(fir_obj, "frequency")),
            magnitude_db=safe_1d_array(get_field(fir_obj, "magnitudeDB")),
            phase=safe_1d_array(get_field(fir_obj, "phase")),
            group_delay=safe_1d_array(get_field(fir_obj, "groupDelay")),
            impulse_response=safe_1d_array(get_field(fir_obj, "impulseResponse")),
            impulse_time=safe_1d_array(get_field(fir_obj, "impulseTime")),
            estimated_3db_cutoff=safe_float(get_field(fir_obj, "estimated3dBCutoff", 3000.0), 3000.0),
            sampling_frequency=safe_float(get_field(fir_obj, "samplingFrequency", fs), fs),
        )
        if "FIR" not in completed_analyses:
            completed_analyses.append("FIR")

    # 5. IIR
    iir_data = None
    if is_field_present_and_nonempty(dsp, "iir"):
        iir_obj = get_field(dsp, "iir")
        iir_data = AudioIIRData(
            order=safe_int(get_field(iir_obj, "order", 6), 6),
            filter_family=safe_str(get_field(iir_obj, "filterFamily", "butter"), "butter"),
            filter_type=safe_str(get_field(iir_obj, "filterType", "low"), "low"),
            passband_frequency=safe_float(get_field(iir_obj, "passbandFrequency", 3000.0), 3000.0),
            stopband_frequency=safe_float(get_field(iir_obj, "stopbandFrequency", 4000.0), 4000.0),
            frequency=safe_1d_array(get_field(iir_obj, "frequency")),
            magnitude_db=safe_1d_array(get_field(iir_obj, "magnitudeDB")),
            phase=safe_1d_array(get_field(iir_obj, "phase")),
            group_delay=safe_1d_array(get_field(iir_obj, "groupDelay")),
            impulse_response=safe_1d_array(get_field(iir_obj, "impulseResponse")),
            impulse_time=safe_1d_array(get_field(iir_obj, "impulseTime")),
            stable=safe_bool(get_field(iir_obj, "stable", True), True),
            peak_frequency=safe_float(get_field(iir_obj, "peakFrequency", 0.0)),
            estimated_3db_cutoff=safe_float(get_field(iir_obj, "estimated3dBCutoff", 3000.0), 3000.0),
            sampling_frequency=safe_float(get_field(iir_obj, "samplingFrequency", fs), fs),
        )
        if "IIR" not in completed_analyses:
            completed_analyses.append("IIR")

    # 6. Convolution
    conv_data = None
    if is_field_present_and_nonempty(dsp, "convolution"):
        c_obj = get_field(dsp, "convolution")
        conv_data = AudioConvolutionData(
            input_signal=safe_1d_array(get_field(c_obj, "input")),
            impulse_response=safe_1d_array(get_field(c_obj, "impulseResponse")),
            output_signal=safe_1d_array(get_field(c_obj, "output")),
            method=safe_str(get_field(c_obj, "method", "direct"), "direct"),
            output_mode=safe_str(get_field(c_obj, "outputMode", "full"), "full"),
            input_energy=safe_float(get_field(c_obj, "inputEnergy", 0.0)),
            output_energy=safe_float(get_field(c_obj, "outputEnergy", 0.0)),
        )
        if "Convolution" not in completed_analyses:
            completed_analyses.append("Convolution")

    # 7. Deconvolution
    deconv_data = None
    if is_field_present_and_nonempty(dsp, "deconvolution"):
        d_obj = get_field(dsp, "deconvolution")
        deconv_data = AudioDeconvolutionData(
            input_signal=safe_1d_array(get_field(d_obj, "input")),
            reconstructed_signal=safe_1d_array(get_field(d_obj, "reconstructed")),
            method=safe_str(get_field(d_obj, "method", "regularized"), "regularized"),
            lambda_param=safe_float(get_field(d_obj, "lambda", 0.01), 0.01),
        )
        if "Deconvolution" not in completed_analyses:
            completed_analyses.append("Deconvolution")

    return AudioDSPResult(
        signal_class=sig_class,
        status=status,
        sampling_frequency=fs,
        signal_length=sig_len,
        duration=duration,
        completed_analyses=completed_analyses,
        failed_analyses=failed_analyses,
        execution_log=execution_log,
        input_file=safe_str(get_field(inp, "file", "")),
        raw_signal=raw_sig,
        analysis_signal=analysis_sig,
        confidence=confidence,
        probabilities=probabilities,
        fft=fft_data,
        stft=stft_data,
        wavelet=wavelet_data,
        fir=fir_data,
        iir=iir_data,
        convolution=conv_data,
        deconvolution=deconv_data,
    )


def parse_image_dsp_result(raw_result: Any) -> ImageDSPResult:
    """
    Parses a MATLAB image pipeline result struct into an ImageDSPResult.
    """
    dsp = get_field(raw_result, "dsp", raw_result)
    ml = get_field(raw_result, "ml", {})
    inp = get_field(raw_result, "input", {})
    prep = get_field(raw_result, "preprocessing", {})

    # Signal Class & Metadata
    raw_class = get_field(dsp, "signalClass") or get_field(ml, "class", "sinusoidal")
    sig_class = canonical_class_name(raw_class)

    status = safe_str(get_field(dsp, "status", "Completed"), "Completed")
    size_raw = get_field(dsp, "imageSize") or get_field(prep, "processedSize", [128, 128])
    if isinstance(size_raw, (np.ndarray, list, tuple)) and len(size_raw) >= 2:
        img_size = (safe_int(size_raw[0], 128), safe_int(size_raw[1], 128))
    else:
        img_size = (128, 128)

    # Completed & Failed Analyses
    completed_raw = get_field(dsp, "completedAnalyses", [])
    if isinstance(completed_raw, (np.ndarray, list)):
        completed_analyses = [safe_str(x) for x in list(completed_raw) if safe_str(x)]
    else:
        completed_analyses = [safe_str(completed_raw)] if safe_str(completed_raw) else []

    failed_raw = get_field(dsp, "failedAnalyses", [])
    if isinstance(failed_raw, (np.ndarray, list)):
        failed_analyses = [safe_str(x) for x in list(failed_raw) if safe_str(x)]
    else:
        failed_analyses = [safe_str(failed_raw)] if safe_str(failed_raw) else []

    exec_log_raw = get_field(dsp, "executionLog", [])
    if isinstance(exec_log_raw, (np.ndarray, list)):
        execution_log = [safe_str(x) for x in list(exec_log_raw) if safe_str(x)]
    else:
        execution_log = [safe_str(exec_log_raw)] if safe_str(exec_log_raw) else []

    # Images
    raw_img = safe_2d_array(get_field(dsp, "input") or get_field(raw_result, "image"))
    analysis_img = safe_2d_array(get_field(dsp, "analysisImage") or raw_img)

    # ML Metadata
    confidence = safe_float(get_field(ml, "confidence", 0.0), 0.0)
    probs_raw = get_field(ml, "probabilities", {})
    probabilities = {}
    if isinstance(probs_raw, dict):
        for k, v in probs_raw.items():
            probabilities[str(k).lower()] = safe_float(v)

    # 1. 2D FFT
    fft2_data = None
    if is_field_present_and_nonempty(dsp, "fft2"):
        f_obj = get_field(dsp, "fft2")
        fft2_data = ImageFFT2Data(
            gray_image=safe_2d_array(get_field(f_obj, "grayImage")),
            magnitude_db=safe_2d_array(get_field(f_obj, "magnitudeDB")),
            power_spectrum=safe_2d_array(get_field(f_obj, "powerSpectrum")),
            frequency_x=safe_1d_array(get_field(f_obj, "frequencyX")),
            frequency_y=safe_1d_array(get_field(f_obj, "frequencyY")),
            frequency_radius=safe_2d_array(get_field(f_obj, "frequencyRadius")),
            spectral_centroid_x=safe_float(get_field(f_obj, "spectralCentroidX", 0.0)),
            spectral_centroid_y=safe_float(get_field(f_obj, "spectralCentroidY", 0.0)),
            spectral_bandwidth=safe_float(get_field(f_obj, "spectralBandwidth", 0.0)),
            total_energy=safe_float(get_field(f_obj, "totalEnergy", 0.0)),
            dominant_frequency_x=safe_float(get_field(f_obj, "dominantFrequencyX", 0.0)),
            dominant_frequency_y=safe_float(get_field(f_obj, "dominantFrequencyY", 0.0)),
            dominant_spatial_frequency=safe_float(get_field(f_obj, "dominantSpatialFrequency", 0.0)),
        )
        if "2-D FFT" not in completed_analyses and "FFT2" not in completed_analyses:
            completed_analyses.append("2-D FFT")

    # 2. 2D Wavelet
    wavelet2d_data = None
    if is_field_present_and_nonempty(dsp, "wavelet2D") or is_field_present_and_nonempty(dsp, "wavelet2d"):
        w_obj = get_field(dsp, "wavelet2D") or get_field(dsp, "wavelet2d")
        wavelet2d_data = ImageWavelet2DData(
            wavelet_name=safe_str(get_field(w_obj, "waveletName", "db4"), "db4"),
            decomposition_level=safe_int(get_field(w_obj, "decompositionLevel", 3), 3),
            approximation=safe_2d_array(get_field(w_obj, "approximation")),
            horizontal_details=safe_list_of_2d_arrays(get_field(w_obj, "horizontalDetails")),
            vertical_details=safe_list_of_2d_arrays(get_field(w_obj, "verticalDetails")),
            diagonal_details=safe_list_of_2d_arrays(get_field(w_obj, "diagonalDetails")),
            detail_energy=safe_1d_array(get_field(w_obj, "detailEnergy")),
            energy_ratio=safe_1d_array(get_field(w_obj, "energyRatio")),
            wavelet_entropy=safe_float(get_field(w_obj, "waveletEntropy", 0.0)),
            reconstructed_signal=safe_2d_array(get_field(w_obj, "reconstructedSignal")),
            reconstruction_rmse=safe_float(get_field(w_obj, "reconstructionRMSE", 0.0)),
        )
        if "2-D Wavelet" not in completed_analyses:
            completed_analyses.append("2-D Wavelet")

    # 3. Image Filter
    filter_data = None
    if is_field_present_and_nonempty(dsp, "filter"):
        flt_obj = get_field(dsp, "filter")
        k_size_raw = get_field(flt_obj, "kernelSize", [5, 5])
        if isinstance(k_size_raw, (np.ndarray, list, tuple)) and len(k_size_raw) >= 2:
            k_size = (safe_int(k_size_raw[0], 5), safe_int(k_size_raw[1], 5))
        elif isinstance(k_size_raw, (int, float)):
            k_size = (safe_int(k_size_raw, 5), safe_int(k_size_raw, 5))
        else:
            k_size = (5, 5)

        filter_data = ImageFilterData(
            input_image=safe_2d_array(get_field(flt_obj, "inputDouble") or get_field(flt_obj, "input")),
            filtered_image=safe_2d_array(get_field(flt_obj, "filteredImage")),
            difference_image=safe_2d_array(get_field(flt_obj, "differenceImage")),
            filter_type=safe_str(get_field(flt_obj, "filterType", "gaussian"), "gaussian"),
            kernel=safe_2d_array(get_field(flt_obj, "kernel")),
            kernel_size=k_size,
            sigma=safe_float(get_field(flt_obj, "sigma", 1.0), 1.0),
            boundary=safe_str(get_field(flt_obj, "boundary", "symmetric"), "symmetric"),
            operation=safe_str(get_field(flt_obj, "operation", "convolution"), "convolution"),
            input_mean=safe_float(get_field(flt_obj, "inputMean", 0.0)),
            output_mean=safe_float(get_field(flt_obj, "outputMean", 0.0)),
            input_std=safe_float(get_field(flt_obj, "inputStandardDeviation", 0.0)),
            output_std=safe_float(get_field(flt_obj, "outputStandardDeviation", 0.0)),
            input_energy=safe_float(get_field(flt_obj, "inputEnergy", 0.0)),
            output_energy=safe_float(get_field(flt_obj, "outputEnergy", 0.0)),
            difference_energy=safe_float(get_field(flt_obj, "differenceEnergy", 0.0)),
            computation_time=safe_float(get_field(flt_obj, "computationTime", 0.0)),
        )
        if "Image Filter" not in completed_analyses:
            completed_analyses.append("Image Filter")

    # 4. 2D Convolution
    conv_data = None
    if is_field_present_and_nonempty(dsp, "convolution"):
        c_obj = get_field(dsp, "convolution")
        conv_data = ImageConvolutionData(
            input_image=safe_2d_array(get_field(c_obj, "input")),
            kernel=safe_2d_array(get_field(c_obj, "kernel")),
            output_image=safe_2d_array(get_field(c_obj, "output")),
            mode=safe_str(get_field(c_obj, "mode", "same"), "same"),
        )
        if "2-D Convolution" not in completed_analyses:
            completed_analyses.append("2-D Convolution")

    # 5. 2D Deconvolution
    deconv_data = None
    if is_field_present_and_nonempty(dsp, "deconvolution"):
        d_obj = get_field(dsp, "deconvolution")
        deconv_data = ImageDeconvolutionData(
            input_image=safe_2d_array(get_field(d_obj, "input")),
            psf=safe_2d_array(get_field(d_obj, "psf")),
            restored_image=safe_2d_array(get_field(d_obj, "reconstructed")),
            method=safe_str(get_field(d_obj, "method", "wiener"), "wiener"),
            nsr=safe_float(get_field(d_obj, "nsr", 0.01), 0.01),
        )
        if "Image Deconvolution" not in completed_analyses:
            completed_analyses.append("Image Deconvolution")

    return ImageDSPResult(
        signal_class=sig_class,
        status=status,
        image_size=img_size,
        completed_analyses=completed_analyses,
        failed_analyses=failed_analyses,
        execution_log=execution_log,
        input_file=safe_str(get_field(inp, "file", "")),
        raw_image=raw_img,
        analysis_image=analysis_img,
        confidence=confidence,
        probabilities=probabilities,
        fft2=fft2_data,
        wavelet2d=wavelet2d_data,
        filter=filter_data,
        convolution=conv_data,
        deconvolution=deconv_data,
    )


def detect_modality_and_parse(raw_result: Any) -> Union[AudioDSPResult, ImageDSPResult]:
    """
    Intelligently determines whether raw_result is an Audio or Image pipeline output,
    and returns the corresponding parsed result dataclass.
    """
    dsp = get_field(raw_result, "dsp", raw_result)

    # Check image indicators
    if (
        is_field_present_and_nonempty(dsp, "fft2")
        or is_field_present_and_nonempty(dsp, "wavelet2D")
        or is_field_present_and_nonempty(dsp, "filter")
        or is_field_present_and_nonempty(dsp, "analysisImage")
        or is_field_present_and_nonempty(dsp, "imageSize")
        or get_field(raw_result, "modality") == "image"
    ):
        return parse_image_dsp_result(raw_result)

    return parse_audio_dsp_result(raw_result)
