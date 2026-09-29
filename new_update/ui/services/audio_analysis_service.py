"""
SPANDHAN — Audio Analysis & Inference Service
=============================================
Handles audio loading, metadata inspection, DSP signal characteristics computation,
and background asynchronous inference using the pre-trained Audio ML model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.io import wavfile
import soundfile as sf
from PySide6.QtCore import QObject, QThread, Signal

# Root path injection to ensure access to ml.audio
import sys
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.audio.inference.predict import AudioPredictor

_CACHED_PREDICTOR: Optional[AudioPredictor] = None


def get_audio_predictor() -> AudioPredictor:
    """Retrieve or initialize the singleton pre-trained AudioPredictor."""
    global _CACHED_PREDICTOR
    if _CACHED_PREDICTOR is None:
        _CACHED_PREDICTOR = AudioPredictor()
    return _CACHED_PREDICTOR


def load_and_inspect_audio(file_path: str | Path) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Load an audio file, inspect its technical parameters, and return:
      - normalized float64 1D signal (mono)
      - sample rate (int)
      - metadata dictionary
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    ext = path.suffix.lower()
    raw_channels = 1
    sr = 16000
    data: np.ndarray

    # Try soundfile first (supports WAV, FLAC, OGG, etc.)
    try:
        data, sr = sf.read(str(path), dtype="float64")
        if data.ndim > 1:
            raw_channels = data.shape[1]
            data = data.mean(axis=1)
        else:
            raw_channels = 1
    except Exception:
        # Fallback to scipy.io.wavfile
        sr_raw, raw_data = wavfile.read(str(path))
        sr = int(sr_raw)
        if raw_data.dtype.kind == "i":
            data = raw_data.astype(np.float64) / np.iinfo(raw_data.dtype).max
        else:
            data = raw_data.astype(np.float64)
        if data.ndim > 1:
            raw_channels = data.shape[1]
            data = data.mean(axis=1)
        else:
            raw_channels = 1

    n_samples = len(data)
    duration_s = float(n_samples / sr) if sr > 0 else 0.0
    file_size_kb = path.stat().st_size / 1024.0

    # Format string
    fmt_str = ext.replace(".", "").upper()
    if not fmt_str:
        fmt_str = "WAV"

    sr_str = f"{sr / 1000:.1f} kHz" if sr >= 1000 else f"{sr} Hz"
    channels_str = "Mono" if raw_channels == 1 else f"Stereo ({raw_channels} ch)"

    metadata: Dict[str, Any] = {
        "filename": path.name,
        "file_path": str(path.resolve()),
        "format": fmt_str,
        "sample_rate": sr,
        "sample_rate_str": sr_str,
        "duration_s": duration_s,
        "duration_str": f"{duration_s:.2f} s",
        "samples": n_samples,
        "samples_str": f"{n_samples:,}",
        "channels": raw_channels,
        "channels_str": channels_str,
        "file_size_kb": f"{file_size_kb:.1f} KB",
        "peak_amplitude": float(np.max(np.abs(data))) if n_samples > 0 else 0.0,
    }

    return data, sr, metadata


def compute_audio_characteristics(signal: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Extract the 11 fundamental signal characteristics required for SPANDHAN Audio ML:
      1. RMS
      2. Standard Deviation
      3. Peak Amplitude
      4. Energy
      5. Peak-to-RMS Ratio (Crest Factor)
      6. Zero Crossing Rate
      7. Dominant Frequency
      8. Spectral Centroid
      9. Spectral Bandwidth
      10. Spectral Flatness
      11. 3-dB Bandwidth
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    n = len(x)
    if n == 0:
        return {
            "rms": 0.0,
            "std": 0.0,
            "peak": 0.0,
            "energy": 0.0,
            "crest_factor": 0.0,
            "zcr": 0.0,
            "dom_freq": 0.0,
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "spectral_flatness": 0.0,
            "bandwidth_3db": 0.0,
        }

    # Time-domain metrics
    rms_val = float(np.sqrt(np.mean(x ** 2)))
    std_val = float(np.std(x))
    peak_val = float(np.max(np.abs(x)))
    energy_val = float(np.sum(x ** 2))
    crest_val = float(peak_val / (rms_val + 1e-12))
    zcr_val = float(np.sum(np.abs(np.diff(np.sign(x)))) / (2 * n))

    # Frequency-domain metrics (RFFT)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag = np.abs(np.fft.rfft(x))
    power = mag ** 2 + 1e-12
    sum_mag = np.sum(mag) + 1e-12

    dom_freq = float(freqs[np.argmax(mag)]) if len(mag) > 0 else 0.0
    s_cent = float(np.sum(freqs * mag) / sum_mag)
    s_bw = float(np.sqrt(np.sum(((freqs - s_cent) ** 2) * mag) / sum_mag))

    # Spectral Flatness (Wiener entropy)
    s_flat = float(np.exp(np.mean(np.log(power))) / (np.mean(power) + 1e-12))

    # 3-dB Bandwidth
    peak_pwr = np.max(power)
    thresh = peak_pwr * 0.5
    above_3db = freqs[power >= thresh]
    if len(above_3db) > 1:
        bw_3db = float(above_3db[-1] - above_3db[0])
    else:
        bw_3db = 0.0

    # 85% Spectral Rolloff
    cum_pwr = np.cumsum(power) / (np.sum(power) + 1e-12)
    rolloff_idx = int(np.searchsorted(cum_pwr, 0.85))
    rolloff_freq = float(freqs[min(rolloff_idx, len(freqs) - 1)]) if len(freqs) > 0 else 0.0

    # 4 Sub-band energy proportions
    total_p = np.sum(power) + 1e-12
    p_bass = float(np.sum(power[(freqs >= 20) & (freqs < 250)]) / total_p)
    p_lowmid = float(np.sum(power[(freqs >= 250) & (freqs < 1000)]) / total_p)
    p_mid = float(np.sum(power[(freqs >= 1000) & (freqs < 4000)]) / total_p)
    p_high = float(np.sum(power[freqs >= 4000]) / total_p)

    return {
        "rms": rms_val,
        "std": std_val,
        "peak": peak_val,
        "energy": energy_val,
        "crest_factor": crest_val,
        "zcr": zcr_val,
        "dom_freq": dom_freq,
        "spectral_centroid": s_cent,
        "spectral_bandwidth": s_bw,
        "spectral_flatness": s_flat,
        "bandwidth_3db": bw_3db,
        "spectral_rolloff": rolloff_freq,
        "bands": {
            "bass": p_bass,
            "lowmid": p_lowmid,
            "mid": p_mid,
            "high": p_high,
        },
    }


class AudioAnalysisWorker(QObject):
    """Worker object to run inference and metric computation on a secondary thread."""

    stage_changed = Signal(str)
    analysis_finished = Signal(dict)
    analysis_failed = Signal(str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            # Stage 1: Load signal
            self.stage_changed.emit("Loading signal...")
            QThread.msleep(120)

            data, sr, metadata = load_and_inspect_audio(self.file_path)

            # Stage 2: Preprocessing
            self.stage_changed.emit("Preprocessing...")
            QThread.msleep(140)

            # Stage 3: Running ML inference
            self.stage_changed.emit("Running ML inference...")
            predictor = get_audio_predictor()
            pred_result = predictor.predict_signal(data, sr)
            QThread.msleep(150)

            # Stage 4: Extracting classification results
            self.stage_changed.emit("Extracting classification results...")
            characteristics = compute_audio_characteristics(data, sr)
            QThread.msleep(100)

            # Package combined output
            result = {
                "metadata": metadata,
                "characteristics": characteristics,
                "prediction": pred_result,
                "signal": data,
                "sample_rate": sr,
            }

            self.analysis_finished.emit(result)

        except Exception as e:
            self.analysis_failed.emit(str(e))
