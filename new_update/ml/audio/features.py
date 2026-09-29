"""
SPANDHAN — Audio ML: Full Feature Extractor  (ASCII-safe, audio-only)
=======================================================================
35 features per mono WAV.  Shared by training and inference.
- n_fft / hop_length clamped to signal length (handles short impulse WAVs)
- All output is ASCII-safe (no Unicode arrows or dashes)
"""

from __future__ import annotations
import warnings
import numpy as np
from scipy import signal as _sig
from scipy.stats import skew as _skew, kurtosis as _kurt

try:
    import librosa
    _LIBROSA = True
except ImportError:
    _LIBROSA = False
    warnings.warn("librosa not installed — MFCCs will be 0.", RuntimeWarning, stacklevel=1)

# ---------------------------------------------------------------------------
FEATURE_NAMES: list[str] = [
    # Time-domain (8)
    "rms_energy", "zero_crossing_rate", "mean_abs_amplitude", "std_amplitude",
    "skewness", "kurtosis", "temporal_flatness", "crest_factor",
    # Frequency-domain (7)
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    "spectral_flatness", "dominant_frequency", "spectral_entropy", "spectral_spread",
    # MFCCs (13)
    *[f"mfcc_{i}" for i in range(1, 14)],
    # Rhythm / energy shape (4)
    "onset_rate", "rms_slope", "short_term_energy_var", "freq_modulation_rate",
    # Instantaneous frequency (3)
    "if_mean", "if_std", "if_range",
]
NUM_FEATURES: int = len(FEATURE_NAMES)   # 35


def _safe(v: float, fb: float = 0.0) -> float:
    return float(v) if np.isfinite(v) else fb


def _spectral_entropy(mag: np.ndarray) -> float:
    p = mag / (mag.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _if_stats(x: np.ndarray, sr: int) -> tuple[float, float, float]:
    analytic = _sig.hilbert(x)
    phase    = np.unwrap(np.angle(analytic))
    if_hz    = np.diff(phase) / (2.0 * np.pi) * sr
    if_hz    = if_hz[np.isfinite(if_hz)]
    if len(if_hz) == 0:
        return 0.0, 0.0, 0.0
    return _safe(float(np.mean(if_hz))), _safe(float(np.std(if_hz))), _safe(float(np.ptp(if_hz)))


def extract_features(signal: np.ndarray, sr: int,
                     n_fft: int = 2048, hop_length: int = 512,
                     n_mfcc: int = 13) -> np.ndarray:
    """Return a (35,) float64 feature vector for one mono audio signal."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    if len(x) == 0:
        return np.zeros(NUM_FEATURES, dtype=np.float64)

    # Peak-normalise
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak

    n = len(x)

    # Clamp FFT params to signal length (short impulse WAVs can be < 2048 samples)
    n_fft      = min(n_fft, n)
    hop_length = min(hop_length, max(1, n_fft // 2))

    # --- 1. Time-domain ---
    rms_energy  = _safe(float(np.sqrt(np.mean(x ** 2))))
    zcr         = _safe(float(np.sum(np.abs(np.diff(np.sign(x)))) / (2 * n)))
    mean_abs    = _safe(float(np.mean(np.abs(x))))
    std_amp     = _safe(float(np.std(x)))
    skewness    = _safe(float(_skew(x)))
    kurtosis    = _safe(float(_kurt(x)))
    abs_x       = np.abs(x) + 1e-12
    temp_flat   = _safe(float(np.exp(np.mean(np.log(abs_x))) / (np.mean(abs_x) + 1e-12)))
    crest       = _safe(float(np.max(np.abs(x)) / (rms_energy + 1e-12)))

    # --- 2. Frequency-domain ---
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag   = np.abs(np.fft.rfft(x, n=n))
    s_cent  = _safe(float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12)))
    s_bw    = _safe(float(np.sqrt(np.sum(((freqs - s_cent)**2) * mag) / (np.sum(mag) + 1e-12))))
    cum_mag = np.cumsum(mag)
    ridx    = np.searchsorted(cum_mag, 0.85 * cum_mag[-1])
    s_roll  = _safe(float(freqs[min(ridx, len(freqs) - 1)]))
    power   = mag ** 2 + 1e-12
    s_flat  = _safe(float(np.exp(np.mean(np.log(power))) / (np.mean(power) + 1e-12)))
    dom_f   = _safe(float(freqs[np.argmax(mag)]))
    s_ent   = _safe(_spectral_entropy(mag))
    s_spread = s_bw   # alias

    # --- 3. MFCCs ---
    if _LIBROSA:
        mfccs      = librosa.feature.mfcc(y=x.astype(np.float32), sr=sr,
                                           n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc_means = [_safe(float(np.mean(mfccs[i]))) for i in range(n_mfcc)]
    else:
        mfcc_means = [0.0] * n_mfcc

    # --- 4. Rhythm / energy shape ---
    frame_sz  = hop_length
    n_frames  = max(1, n // frame_sz)
    energies  = np.array([np.mean(x[i*frame_sz:(i+1)*frame_sz]**2) for i in range(n_frames)])
    onset_rate = _safe(float(np.sum(np.diff(energies) > 0) / (n / sr + 1e-12)))
    frms = np.sqrt(energies + 1e-12)
    rms_slope = _safe(float(np.polyfit(np.arange(n_frames, dtype=float), frms, 1)[0])) if n_frames > 1 else 0.0
    energy_var = _safe(float(np.var(energies)))
    analytic_r = _sig.hilbert(x)
    phase_r    = np.unwrap(np.angle(analytic_r))
    if_r       = np.diff(phase_r) / (2.0 * np.pi) * sr
    freq_mod   = _safe(float(np.std(if_r[np.isfinite(if_r)])))

    # --- 5. Instantaneous frequency ---
    if_mean, if_std, if_range = _if_stats(x, sr)

    # --- Assemble ---
    vec = np.array([
        rms_energy, zcr, mean_abs, std_amp, skewness, kurtosis, temp_flat, crest,
        s_cent, s_bw, s_roll, s_flat, dom_f, s_ent, s_spread,
        *mfcc_means,
        onset_rate, rms_slope, energy_var, freq_mod,
        if_mean, if_std, if_range,
    ], dtype=np.float64)

    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
