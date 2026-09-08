"""
domains/audio/analyzer.py
==========================
Synthetic audio signal analyzer for Spandhan — Milestone B.

Computes a comprehensive set of time-domain, spectral, and noise
metrics for a 1-D audio signal.  Delegates all DSP computations to
the existing dsp.* modules — no duplication.

The analyzer works on any stage: clean, noisy, or cleaned.
The caller is responsible for calling it three times and comparing.

Noise classification
--------------------
Attempts to load the project's trained noise classifier from the
default model path.  If the model is unavailable, the result is
annotated with a clear "model unavailable" marker rather than crashing
or fabricating predictions.

Future-compatibility
--------------------
Replace _load_classifier() with a real-dataset trained model without
changing the AudioSignalAnalysis result contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.signal import welch as _welch
from scipy.stats import entropy as _scipy_entropy

# ------------------------------------------------------------------
# Reuse existing DSP engine — never re-implement
# ------------------------------------------------------------------
from dsp.fft import compute_fft
from dsp.psd import compute_psd
from dsp.stft import compute_stft

# ------------------------------------------------------------------
# Reuse existing noise intelligence
# ------------------------------------------------------------------
from core.config import DEFAULT_MODEL_PATH, DEFAULT_CONFIDENCE_THRESHOLD
from intelligence.noise.features import extract_noise_features


# ==================================================================
# Result dataclass
# ==================================================================


@dataclass
class AudioSignalMetrics:
    """
    All scalar metrics for one signal stage (clean / noisy / cleaned).

    Time-domain
    -----------
    mean, rms, variance, std, peak_amplitude, crest_factor,
    zero_crossing_rate, signal_energy

    Spectral
    --------
    dominant_frequency_hz, spectral_centroid_hz, spectral_bandwidth_hz,
    spectral_flatness, spectral_entropy, spectral_rolloff_hz,
    spectral_rolloff_pct (rolloff as % of Nyquist),
    total_spectral_power

    Arrays (kept for visualisation)
    --------------------------------
    fft_frequencies, fft_magnitude,
    psd_frequencies, psd_values,
    stft_frequencies, stft_times, stft_magnitude
    """

    # Time-domain scalars
    mean: float = 0.0
    rms: float = 0.0
    variance: float = 0.0
    std: float = 0.0
    peak_amplitude: float = 0.0
    crest_factor: float = 0.0
    zero_crossing_rate: float = 0.0
    signal_energy: float = 0.0

    # Spectral scalars
    dominant_frequency_hz: float = 0.0
    spectral_centroid_hz: float = 0.0
    spectral_bandwidth_hz: float = 0.0
    spectral_flatness: float = 0.0
    spectral_entropy: float = 0.0
    spectral_rolloff_hz: float = 0.0
    spectral_rolloff_pct: float = 0.0
    total_spectral_power: float = 0.0

    # Arrays (None until computed)
    fft_frequencies: Optional[np.ndarray] = field(default=None, repr=False)
    fft_magnitude: Optional[np.ndarray] = field(default=None, repr=False)
    psd_frequencies: Optional[np.ndarray] = field(default=None, repr=False)
    psd_values: Optional[np.ndarray] = field(default=None, repr=False)
    stft_frequencies: Optional[np.ndarray] = field(default=None, repr=False)
    stft_times: Optional[np.ndarray] = field(default=None, repr=False)
    stft_magnitude: Optional[np.ndarray] = field(default=None, repr=False)

    # Noise-feature vector (20 features used by the classifier)
    noise_features: Optional[dict[str, float]] = field(default=None, repr=False)

    # Error/warning from computation
    warnings: list[str] = field(default_factory=list)


@dataclass
class AudioClassificationResult:
    """
    Result from the noise classifier.

    Explicitly separates:
    - ground_truth  : the label the synthetic generator assigned
    - prediction    : what the ML model predicted
    - confidence    : model's own probability for its top class
    - status        : "Detected" / "Low confidence" / "Model unavailable"
    """

    ground_truth: Optional[str] = None       # from generator, not from model
    prediction: Optional[str] = None         # from ML model
    raw_prediction: Optional[str] = None     # before threshold filtering
    confidence: float = 0.0
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    is_unknown: bool = True
    status: str = "Model unavailable"
    probabilities: Optional[np.ndarray] = field(default=None, repr=False)
    classes: Optional[list[str]] = field(default=None, repr=False)
    model_available: bool = False
    feature_names: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AudioSignalAnalysis:
    """
    Complete analysis result for one (clean / noisy / cleaned) signal.

    stage : str
        One of "clean", "noisy", "cleaned".
    signal : np.ndarray
        The signal that was analysed.
    sampling_rate : float
    metrics : AudioSignalMetrics
    classification : AudioClassificationResult
    processing_time_s : float
    warnings : list[str]
    """

    stage: str
    signal: np.ndarray
    sampling_rate: float
    metrics: AudioSignalMetrics
    classification: AudioClassificationResult
    processing_time_s: float
    warnings: list[str] = field(default_factory=list)


# ==================================================================
# Classifier loader (lazy, cached at module level)
# ==================================================================

_CLASSIFIER_CACHE: dict[str, Any] = {}


def _load_classifier() -> Optional[Any]:
    """
    Attempt to load the project's noise classifier from disk.

    Returns the NoiseClassifier instance or None if unavailable.
    The result is cached so subsequent calls are free.
    """
    model_path = DEFAULT_MODEL_PATH

    if "classifier" in _CLASSIFIER_CACHE:
        return _CLASSIFIER_CACHE["classifier"]

    if not Path(model_path).exists():
        _CLASSIFIER_CACHE["classifier"] = None
        return None

    try:
        from intelligence.noise.classifier import NoiseClassifier
        clf = NoiseClassifier.load(model_path)
        _CLASSIFIER_CACHE["classifier"] = clf
        return clf
    except Exception:
        _CLASSIFIER_CACHE["classifier"] = None
        return None


def _reset_classifier_cache() -> None:
    """Test helper: clear the cached classifier so it is reloaded."""
    _CLASSIFIER_CACHE.clear()


# ==================================================================
# Metric computation helpers
# ==================================================================


def _compute_time_domain(signal: np.ndarray, sr: float) -> dict[str, float]:
    n = len(signal)
    mean = float(np.mean(signal))
    rms = float(np.sqrt(np.mean(signal ** 2)))
    variance = float(np.var(signal))
    std = float(np.std(signal))
    peak = float(np.max(np.abs(signal)))
    crest = peak / rms if rms > 0 else 0.0
    energy = float(np.sum(signal ** 2))

    if n > 1:
        centered = signal - mean
        signs = np.sign(centered)
        signs[signs == 0] = 1.0
        crossings = np.sum(signs[:-1] != signs[1:])
        zcr = float(crossings / (n - 1))
    else:
        zcr = 0.0

    return {
        "mean": mean,
        "rms": rms,
        "variance": variance,
        "std": std,
        "peak_amplitude": peak,
        "crest_factor": float(crest),
        "zero_crossing_rate": zcr,
        "signal_energy": energy,
    }


def _compute_spectral_scalars(
    psd_freqs: np.ndarray,
    psd_vals: np.ndarray,
    sampling_rate: float,
) -> dict[str, float]:
    """
    Derive spectral scalar metrics from a pre-computed PSD.
    """
    total_power = float(np.trapezoid(psd_vals, psd_freqs))

    if total_power <= 0 or len(psd_freqs) == 0:
        return {
            "dominant_frequency_hz": 0.0,
            "spectral_centroid_hz": 0.0,
            "spectral_bandwidth_hz": 0.0,
            "spectral_flatness": 0.0,
            "spectral_entropy": 0.0,
            "spectral_rolloff_hz": 0.0,
            "spectral_rolloff_pct": 0.0,
            "total_spectral_power": 0.0,
        }

    # Dominant frequency
    dominant = float(psd_freqs[np.argmax(psd_vals)])

    # Spectral centroid
    centroid = float(np.trapezoid(psd_freqs * psd_vals, psd_freqs) / total_power)

    # Spectral bandwidth (weighted std around centroid)
    bandwidth = float(
        np.sqrt(
            np.trapezoid((psd_freqs - centroid) ** 2 * psd_vals, psd_freqs)
            / total_power
        )
    )

    # Spectral flatness (geometric / arithmetic mean of PSD)
    pos = psd_vals[psd_vals > 0]
    if len(pos) > 0:
        flatness = float(
            np.exp(np.mean(np.log(pos))) / np.mean(pos)
            if np.mean(pos) > 0
            else 0.0
        )
    else:
        flatness = 0.0

    # Spectral entropy (normalised)
    prob = psd_vals / np.sum(psd_vals)
    prob = prob[prob > 0]
    ent = float(-np.sum(prob * np.log2(prob)))
    if len(prob) > 1:
        ent /= np.log2(len(prob))

    # 85% spectral rolloff
    cumulative = np.cumsum(psd_vals)
    threshold = 0.85 * cumulative[-1]
    rolloff_idx = int(np.searchsorted(cumulative, threshold))
    rolloff_idx = min(rolloff_idx, len(psd_freqs) - 1)
    rolloff_hz = float(psd_freqs[rolloff_idx])
    nyquist = sampling_rate / 2.0
    rolloff_pct = float(100.0 * rolloff_hz / nyquist) if nyquist > 0 else 0.0

    return {
        "dominant_frequency_hz": dominant,
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_flatness": flatness,
        "spectral_entropy": ent,
        "spectral_rolloff_hz": rolloff_hz,
        "spectral_rolloff_pct": rolloff_pct,
        "total_spectral_power": total_power,
    }


# ==================================================================
# Per-stage analysis
# ==================================================================


def _safe_stft(
    signal: np.ndarray,
    sr: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute STFT with a safe nperseg adapted to signal length.
    Returns (None, None, None) if the signal is too short.
    """
    n = len(signal)
    nperseg = min(256, n)
    if nperseg < 2:
        return None, None, None
    try:
        freqs, times, mag = compute_stft(signal, sr, nperseg=nperseg)
        return freqs, times, mag
    except Exception:
        return None, None, None


def analyze_audio_signal(
    signal: np.ndarray,
    sampling_rate: float,
    stage: str = "unknown",
    ground_truth_noise: Optional[str] = None,
    measured_snr_db: Optional[float] = None,
) -> AudioSignalAnalysis:
    """
    Analyse a single audio signal stage (clean / noisy / cleaned).

    Parameters
    ----------
    signal : array-like
        1-D float audio signal.
    sampling_rate : float
        Sampling frequency in Hz.
    stage : str
        Label for this stage: "clean", "noisy", or "cleaned".
    ground_truth_noise : str or None
        The synthetic noise label from the generator.
        Never passed to the classifier — used only for display.
    measured_snr_db : float or None
        Pre-computed SNR (from noise injection result).
        Used in noise feature extraction when provided.

    Returns
    -------
    AudioSignalAnalysis
    """
    t0 = time.perf_counter()
    signal = np.asarray(signal, dtype=np.float64)
    sampling_rate = float(sampling_rate)
    warnings: list[str] = []

    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size == 0:
        raise ValueError("signal cannot be empty.")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal contains NaN or infinite values.")

    # ---- Time-domain metrics ----
    td = _compute_time_domain(signal, sampling_rate)

    # ---- FFT ----
    try:
        fft_freqs, fft_mag = compute_fft(signal, sampling_rate)
    except Exception as exc:
        fft_freqs, fft_mag = None, None
        warnings.append(f"FFT failed: {exc}")

    # ---- PSD (Welch) ----
    try:
        psd_freqs, psd_vals = compute_psd(signal, sampling_rate)
    except Exception as exc:
        psd_freqs = np.array([])
        psd_vals = np.array([])
        warnings.append(f"PSD failed: {exc}")

    # ---- Spectral scalars from PSD ----
    if psd_freqs is not None and len(psd_freqs) > 0:
        spectral = _compute_spectral_scalars(psd_freqs, psd_vals, sampling_rate)
    else:
        spectral = {
            "dominant_frequency_hz": 0.0,
            "spectral_centroid_hz": 0.0,
            "spectral_bandwidth_hz": 0.0,
            "spectral_flatness": 0.0,
            "spectral_entropy": 0.0,
            "spectral_rolloff_hz": 0.0,
            "spectral_rolloff_pct": 0.0,
            "total_spectral_power": 0.0,
        }
        warnings.append("Spectral scalars could not be computed (no PSD).")

    # ---- STFT ----
    stft_freqs, stft_times, stft_mag = _safe_stft(signal, sampling_rate)
    if stft_freqs is None:
        warnings.append("STFT skipped (signal too short).")

    # ---- Noise feature vector (20 features) ----
    try:
        # SNR must be finite for extract_noise_features — supply only when valid
        snr_for_features = (
            float(measured_snr_db)
            if (measured_snr_db is not None and np.isfinite(measured_snr_db))
            else None
        )
        noise_feats = extract_noise_features(signal, sampling_rate, snr_db=snr_for_features)
        # If SNR was not finite (e.g. clean signal), patch it in separately
        if measured_snr_db is not None:
            noise_feats["snr_db"] = float(measured_snr_db)
    except Exception as exc:
        noise_feats = None
        warnings.append(f"Noise feature extraction failed: {exc}")

    # ---- Assemble metrics ----
    metrics = AudioSignalMetrics(
        mean=td["mean"],
        rms=td["rms"],
        variance=td["variance"],
        std=td["std"],
        peak_amplitude=td["peak_amplitude"],
        crest_factor=td["crest_factor"],
        zero_crossing_rate=td["zero_crossing_rate"],
        signal_energy=td["signal_energy"],
        dominant_frequency_hz=spectral["dominant_frequency_hz"],
        spectral_centroid_hz=spectral["spectral_centroid_hz"],
        spectral_bandwidth_hz=spectral["spectral_bandwidth_hz"],
        spectral_flatness=spectral["spectral_flatness"],
        spectral_entropy=spectral["spectral_entropy"],
        spectral_rolloff_hz=spectral["spectral_rolloff_hz"],
        spectral_rolloff_pct=spectral["spectral_rolloff_pct"],
        total_spectral_power=spectral["total_spectral_power"],
        fft_frequencies=fft_freqs,
        fft_magnitude=fft_mag,
        psd_frequencies=psd_freqs,
        psd_values=psd_vals,
        stft_frequencies=stft_freqs,
        stft_times=stft_times,
        stft_magnitude=stft_mag,
        noise_features=noise_feats,
        warnings=warnings[:],
    )

    # ---- Noise classification ----
    clf = _load_classifier()
    clf_result = AudioClassificationResult(
        ground_truth=ground_truth_noise,
        model_available=(clf is not None),
    )

    if clf is None:
        clf_result.status = "Model unavailable"
        clf_result.warnings = ["Noise model not found. Showing synthetic ground truth only."]
    elif noise_feats is None:
        clf_result.status = "Feature extraction failed"
        clf_result.warnings = ["Cannot classify: noise feature extraction failed."]
    else:
        try:
            # Classifier requires finite snr_db — use NaN-safe value
            snr_for_clf = noise_feats.get("snr_db", float("nan"))
            if not np.isfinite(snr_for_clf):
                # For the clean signal, SNR = +inf. Cap to 100 as dataset_builder does.
                noise_feats_for_clf = dict(noise_feats)
                noise_feats_for_clf["snr_db"] = 100.0
            else:
                noise_feats_for_clf = noise_feats

            clf_raw = clf.predict(noise_feats_for_clf)

            clf_result.prediction = clf_raw["prediction"]
            clf_result.raw_prediction = clf_raw["raw_prediction"]
            clf_result.confidence = float(clf_raw["confidence"])
            clf_result.threshold = float(clf_raw["threshold"])
            clf_result.is_unknown = bool(clf_raw["is_unknown"])
            clf_result.probabilities = clf_raw["probabilities"]
            clf_result.classes = clf_raw["classes"]
            clf_result.feature_names = list(clf_raw["feature_names"])
            clf_result.status = (
                "Low confidence" if clf_raw["is_unknown"] else "Detected"
            )

        except Exception as exc:
            clf_result.status = "Classification error"
            clf_result.warnings = [f"Classifier error: {exc}"]

    elapsed = time.perf_counter() - t0

    return AudioSignalAnalysis(
        stage=stage,
        signal=signal.copy(),
        sampling_rate=sampling_rate,
        metrics=metrics,
        classification=clf_result,
        processing_time_s=float(elapsed),
        warnings=warnings,
    )
