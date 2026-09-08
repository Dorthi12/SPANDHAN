"""
preprocessing/audio_denoising.py
==================================
Audio denoising pipeline for Spandhan — Milestone C.

Design principles
-----------------
* Delegates ALL Butterworth filtering to the existing dsp.filters module.
  No filter implementation is duplicated here.
* Adds wavelet thresholding (new capability not in dsp.wavelet which only
  wraps the forward DWT without reconstruction/thresholding).
* compare_signal_quality() from intelligence.noise.comparison is used for
  before/after metrics — no metric code is duplicated.
* AudioDenoisingResult is a typed dataclass — downstream stages never
  depend on dict key names.
* Method selection is explicit, transparent, and logged in the result.
* Every denoising method can be selected manually OR the pipeline can
  suggest a method based on the detected noise type.
* Staged denoising for mixed noise is a concrete sequence, not magic.

Future-compatibility
--------------------
Replace or extend _apply_* functions with real trained denoisers without
touching the AudioDenoisingResult contract or the selection logic.

Supported methods
-----------------
"lowpass"        : Butterworth low-pass filter
"highpass"       : Butterworth high-pass filter
"bandpass"       : Butterworth band-pass filter
"bandstop"       : Butterworth band-stop / notch filter
"wavelet"        : Wavelet soft-thresholding (Bayes shrink or universal)
"staged"         : Multi-step pipeline for mixed noise
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# ------------------------------------------------------------------
# Reuse existing DSP filters — never re-implement
# ------------------------------------------------------------------
from dsp.filters import (
    apply_bandpass,
    apply_bandstop,
    apply_highpass,
    apply_lowpass,
)

# ------------------------------------------------------------------
# Reuse existing before/after comparison — never re-implement
# ------------------------------------------------------------------
from intelligence.noise.comparison import compare_signal_quality

# pywt is optional: wavelet denoising will be skipped gracefully
# if it is not installed, falling back to lowpass.
try:
    import pywt as _pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _pywt = None           # type: ignore[assignment]
    _PYWT_AVAILABLE = False


# ==================================================================
# Constants
# ==================================================================

SUPPORTED_METHODS = (
    "lowpass",
    "highpass",
    "bandpass",
    "bandstop",
    "wavelet",
    "staged",
)

# Noise type → recommended method (transparent mapping, not magic)
_RECOMMENDED_METHOD: dict[str, str] = {
    "gaussian":           "wavelet",
    "impulse":            "bandpass",    # median not available; bandpass reduces spikes
    "periodic":           "bandstop",
    "colored":            "lowpass",
    "mixed":              "staged",
    # Classifier output aliases (capitalised)
    "Gaussian":           "wavelet",
    "Impulse":            "bandpass",
    "Periodic":           "bandstop",
    "Colored":            "lowpass",
    "Mixed":              "staged",
    "Clean":              "lowpass",     # nothing to remove; apply a gentle lowpass
}


# ==================================================================
# Result dataclass
# ==================================================================


@dataclass
class AudioDenoisingResult:
    """
    Typed result from one audio denoising operation.

    Attributes
    ----------
    noisy_signal : np.ndarray
        Input signal before denoising.
    cleaned_signal : np.ndarray
        Output signal after denoising.
    method : str
        Denoising method applied (e.g. "wavelet", "lowpass").
    method_parameters : dict
        All parameters used by the method.
    quality_comparison : dict
        Output of compare_signal_quality(clean, noisy, cleaned).
        Contains 'before', 'after', 'improvement' sub-dicts.
        Only populated when clean_reference is provided.
    snr_improvement_db : float or None
        SNR improvement in dB (after - before). None if no reference.
    rmse_reduction : float or None
        RMSE reduction (before - after). None if no reference.
    processing_time_s : float
    warnings : list[str]
    """

    noisy_signal: np.ndarray
    cleaned_signal: np.ndarray
    method: str
    method_parameters: dict[str, Any]
    quality_comparison: dict[str, Any]
    snr_improvement_db: Optional[float]
    rmse_reduction: Optional[float]
    processing_time_s: float
    warnings: list[str] = field(default_factory=list)


# ==================================================================
# Validation helpers
# ==================================================================


def _validate_signal(signal: np.ndarray, name: str = "signal") -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if signal.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return signal


def _validate_sampling_rate(sr: float) -> float:
    sr = float(sr)
    if not np.isfinite(sr) or sr <= 0:
        raise ValueError("sampling_rate must be positive and finite.")
    return sr


# ==================================================================
# Method recommendation
# ==================================================================


def recommend_method(noise_type: Optional[str]) -> str:
    """
    Return the recommended denoising method for a given noise type.

    Parameters
    ----------
    noise_type : str or None
        Noise type string (case-insensitive match attempted).

    Returns
    -------
    str
        One of SUPPORTED_METHODS.
    """
    if noise_type is None:
        return "lowpass"
    method = _RECOMMENDED_METHOD.get(noise_type)
    if method is None:
        method = _RECOMMENDED_METHOD.get(noise_type.lower(), "lowpass")
    # Fall back to lowpass if pywt is unavailable and wavelet was chosen
    if method == "wavelet" and not _PYWT_AVAILABLE:
        method = "lowpass"
    return method


# ==================================================================
# Low-level method implementations
# ==================================================================


def _apply_lowpass(
    signal: np.ndarray,
    sr: float,
    cutoff: float,
    order: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = apply_lowpass(signal, sr, cutoff, order)
    params = {"cutoff_hz": cutoff, "order": order, "filter_type": "lowpass"}
    return result, params


def _apply_highpass(
    signal: np.ndarray,
    sr: float,
    cutoff: float,
    order: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = apply_highpass(signal, sr, cutoff, order)
    params = {"cutoff_hz": cutoff, "order": order, "filter_type": "highpass"}
    return result, params


def _apply_bandpass(
    signal: np.ndarray,
    sr: float,
    low: float,
    high: float,
    order: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = apply_bandpass(signal, sr, low, high, order)
    params = {"low_hz": low, "high_hz": high, "order": order, "filter_type": "bandpass"}
    return result, params


def _apply_bandstop(
    signal: np.ndarray,
    sr: float,
    low: float,
    high: float,
    order: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    result = apply_bandstop(signal, sr, low, high, order)
    params = {"low_hz": low, "high_hz": high, "order": order, "filter_type": "bandstop"}
    return result, params


def _apply_wavelet(
    signal: np.ndarray,
    wavelet: str,
    level: Optional[int],
    threshold_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Wavelet soft-thresholding denoising.

    Uses universal threshold (VisuShrink) by default:
        lambda = sigma * sqrt(2 * log(N))
    where sigma is estimated from the finest detail coefficients
    using the robust MAD estimator:
        sigma = MAD(d1) / 0.6745

    threshold_mode : "soft" | "hard"
    """
    if not _PYWT_AVAILABLE:
        raise RuntimeError(
            "PyWavelets (pywt) is not installed. "
            "Cannot apply wavelet denoising."
        )

    n = len(signal)
    max_level = _pywt.dwt_max_level(n, _pywt.Wavelet(wavelet).dec_len)

    if max_level < 1:
        raise ValueError("Signal is too short for wavelet decomposition.")

    level = min(level if level is not None else 2, max_level)

    # Decompose
    coeffs = _pywt.wavedec(signal, wavelet, level=level)

    # Estimate noise sigma from finest detail band (MAD estimator)
    detail1 = coeffs[-1]
    if len(detail1) > 0:
        sigma = float(np.median(np.abs(detail1 - np.median(detail1))) / 0.6745)
    else:
        sigma = float(np.std(signal))

    # Soft-threshold high-frequency detail bands
    thresholded = [coeffs[0]]
    thresholds = []
    for i, c in enumerate(coeffs[1:], 1):
        # Level noise estimate
        c_mad = float(np.median(np.abs(c - np.median(c))) / 0.6745)
        c_mad = c_mad if c_mad > 1e-12 else sigma
        # VisuShrink threshold with soft multiplier to preserve signal energy
        th = float(c_mad * np.sqrt(2.0 * np.log(max(len(c), 2))) * 0.4)
        thresholds.append(th)
        thresholded.append(_pywt.threshold(c, th, mode=threshold_mode))

    cleaned = _pywt.waverec(thresholded, wavelet)

    # waverec may add one sample when signal length is odd — trim to match
    cleaned = cleaned[:n]

    params = {
        "wavelet": wavelet,
        "level": level,
        "threshold": float(thresholds[0]) if thresholds else 0.0,
        "threshold_mode": threshold_mode,
        "sigma_estimate": float(sigma),
    }
    return cleaned, params


def _apply_staged(
    signal: np.ndarray,
    sr: float,
    order: int,
    wavelet: str,
    wavelet_level: Optional[int],
    notch_freq: float,
    notch_bandwidth: float,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """
    Staged denoising for mixed noise:
      Step 1: Band-stop (notch) at periodic frequency
      Step 2: Wavelet soft-thresholding for Gaussian component
    Falls back gracefully if wavelet is unavailable.
    """
    warnings: list[str] = []
    nyquist = sr / 2.0
    params: dict[str, Any] = {"stages": []}

    # Stage 1: notch to remove periodic component
    low_notch = notch_freq - notch_bandwidth / 2.0
    high_notch = notch_freq + notch_bandwidth / 2.0
    low_notch = max(low_notch, 1.0)
    high_notch = min(high_notch, nyquist - 1.0)

    if 0 < low_notch < high_notch < nyquist:
        after_notch = apply_bandstop(signal, sr, low_notch, high_notch, order)
        params["stages"].append({
            "step": 1,
            "method": "bandstop",
            "low_hz": low_notch,
            "high_hz": high_notch,
            "order": order,
        })
    else:
        after_notch = signal.copy()
        warnings.append(
            f"Notch filter skipped: notch [{low_notch:.1f}, {high_notch:.1f}] Hz "
            f"outside valid range (0, {nyquist:.1f}) Hz."
        )

    # Stage 2: wavelet to remove Gaussian component
    if _PYWT_AVAILABLE:
        cleaned, wt_params = _apply_wavelet(after_notch, wavelet, wavelet_level, "soft")
        params["stages"].append({
            "step": 2,
            "method": "wavelet",
            **wt_params,
        })
    else:
        cutoff = min(nyquist * 0.8, nyquist - 1.0)
        cleaned = apply_lowpass(after_notch, sr, cutoff, order)
        params["stages"].append({
            "step": 2,
            "method": "lowpass_fallback",
            "cutoff_hz": cutoff,
            "order": order,
            "reason": "pywt unavailable",
        })
        warnings.append("pywt not installed; lowpass used in place of wavelet stage.")

    return cleaned, params, warnings


# ==================================================================
# Public denoiser class
# ==================================================================


class AudioDenoiser:
    """
    Configurable audio denoising pipeline.

    Usage
    -----
    denoiser = AudioDenoiser(method="wavelet")
    result = denoiser.denoise(noisy_signal, sampling_rate, clean_reference)

    The clean_reference is optional.  When provided, quality metrics
    (SNR improvement, RMSE reduction, correlation change, etc.) are
    computed via compare_signal_quality().

    When not provided (e.g. real-world scenario where clean is unknown),
    quality_comparison is empty and snr_improvement_db is None.

    Method parameters have sensible defaults but can all be overridden.
    """

    def __init__(
        self,
        method: str = "lowpass",
        # Lowpass / Highpass
        cutoff_hz: float = 4000.0,
        filter_order: int = 4,
        # Bandpass
        bandpass_low_hz: float = 300.0,
        bandpass_high_hz: float = 3400.0,
        # Bandstop (notch)
        notch_freq_hz: float = 50.0,
        notch_bandwidth_hz: float = 10.0,
        # Wavelet
        wavelet: str = "db4",
        wavelet_level: Optional[int] = None,
        wavelet_threshold_mode: str = "soft",
        # Staged (mixed)
        staged_notch_freq_hz: float = 50.0,
        staged_notch_bandwidth_hz: float = 10.0,
    ) -> None:
        method = method.lower().strip()
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported denoising method: {method!r}. "
                f"Supported: {SUPPORTED_METHODS}"
            )

        if filter_order < 1:
            raise ValueError("filter_order must be at least 1.")
        if wavelet_threshold_mode not in ("soft", "hard"):
            raise ValueError("wavelet_threshold_mode must be 'soft' or 'hard'.")

        self.method = method
        self.cutoff_hz = float(cutoff_hz)
        self.filter_order = int(filter_order)
        self.bandpass_low_hz = float(bandpass_low_hz)
        self.bandpass_high_hz = float(bandpass_high_hz)
        self.notch_freq_hz = float(notch_freq_hz)
        self.notch_bandwidth_hz = float(notch_bandwidth_hz)
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.wavelet_threshold_mode = wavelet_threshold_mode
        self.staged_notch_freq_hz = float(staged_notch_freq_hz)
        self.staged_notch_bandwidth_hz = float(staged_notch_bandwidth_hz)

    def denoise(
        self,
        noisy_signal: np.ndarray,
        sampling_rate: float,
        clean_reference: Optional[np.ndarray] = None,
    ) -> AudioDenoisingResult:
        """
        Denoise a signal and compute quality metrics.

        Parameters
        ----------
        noisy_signal : array-like
            1-D noisy audio signal.
        sampling_rate : float
            Sampling frequency in Hz.
        clean_reference : array-like or None
            Original clean signal.  When supplied, the full
            before/after quality comparison is computed.
            When None, quality_comparison is empty and
            snr_improvement_db / rmse_reduction are None.

        Returns
        -------
        AudioDenoisingResult
        """
        t0 = time.perf_counter()
        noisy = _validate_signal(noisy_signal, "noisy_signal")
        sr = _validate_sampling_rate(sampling_rate)
        warnings: list[str] = []

        if clean_reference is not None:
            clean_ref = _validate_signal(clean_reference, "clean_reference")
            if len(clean_ref) != len(noisy):
                raise ValueError(
                    "clean_reference and noisy_signal must have the same length."
                )
        else:
            clean_ref = None

        # ---- Dispatch to method ----
        method = self.method
        nyquist = sr / 2.0
        extra_warnings: list[str] = []

        if method == "lowpass":
            cutoff = min(self.cutoff_hz, nyquist - 1.0)
            cleaned, params = _apply_lowpass(noisy, sr, cutoff, self.filter_order)

        elif method == "highpass":
            cutoff = min(self.cutoff_hz, nyquist - 1.0)
            cleaned, params = _apply_highpass(noisy, sr, cutoff, self.filter_order)

        elif method == "bandpass":
            low = max(self.bandpass_low_hz, 1.0)
            high = min(self.bandpass_high_hz, nyquist - 1.0)
            if low >= high:
                raise ValueError(
                    f"bandpass_low_hz ({low}) must be < bandpass_high_hz ({high})."
                )
            cleaned, params = _apply_bandpass(noisy, sr, low, high, self.filter_order)

        elif method == "bandstop":
            low_n = max(self.notch_freq_hz - self.notch_bandwidth_hz / 2.0, 1.0)
            high_n = min(self.notch_freq_hz + self.notch_bandwidth_hz / 2.0, nyquist - 1.0)
            if low_n >= high_n:
                raise ValueError(
                    f"Computed notch band [{low_n:.1f}, {high_n:.1f}] Hz is invalid."
                )
            cleaned, params = _apply_bandstop(noisy, sr, low_n, high_n, self.filter_order)

        elif method == "wavelet":
            if not _PYWT_AVAILABLE:
                warnings.append(
                    "pywt not installed; falling back to lowpass denoising."
                )
                cutoff = min(self.cutoff_hz, nyquist - 1.0)
                cleaned, params = _apply_lowpass(noisy, sr, cutoff, self.filter_order)
                params["wavelet_fallback"] = True
            else:
                cleaned, params = _apply_wavelet(
                    noisy,
                    self.wavelet,
                    self.wavelet_level,
                    self.wavelet_threshold_mode,
                )

        elif method == "staged":
            cleaned, params, extra_warnings = _apply_staged(
                noisy,
                sr,
                self.filter_order,
                self.wavelet,
                self.wavelet_level,
                self.staged_notch_freq_hz,
                self.staged_notch_bandwidth_hz,
            )
            warnings.extend(extra_warnings)

        else:
            raise ValueError(f"Unsupported method: {method!r}")

        # ---- Validate output ----
        if not np.all(np.isfinite(cleaned)):
            raise RuntimeError(
                f"Denoising ({method}) produced non-finite values."
            )
        if len(cleaned) != len(noisy):
            # wavelet reconstruction can differ by ±1 — pad or trim
            if abs(len(cleaned) - len(noisy)) <= 2:
                cleaned = cleaned[: len(noisy)]
                if len(cleaned) < len(noisy):
                    cleaned = np.pad(cleaned, (0, len(noisy) - len(cleaned)), "edge")
            else:
                raise RuntimeError(
                    f"Denoising ({method}) changed signal length "
                    f"from {len(noisy)} to {len(cleaned)}."
                )

        # ---- Before/after quality comparison ----
        if clean_ref is not None:
            comparison = compare_signal_quality(clean_ref, noisy, cleaned, sr)
            snr_improvement = float(comparison["improvement"]["snr_db"])
            rmse_reduction = float(comparison["improvement"]["rmse_reduction"])
        else:
            comparison = {}
            snr_improvement = None
            rmse_reduction = None

        elapsed = time.perf_counter() - t0

        return AudioDenoisingResult(
            noisy_signal=noisy.copy(),
            cleaned_signal=cleaned,
            method=method,
            method_parameters=params,
            quality_comparison=comparison,
            snr_improvement_db=snr_improvement,
            rmse_reduction=rmse_reduction,
            processing_time_s=float(elapsed),
            warnings=warnings,
        )


# ==================================================================
# Convenience factory
# ==================================================================


def denoise_audio(
    noisy_signal: np.ndarray,
    sampling_rate: float,
    method: str = "wavelet",
    clean_reference: Optional[np.ndarray] = None,
    noise_type: Optional[str] = None,
    **kwargs: Any,
) -> AudioDenoisingResult:
    """
    Convenience function: denoise an audio signal.

    If method is "auto", the method is selected based on noise_type.

    Parameters
    ----------
    noisy_signal : array-like
    sampling_rate : float
    method : str
        Denoising method or "auto".
    clean_reference : array-like or None
        Known clean reference for quality metrics.
    noise_type : str or None
        Used only when method == "auto".
    **kwargs :
        Forwarded to AudioDenoiser constructor.

    Returns
    -------
    AudioDenoisingResult
    """
    if method == "auto":
        method = recommend_method(noise_type)

    denoiser = AudioDenoiser(method=method, **kwargs)
    return denoiser.denoise(noisy_signal, sampling_rate, clean_reference)
