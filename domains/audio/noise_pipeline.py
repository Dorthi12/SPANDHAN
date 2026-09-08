"""
domains/audio/noise_pipeline.py
================================
Audio noise injection engine for Spandhan — Milestone B.

Design principles
-----------------
* Extends (never duplicates) the existing generators.noise_generator module.
* Every injection operation records parameters AND independently measures
  the actual achieved SNR from the clean/noisy pair.
* The target_snr_db is configuration; measured_snr_db is computed fact.
  These two values must NEVER be confused or substituted for each other.
* The result object is typed so downstream stages never rely on key names.
* Future-compatible: replace _inject_* functions with real-noise loaders
  without changing the AudioNoisePipeline interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# ------------------------------------------------------------------
# Reuse the existing noise generators — no duplication
# ------------------------------------------------------------------
from generators.noise_generator import (
    add_colored_noise,
    add_gaussian_noise,
    add_impulse_noise,
    add_mixed_noise,
    add_periodic_noise,
)

# ------------------------------------------------------------------
# Supported noise types (must match NOISE_CLASSES in dataset_builder)
# ------------------------------------------------------------------
AUDIO_NOISE_TYPES = (
    "gaussian",
    "impulse",
    "periodic",
    "colored",
    "mixed",
)


# ==================================================================
# Result dataclass
# ==================================================================


@dataclass
class AudioNoiseInjectionResult:
    """
    Typed result from one audio noise injection operation.

    Attributes
    ----------
    clean_signal : np.ndarray
        Original clean signal (reference, never modified).
    noisy_signal : np.ndarray
        Clean signal with noise added.
    noise_signal : np.ndarray
        Isolated noise component (noisy - clean).
    noise_type : str
        Type of noise injected (e.g. "gaussian").
    target_snr_db : float or None
        Requested SNR in dB (None for noise types without SNR control).
    measured_snr_db : float
        Independently computed SNR from the clean/noisy pair.
        This is the ground truth; never equal to target_snr_db by definition.
    signal_energy : float
        Energy of the clean signal (sum of squares).
    noise_energy : float
        Energy of the injected noise.
    noise_parameters : dict
        All parameters passed to the noise generator.
    seed : int or None
        Random seed used.
    processing_time_s : float
        Wall-clock time for the injection step.
    warnings : list[str]
        Non-fatal warnings (e.g. SNR not achievable at target).
    """

    clean_signal: np.ndarray
    noisy_signal: np.ndarray
    noise_signal: np.ndarray
    noise_type: str
    target_snr_db: Optional[float]
    measured_snr_db: float
    signal_energy: float
    noise_energy: float
    noise_parameters: dict[str, Any]
    seed: Optional[int]
    processing_time_s: float
    warnings: list[str] = field(default_factory=list)


# ==================================================================
# Internal SNR measurement
# ==================================================================


def _measure_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """
    Compute SNR independently from the clean/noisy pair.

    Returns np.inf if noise power is zero (clean signal unchanged).
    Returns -np.inf if signal power is zero.
    """
    noise = noisy - clean
    signal_power = float(np.mean(clean ** 2))
    noise_power = float(np.mean(noise ** 2))

    if noise_power <= 0:
        return np.inf
    if signal_power <= 0:
        return -np.inf

    return float(10.0 * np.log10(signal_power / noise_power))


def _signal_energy(signal: np.ndarray) -> float:
    return float(np.sum(signal ** 2))


# ==================================================================
# Validation helpers
# ==================================================================


def _validate_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("Audio signal must be one-dimensional.")
    if signal.size == 0:
        raise ValueError("Audio signal cannot be empty.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Audio signal contains NaN or infinite values.")
    return signal


def _validate_sampling_rate(sampling_rate: float) -> float:
    sampling_rate = float(sampling_rate)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive and finite.")
    return sampling_rate


def _validate_noise_type(noise_type: str) -> str:
    noise_type = noise_type.lower().strip()
    if noise_type not in AUDIO_NOISE_TYPES:
        raise ValueError(
            f"Unsupported noise type: {noise_type!r}. "
            f"Supported: {AUDIO_NOISE_TYPES}"
        )
    return noise_type


# ==================================================================
# Per-type injection helpers
# (call existing noise_generator functions, capture params)
# ==================================================================


def _inject_gaussian(
    signal: np.ndarray,
    snr_db: float,
    seed: Optional[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    noisy = add_gaussian_noise(signal, snr_db=snr_db, random_state=seed)
    params = {"snr_db_target": float(snr_db)}
    return noisy, params


def _inject_impulse(
    signal: np.ndarray,
    probability: float,
    amplitude_factor: float,
    seed: Optional[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    noisy = add_impulse_noise(
        signal,
        probability=probability,
        amplitude_factor=amplitude_factor,
        random_state=seed,
    )
    params = {
        "probability": float(probability),
        "amplitude_factor": float(amplitude_factor),
    }
    return noisy, params


def _inject_periodic(
    signal: np.ndarray,
    sampling_rate: float,
    frequency: float,
    amplitude: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    noisy = add_periodic_noise(
        signal,
        sampling_rate=sampling_rate,
        frequency=frequency,
        amplitude=amplitude,
    )
    params = {
        "frequency_hz": float(frequency),
        "amplitude": float(amplitude),
    }
    return noisy, params


def _inject_colored(
    signal: np.ndarray,
    color: str,
    strength: float,
    seed: Optional[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    noisy = add_colored_noise(signal, color=color, strength=strength, random_state=seed)
    params = {"color": color, "strength": float(strength)}
    return noisy, params


def _inject_mixed(
    signal: np.ndarray,
    sampling_rate: float,
    snr_db: float,
    periodic_frequency: float,
    periodic_amplitude: float,
    impulse_probability: float,
    seed: Optional[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    noisy = add_mixed_noise(
        signal,
        sampling_rate=sampling_rate,
        snr_db=snr_db,
        periodic_frequency=periodic_frequency,
        periodic_amplitude=periodic_amplitude,
        impulse_probability=impulse_probability,
        random_state=seed,
    )
    params = {
        "snr_db_target": float(snr_db),
        "periodic_frequency_hz": float(periodic_frequency),
        "periodic_amplitude": float(periodic_amplitude),
        "impulse_probability": float(impulse_probability),
    }
    return noisy, params


# ==================================================================
# Public pipeline class
# ==================================================================


class AudioNoisePipeline:
    """
    Configurable audio noise injection engine.

    Usage
    -----
    pipeline = AudioNoisePipeline(
        noise_type="gaussian",
        target_snr_db=10.0,
        seed=42,
    )
    result = pipeline.inject(clean_signal, sampling_rate)

    All parameters are configurable per injection call so the same
    pipeline object can be reused across different signals.

    Future-compatibility
    --------------------
    Replace inject() with a version that loads noise from a real-world
    dataset without changing the AudioNoiseInjectionResult contract.
    """

    def __init__(
        self,
        noise_type: str = "gaussian",
        target_snr_db: float = 10.0,
        # Gaussian / Mixed
        snr_db: Optional[float] = None,
        # Impulse
        impulse_probability: float = 0.01,
        impulse_amplitude_factor: float = 5.0,
        # Periodic
        periodic_frequency: float = 50.0,
        periodic_amplitude: float = 0.1,
        # Colored
        color: str = "pink",
        colored_strength: float = 0.1,
        # Mixed extra
        mixed_periodic_frequency: float = 50.0,
        mixed_periodic_amplitude: float = 0.05,
        mixed_impulse_probability: float = 0.005,
        # Common
        seed: Optional[int] = None,
    ) -> None:
        self.noise_type = _validate_noise_type(noise_type)
        self.target_snr_db = float(target_snr_db)
        self.snr_db = float(snr_db) if snr_db is not None else target_snr_db
        self.impulse_probability = float(impulse_probability)
        self.impulse_amplitude_factor = float(impulse_amplitude_factor)
        self.periodic_frequency = float(periodic_frequency)
        self.periodic_amplitude = float(periodic_amplitude)
        self.color = color.lower()
        self.colored_strength = float(colored_strength)
        self.mixed_periodic_frequency = float(mixed_periodic_frequency)
        self.mixed_periodic_amplitude = float(mixed_periodic_amplitude)
        self.mixed_impulse_probability = float(mixed_impulse_probability)
        self.seed = seed

        # Validate impulse params
        if not 0 < self.impulse_probability <= 1:
            raise ValueError("impulse_probability must be in (0, 1].")
        if self.impulse_amplitude_factor <= 0:
            raise ValueError("impulse_amplitude_factor must be positive.")
        if self.colored_strength < 0:
            raise ValueError("colored_strength must be non-negative.")
        if self.color not in {"pink", "brown"}:
            raise ValueError("color must be 'pink' or 'brown'.")
        if self.periodic_amplitude < 0:
            raise ValueError("periodic_amplitude must be non-negative.")

    def inject(
        self,
        clean_signal: np.ndarray,
        sampling_rate: float,
    ) -> AudioNoiseInjectionResult:
        """
        Inject noise into clean_signal and return a fully documented result.

        Parameters
        ----------
        clean_signal : array-like
            Original clean audio signal.
        sampling_rate : float
            Sampling frequency in Hz.

        Returns
        -------
        AudioNoiseInjectionResult
            Contains both signals, isolated noise, and measured metrics.
        """
        t0 = time.perf_counter()
        clean = _validate_signal(clean_signal)
        sr = _validate_sampling_rate(sampling_rate)
        warnings: list[str] = []

        # ---- Dispatch ----
        noise_type = self.noise_type

        if noise_type == "gaussian":
            noisy, params = _inject_gaussian(clean, self.snr_db, self.seed)

        elif noise_type == "impulse":
            noisy, params = _inject_impulse(
                clean,
                probability=self.impulse_probability,
                amplitude_factor=self.impulse_amplitude_factor,
                seed=self.seed,
            )

        elif noise_type == "periodic":
            noisy, params = _inject_periodic(
                clean,
                sampling_rate=sr,
                frequency=self.periodic_frequency,
                amplitude=self.periodic_amplitude,
            )

        elif noise_type == "colored":
            noisy, params = _inject_colored(
                clean,
                color=self.color,
                strength=self.colored_strength,
                seed=self.seed,
            )

        elif noise_type == "mixed":
            noisy, params = _inject_mixed(
                clean,
                sampling_rate=sr,
                snr_db=self.snr_db,
                periodic_frequency=self.mixed_periodic_frequency,
                periodic_amplitude=self.mixed_periodic_amplitude,
                impulse_probability=self.mixed_impulse_probability,
                seed=self.seed,
            )

        else:
            raise ValueError(f"Unsupported noise type: {noise_type!r}")

        # ---- Post-injection validation ----
        if not np.all(np.isfinite(noisy)):
            raise RuntimeError(
                f"Noise injection ({noise_type}) produced non-finite values."
            )

        noise_component = noisy - clean

        measured_snr = _measure_snr(clean, noisy)
        sig_energy = _signal_energy(clean)
        noise_energy = _signal_energy(noise_component)

        # Warn if SNR is finite but far from target
        if (
            noise_type in ("gaussian", "mixed")
            and np.isfinite(measured_snr)
            and abs(measured_snr - self.target_snr_db) > 5.0
        ):
            warnings.append(
                f"Measured SNR ({measured_snr:.1f} dB) differs from target "
                f"({self.target_snr_db:.1f} dB) by more than 5 dB."
            )

        elapsed = time.perf_counter() - t0

        return AudioNoiseInjectionResult(
            clean_signal=clean.copy(),
            noisy_signal=noisy,
            noise_signal=noise_component,
            noise_type=noise_type,
            target_snr_db=self.target_snr_db,
            measured_snr_db=float(measured_snr),
            signal_energy=sig_energy,
            noise_energy=noise_energy,
            noise_parameters=params,
            seed=self.seed,
            processing_time_s=float(elapsed),
            warnings=warnings,
        )


# ==================================================================
# Convenience factory — mirrors build_audio_dataset() pattern
# ==================================================================


def inject_audio_noise(
    clean_signal: np.ndarray,
    sampling_rate: float,
    noise_type: str = "gaussian",
    target_snr_db: float = 10.0,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> AudioNoiseInjectionResult:
    """
    Convenience function: inject noise and return a result.

    Parameters
    ----------
    clean_signal : array-like
    sampling_rate : float
    noise_type : str
        One of AUDIO_NOISE_TYPES.
    target_snr_db : float
        Desired SNR in dB (used for gaussian and mixed noise).
    seed : int or None
    **kwargs :
        Additional type-specific parameters forwarded to AudioNoisePipeline.

    Returns
    -------
    AudioNoiseInjectionResult
    """
    pipeline = AudioNoisePipeline(
        noise_type=noise_type,
        target_snr_db=target_snr_db,
        snr_db=target_snr_db,
        seed=seed,
        **kwargs,
    )
    return pipeline.inject(clean_signal, sampling_rate)
