"""
intelligence/image/noise_injector.py
======================================
Image noise injection for Spandhan — Milestone E.

Supported noise types
---------------------
"gaussian"        : Additive white Gaussian noise (target PSNR in dB)
"salt_and_pepper" : Impulse noise with pixel density
"speckle"         : Multiplicative noise  (variance param)
"periodic"        : 2-D sinusoidal interference (freq_x, freq_y cycles/px)
"uniform"         : Additive uniform noise bounded to ±amplitude

All images are assumed to be float64 in [0, 1].
Output noisy images are clipped to [0, 1].
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# Noise type registry — keeps API explicit
IMAGE_NOISE_TYPES = (
    "gaussian",
    "salt_and_pepper",
    "speckle",
    "periodic",
    "uniform",
)


# ==================================================================
# Result dataclass
# ==================================================================


@dataclass
class ImageNoiseInjectionResult:
    """
    Typed result from one image noise injection operation.

    Attributes
    ----------
    clean_image : np.ndarray   float64 [0,1] (H, W)
    noisy_image : np.ndarray   float64 [0,1] (H, W)
    noise_type  : str
    noise_parameters : dict
    psnr_before : float        PSNR of clean vs noisy (dB)
    seed        : int or None
    processing_time_s : float
    warnings    : list[str]
    """

    clean_image: np.ndarray
    noisy_image: np.ndarray
    noise_type: str
    noise_parameters: dict[str, Any]
    psnr_before: float           # PSNR(clean, noisy) in dB — lower = more noise
    seed: Optional[int]
    processing_time_s: float
    warnings: list[str] = field(default_factory=list)


# ==================================================================
# Low-level injectors
# ==================================================================


def _psnr(reference: np.ndarray, distorted: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB); signals in [0,1]."""
    mse = float(np.mean((reference - distorted) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _inject_gaussian(
    image: np.ndarray,
    target_psnr_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Additive Gaussian noise calibrated to a target PSNR (dB)."""
    # sigma² = 1 / (10^(PSNR/10))
    sigma = float(10.0 ** (-target_psnr_db / 20.0))
    noise = rng.normal(0.0, sigma, size=image.shape)
    noisy = np.clip(image + noise, 0.0, 1.0)
    params = {"noise_type": "gaussian", "target_psnr_db": target_psnr_db, "sigma": sigma}
    return noisy, params


def _inject_salt_and_pepper(
    image: np.ndarray,
    density: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Salt-and-pepper impulse noise with given pixel density in (0,1)."""
    density = float(np.clip(density, 0.0, 1.0))
    noisy = image.copy()
    total = image.size
    n_corrupt = int(round(density * total))
    indices = rng.choice(total, size=n_corrupt, replace=False)
    flat = noisy.ravel()
    # half salt (1), half pepper (0)
    half = n_corrupt // 2
    flat[indices[:half]] = 1.0
    flat[indices[half:]] = 0.0
    params = {"noise_type": "salt_and_pepper", "density": density, "n_corrupt": n_corrupt}
    return noisy, params


def _inject_speckle(
    image: np.ndarray,
    variance: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Multiplicative speckle noise: noisy = image + image * N(0, var)."""
    variance = max(float(variance), 1e-6)
    noise = rng.normal(0.0, np.sqrt(variance), size=image.shape)
    noisy = np.clip(image + image * noise, 0.0, 1.0)
    params = {"noise_type": "speckle", "variance": variance}
    return noisy, params


def _inject_periodic(
    image: np.ndarray,
    freq_x: float,
    freq_y: float,
    amplitude: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Additive 2-D sinusoidal interference."""
    H, W = image.shape
    x = np.arange(W)[None, :] / W
    y = np.arange(H)[:, None] / H
    pattern = amplitude * np.sin(2 * np.pi * (freq_x * x + freq_y * y))
    noisy = np.clip(image + pattern, 0.0, 1.0)
    params = {
        "noise_type": "periodic",
        "freq_x": freq_x,
        "freq_y": freq_y,
        "amplitude": amplitude,
    }
    return noisy, params


def _inject_uniform(
    image: np.ndarray,
    amplitude: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Additive uniform noise bounded to ±amplitude."""
    amplitude = max(float(amplitude), 1e-6)
    noise = rng.uniform(-amplitude, amplitude, size=image.shape)
    noisy = np.clip(image + noise, 0.0, 1.0)
    params = {"noise_type": "uniform", "amplitude": amplitude}
    return noisy, params


# ==================================================================
# Public API
# ==================================================================


def inject_image_noise(
    image: np.ndarray,
    noise_type: str = "gaussian",
    *,
    # Gaussian
    target_psnr_db: float = 25.0,
    # Salt-and-pepper
    density: float = 0.05,
    # Speckle
    variance: float = 0.04,
    # Periodic
    freq_x: float = 5.0,
    freq_y: float = 3.0,
    periodic_amplitude: float = 0.2,
    # Uniform
    uniform_amplitude: float = 0.15,
    # Common
    seed: Optional[int] = None,
) -> ImageNoiseInjectionResult:
    """
    Inject noise into a float64 grayscale image in [0,1].

    Parameters
    ----------
    image       : np.ndarray  shape (H, W), float64, [0, 1]
    noise_type  : one of IMAGE_NOISE_TYPES
    ...         : noise-type-specific parameters
    seed        : random seed (reproducibility)

    Returns
    -------
    ImageNoiseInjectionResult
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("image must be a 2-D (H, W) array.")
    if not np.all(np.isfinite(image)):
        raise ValueError("image contains NaN or Inf values.")

    if noise_type not in IMAGE_NOISE_TYPES:
        raise ValueError(
            f"Unknown noise_type '{noise_type}'. Valid: {IMAGE_NOISE_TYPES}"
        )

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    warnings_out: list[str] = []

    if noise_type == "gaussian":
        noisy, params = _inject_gaussian(image, target_psnr_db, rng)
    elif noise_type == "salt_and_pepper":
        noisy, params = _inject_salt_and_pepper(image, density, rng)
    elif noise_type == "speckle":
        noisy, params = _inject_speckle(image, variance, rng)
    elif noise_type == "periodic":
        noisy, params = _inject_periodic(image, freq_x, freq_y, periodic_amplitude)
    elif noise_type == "uniform":
        noisy, params = _inject_uniform(image, uniform_amplitude, rng)
    else:
        raise RuntimeError(f"Unhandled noise_type: {noise_type}")

    psnr = _psnr(image, noisy)

    return ImageNoiseInjectionResult(
        clean_image=image,
        noisy_image=noisy,
        noise_type=noise_type,
        noise_parameters=params,
        psnr_before=psnr,
        seed=seed,
        processing_time_s=time.perf_counter() - t0,
        warnings=warnings_out,
    )
