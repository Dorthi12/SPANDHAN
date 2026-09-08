"""
intelligence/image/comparison.py
===================================
Image quality comparison — before/after denoising — for Spandhan Milestone E.

compare_image_quality(clean, noisy, cleaned) -> dict
  Returns {"before": {...}, "after": {...}, "improvement": {...}}

Metrics
-------
psnr_db      — Peak Signal-to-Noise Ratio (dB)
ssim         — Structural Similarity Index [-1, 1]
snr_db       — Signal-to-Noise Ratio (dB)
rmse         — Root Mean Squared Error [0, 1]
hist_corr    — Pearson correlation between normalised histograms

Uses skimage.metrics for PSNR and SSIM — gracefully falls back to
numpy-only implementations if skimage is unavailable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# skimage is optional; fall back to numpy-only if missing
try:
    from skimage.metrics import (
        peak_signal_noise_ratio as _sk_psnr,
        structural_similarity as _sk_ssim,
    )
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False


# ==================================================================
# Individual metric helpers
# ==================================================================


def _psnr(reference: np.ndarray, distorted: np.ndarray) -> float:
    mse = float(np.mean((reference - distorted) ** 2))
    if mse == 0.0:
        return float("inf")
    if _SKIMAGE_AVAILABLE:
        return float(_sk_psnr(reference, distorted, data_range=1.0))
    return float(10.0 * np.log10(1.0 / mse))


def _ssim(reference: np.ndarray, distorted: np.ndarray) -> float:
    if _SKIMAGE_AVAILABLE:
        return float(_sk_ssim(reference, distorted, data_range=1.0))
    # Fallback: simplified SSIM
    mu_x, mu_y = reference.mean(), distorted.mean()
    sig_x = float(np.std(reference))
    sig_y = float(np.std(distorted))
    sig_xy = float(np.mean((reference - mu_x) * (distorted - mu_y)))
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2)
    return float(num / den) if den != 0.0 else 0.0


def _snr_db(reference: np.ndarray, distorted: np.ndarray) -> float:
    signal_power = float(np.mean(reference ** 2))
    noise_power = float(np.mean((reference - distorted) ** 2))
    if noise_power == 0.0:
        return float("inf")
    if signal_power == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def _rmse(reference: np.ndarray, distorted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((reference - distorted) ** 2)))


def _hist_corr(reference: np.ndarray, distorted: np.ndarray, bins: int = 64) -> float:
    """Pearson correlation between normalised intensity histograms."""
    h_ref, _ = np.histogram(reference.ravel(), bins=bins, range=(0, 1))
    h_dis, _ = np.histogram(distorted.ravel(), bins=bins, range=(0, 1))
    h_ref = h_ref.astype(np.float64) / (h_ref.sum() + 1e-12)
    h_dis = h_dis.astype(np.float64) / (h_dis.sum() + 1e-12)
    if h_ref.std() == 0 or h_dis.std() == 0:
        return 0.0
    return float(np.corrcoef(h_ref, h_dis)[0, 1])


def _metrics(reference: np.ndarray, distorted: np.ndarray) -> dict[str, float]:
    return {
        "psnr_db": _psnr(reference, distorted),
        "ssim": _ssim(reference, distorted),
        "snr_db": _snr_db(reference, distorted),
        "rmse": _rmse(reference, distorted),
        "hist_corr": _hist_corr(reference, distorted),
    }


# ==================================================================
# Public API
# ==================================================================


def compare_image_quality(
    clean: np.ndarray,
    noisy: np.ndarray,
    cleaned: np.ndarray,
) -> dict[str, Any]:
    """
    Compare image quality before and after denoising.

    Parameters
    ----------
    clean   : ground-truth clean image  (H, W) float64 [0,1]
    noisy   : noise-corrupted image     (H, W) float64 [0,1]
    cleaned : denoised image            (H, W) float64 [0,1]

    Returns
    -------
    dict with keys "before", "after", "improvement"
    Each sub-dict has: psnr_db, ssim, snr_db, rmse, hist_corr
    "improvement" = after - before  (positive = better, except rmse where negative = better)
    """
    before = _metrics(clean, noisy)
    after = _metrics(clean, cleaned)
    improvement: dict[str, float] = {}
    for k in before:
        bv = before[k]
        av = after[k]
        if k == "rmse":
            # Lower is better: improvement = before - after (positive = improved)
            improvement[k] = (bv - av) if (np.isfinite(bv) and np.isfinite(av)) else float("nan")
        else:
            # Higher is better: improvement = after - before
            improvement[k] = (av - bv) if (np.isfinite(bv) and np.isfinite(av)) else float("nan")

    return {"before": before, "after": after, "improvement": improvement}
