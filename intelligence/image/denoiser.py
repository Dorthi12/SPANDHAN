"""
intelligence/image/denoiser.py
================================
Image denoising pipeline for Spandhan — Milestone E.

Supported methods
-----------------
"gaussian_blur"  : Gaussian blur (kernel_size, sigma)
"median"         : Median filter (kernel_size — must be odd)
"bilateral"      : Bilateral filter (d, sigma_color, sigma_space) — edge-preserving
"wavelet"        : 2-D Wavelet soft-thresholding via PyWavelets (db4, BayesShrink)
"nlm"            : Non-Local Means via cv2.fastNlMeansDenoising

Graceful fallbacks
------------------
* If cv2 is unavailable, bilateral and nlm fall back to gaussian_blur.
* If pywt is unavailable, wavelet falls back to gaussian_blur.

All images are float64 [0, 1] (H, W).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from intelligence.image.comparison import compare_image_quality

# ------------------------------------------------------------------
# Optional dependencies
# ------------------------------------------------------------------
try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None          # type: ignore[assignment]
    _CV2_AVAILABLE = False

try:
    import pywt as _pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _pywt = None         # type: ignore[assignment]
    _PYWT_AVAILABLE = False

SUPPORTED_METHODS = (
    "gaussian_blur",
    "median",
    "bilateral",
    "wavelet",
    "nlm",
)

# Noise → recommended method
_RECOMMENDED_METHOD: dict[str, str] = {
    "gaussian":        "wavelet",
    "salt_and_pepper": "median",
    "speckle":         "bilateral",
    "periodic":        "gaussian_blur",
    "uniform":         "gaussian_blur",
}


# ==================================================================
# Result dataclass
# ==================================================================


@dataclass
class ImageDenoisingResult:
    """
    Typed result from one image denoising operation.

    Attributes
    ----------
    noisy_image     : np.ndarray   float64 [0,1] (H, W)
    cleaned_image   : np.ndarray   float64 [0,1] (H, W)
    method          : str
    method_parameters : dict
    quality_comparison : dict      from compare_image_quality (if ref given)
    psnr_improvement  : float or None   after_psnr - before_psnr
    ssim_improvement  : float or None   after_ssim - before_ssim
    processing_time_s : float
    warnings        : list[str]
    """

    noisy_image: np.ndarray
    cleaned_image: np.ndarray
    method: str
    method_parameters: dict[str, Any]
    quality_comparison: dict[str, Any]
    psnr_improvement: Optional[float]
    ssim_improvement: Optional[float]
    processing_time_s: float
    warnings: list[str] = field(default_factory=list)


# ==================================================================
# Low-level method implementations
# ==================================================================


def _apply_gaussian_blur(
    image: np.ndarray, kernel_size: int, sigma: float
) -> tuple[np.ndarray, dict[str, Any]]:
    if kernel_size % 2 == 0:
        kernel_size += 1
    cleaned = gaussian_filter(image, sigma=sigma)
    cleaned = np.clip(cleaned, 0.0, 1.0)
    params = {"method": "gaussian_blur", "kernel_size": kernel_size, "sigma": sigma}
    return cleaned, params


def _apply_median(
    image: np.ndarray, kernel_size: int
) -> tuple[np.ndarray, dict[str, Any]]:
    if kernel_size % 2 == 0:
        kernel_size += 1
    cleaned = median_filter(image, size=kernel_size)
    cleaned = np.clip(cleaned, 0.0, 1.0)
    params = {"method": "median", "kernel_size": kernel_size}
    return cleaned, params


def _apply_bilateral(
    image: np.ndarray,
    d: int,
    sigma_color: float,
    sigma_space: float,
    warnings_out: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    if _CV2_AVAILABLE:
        img_u8 = (image * 255.0).astype(np.uint8)
        out_u8 = _cv2.bilateralFilter(img_u8, d, sigma_color, sigma_space)
        cleaned = out_u8.astype(np.float64) / 255.0
    else:
        warnings_out.append("cv2 not installed; bilateral fell back to gaussian_blur.")
        cleaned, _ = _apply_gaussian_blur(image, 5, sigma_space / 20.0)
    cleaned = np.clip(cleaned, 0.0, 1.0)
    params = {
        "method": "bilateral",
        "d": d,
        "sigma_color": sigma_color,
        "sigma_space": sigma_space,
        "used_cv2": _CV2_AVAILABLE,
    }
    return cleaned, params


def _apply_wavelet(
    image: np.ndarray,
    wavelet: str,
    level: Optional[int],
    threshold_mode: str,
    warnings_out: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    if not _PYWT_AVAILABLE:
        warnings_out.append("pywt not installed; wavelet fell back to gaussian_blur.")
        cleaned, params = _apply_gaussian_blur(image, 5, 1.0)
        params["method"] = "wavelet"
        params["fallback"] = "gaussian_blur"
        return cleaned, params

    H, W = image.shape
    max_level_h = _pywt.dwt_max_level(H, _pywt.Wavelet(wavelet).dec_len)
    max_level_w = _pywt.dwt_max_level(W, _pywt.Wavelet(wavelet).dec_len)
    max_level = min(max_level_h, max_level_w)
    level = min(level if level is not None else 2, max(max_level, 1))

    # 2-D DWT
    coeffs = _pywt.wavedec2(image, wavelet, level=level)

    # Estimate noise sigma from finest detail subband (MAD)
    detail_finest = coeffs[-1][0]  # LH subband at finest level
    sigma = float(np.median(np.abs(detail_finest - np.median(detail_finest))) / 0.6745)
    if sigma < 1e-12:
        sigma = float(np.std(image) * 0.1)

    # Per-subband BayesShrink
    new_coeffs = [coeffs[0]]  # keep approximation unchanged
    for subband_tuple in coeffs[1:]:
        new_sub = []
        for sb in subband_tuple:
            c_mad = float(np.median(np.abs(sb - np.median(sb))) / 0.6745)
            c_mad = c_mad if c_mad > 1e-12 else sigma
            th = float(c_mad * np.sqrt(2.0 * np.log(max(sb.size, 2))) * 0.4)
            new_sub.append(_pywt.threshold(sb, th, mode=threshold_mode))
        new_coeffs.append(tuple(new_sub))

    cleaned = _pywt.waverec2(new_coeffs, wavelet)
    # Trim to original size (waverec2 may add 1 row/col for odd dims)
    cleaned = cleaned[:H, :W]
    cleaned = np.clip(cleaned, 0.0, 1.0)

    params = {
        "method": "wavelet",
        "wavelet": wavelet,
        "level": level,
        "threshold_mode": threshold_mode,
        "sigma_estimate": sigma,
    }
    return cleaned, params


def _apply_nlm(
    image: np.ndarray,
    h: float,
    template_window: int,
    search_window: int,
    warnings_out: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    if _CV2_AVAILABLE:
        img_u8 = (image * 255.0).astype(np.uint8)
        out_u8 = _cv2.fastNlMeansDenoising(
            img_u8,
            h=h,
            templateWindowSize=template_window,
            searchWindowSize=search_window,
        )
        cleaned = out_u8.astype(np.float64) / 255.0
    else:
        warnings_out.append("cv2 not installed; NLM fell back to gaussian_blur.")
        cleaned, _ = _apply_gaussian_blur(image, 5, 1.0)
    cleaned = np.clip(cleaned, 0.0, 1.0)
    params = {
        "method": "nlm",
        "h": h,
        "template_window": template_window,
        "search_window": search_window,
        "used_cv2": _CV2_AVAILABLE,
    }
    return cleaned, params


# ==================================================================
# Public denoiser class
# ==================================================================


def recommend_method(noise_type: Optional[str]) -> str:
    """Return the recommended denoising method for a given noise type."""
    if noise_type is None:
        return "gaussian_blur"
    method = _RECOMMENDED_METHOD.get(noise_type)
    if method is None:
        method = _RECOMMENDED_METHOD.get(noise_type.lower(), "gaussian_blur")
    # Fall back gracefully
    if method == "wavelet" and not _PYWT_AVAILABLE:
        method = "gaussian_blur"
    if method in ("bilateral", "nlm") and not _CV2_AVAILABLE:
        method = "gaussian_blur"
    return method


class ImageDenoiser:
    """
    Stateless image denoiser factory.

    Parameters
    ----------
    method          : one of SUPPORTED_METHODS or "auto"
    kernel_size     : for gaussian_blur / median
    sigma           : for gaussian_blur
    d               : for bilateral
    sigma_color     : for bilateral
    sigma_space     : for bilateral
    wavelet         : wavelet name for wavelet method
    wavelet_level   : decomposition level
    threshold_mode  : "soft" | "hard"
    nlm_h           : filter strength for NLM
    template_window : template patch radius for NLM
    search_window   : search area radius for NLM
    """

    def __init__(
        self,
        method: str = "gaussian_blur",
        *,
        kernel_size: int = 5,
        sigma: float = 1.0,
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0,
        wavelet: str = "db4",
        wavelet_level: Optional[int] = None,
        threshold_mode: str = "soft",
        nlm_h: float = 10.0,
        template_window: int = 7,
        search_window: int = 21,
    ) -> None:
        if method not in SUPPORTED_METHODS and method != "auto":
            raise ValueError(
                f"Unknown method '{method}'. Valid: {SUPPORTED_METHODS + ('auto',)}"
            )
        self.method = method
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.threshold_mode = threshold_mode
        self.nlm_h = nlm_h
        self.template_window = template_window
        self.search_window = search_window

    def denoise(
        self,
        noisy_image: np.ndarray,
        *,
        clean_reference: Optional[np.ndarray] = None,
        noise_type: Optional[str] = None,
    ) -> ImageDenoisingResult:
        """
        Denoise a float64 [0, 1] grayscale image.

        Parameters
        ----------
        noisy_image     : (H, W) float64 [0, 1]
        clean_reference : (H, W) float64 [0, 1] — for quality comparison
        noise_type      : hint for "auto" method selection

        Returns
        -------
        ImageDenoisingResult
        """
        noisy_image = np.asarray(noisy_image, dtype=np.float64)
        if noisy_image.ndim != 2:
            raise ValueError("noisy_image must be 2-D (H, W).")

        method = self.method
        if method == "auto":
            method = recommend_method(noise_type)

        t0 = time.perf_counter()
        warnings_out: list[str] = []

        if method == "gaussian_blur":
            cleaned, params = _apply_gaussian_blur(
                noisy_image, self.kernel_size, self.sigma
            )
        elif method == "median":
            cleaned, params = _apply_median(noisy_image, self.kernel_size)
        elif method == "bilateral":
            cleaned, params = _apply_bilateral(
                noisy_image, self.d, self.sigma_color, self.sigma_space, warnings_out
            )
        elif method == "wavelet":
            cleaned, params = _apply_wavelet(
                noisy_image, self.wavelet, self.wavelet_level, self.threshold_mode, warnings_out
            )
        elif method == "nlm":
            cleaned, params = _apply_nlm(
                noisy_image, self.nlm_h, self.template_window, self.search_window, warnings_out
            )
        else:
            raise RuntimeError(f"Unhandled method: {method}")

        # Quality comparison
        quality_cmp: dict[str, Any] = {}
        psnr_imp: Optional[float] = None
        ssim_imp: Optional[float] = None

        if clean_reference is not None:
            ref = np.asarray(clean_reference, dtype=np.float64)
            quality_cmp = compare_image_quality(ref, noisy_image, cleaned)
            psnr_imp = quality_cmp["improvement"].get("psnr_db")
            ssim_imp = quality_cmp["improvement"].get("ssim")

        return ImageDenoisingResult(
            noisy_image=noisy_image,
            cleaned_image=cleaned,
            method=method,
            method_parameters=params,
            quality_comparison=quality_cmp,
            psnr_improvement=psnr_imp,
            ssim_improvement=ssim_imp,
            processing_time_s=time.perf_counter() - t0,
            warnings=warnings_out,
        )


# ==================================================================
# Convenience function
# ==================================================================


def denoise_image(
    noisy_image: np.ndarray,
    method: str = "gaussian_blur",
    *,
    clean_reference: Optional[np.ndarray] = None,
    noise_type: Optional[str] = None,
    **kwargs: Any,
) -> ImageDenoisingResult:
    """
    One-call convenience wrapper around ImageDenoiser.

    Parameters
    ----------
    noisy_image     : (H, W) float64 [0, 1]
    method          : denoising method name
    clean_reference : optional ground-truth image for quality metrics
    noise_type      : optional hint for "auto" method selection
    **kwargs        : forwarded to ImageDenoiser constructor

    Returns
    -------
    ImageDenoisingResult
    """
    return ImageDenoiser(method=method, **kwargs).denoise(
        noisy_image,
        clean_reference=clean_reference,
        noise_type=noise_type,
    )
