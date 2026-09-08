"""
domains/image/experiment.py
==============================
Image Synthetic Experiment Orchestrator — Milestone E.

Pipeline
--------
Stage A: Generate clean synthetic image (SyntheticImageDataset)
Stage B: Inject noise (inject_image_noise)
Stage C: Denoise (ImageDenoiser)
Stage D: Compare quality (compare_image_quality)
Stage E: Build result

All computation is pure Python/NumPy — no Qt here.
The Qt frontend (image_experiment_page.py) calls run_image_experiment()
in a Worker thread and receives ImageExperimentResult.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from generators.image_dataset_generator import (
    IMAGE_TYPES,
    ImageDatasetConfig,
    SyntheticImageDataset,
)
from intelligence.image.noise_injector import (
    IMAGE_NOISE_TYPES,
    ImageNoiseInjectionResult,
    inject_image_noise,
)
from intelligence.image.denoiser import (
    SUPPORTED_METHODS as DENOISING_METHODS,
    ImageDenoisingResult,
    ImageDenoiser,
    recommend_method,
)
from intelligence.image.comparison import compare_image_quality


# ==================================================================
# Config
# ==================================================================


@dataclass
class ImageExperimentConfig:
    """
    Fully-specified configuration for one image experiment run.
    All fields are plain Python scalars — JSON-serialisable.
    """

    # Dataset generation
    image_type: str = "checkerboard"   # must be in IMAGE_TYPES
    image_size: int = 128              # both height and width

    # Noise injection
    noise_type: str = "gaussian"       # must be in IMAGE_NOISE_TYPES
    target_psnr_db: float = 25.0       # for gaussian
    sp_density: float = 0.05           # for salt_and_pepper
    speckle_variance: float = 0.04     # for speckle
    periodic_freq_x: float = 5.0       # for periodic
    periodic_freq_y: float = 3.0       # for periodic
    periodic_amplitude: float = 0.2    # for periodic
    uniform_amplitude: float = 0.15    # for uniform

    # Denoising
    denoising_method: str = "auto"     # DENOISING_METHODS + "auto"
    kernel_size: int = 5
    sigma: float = 1.0
    wavelet: str = "db4"
    wavelet_level: Optional[int] = None
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    nlm_h: float = 10.0

    # Reproducibility
    seed: int = 42


# ==================================================================
# Sub-result types (simple wrappers with error field)
# ==================================================================


@dataclass
class _SubResult:
    success: bool = False
    error: Optional[str] = None
    timing_s: float = 0.0


@dataclass
class GenerationSubResult(_SubResult):
    image: Optional[np.ndarray] = None
    image_type: str = ""
    height: int = 0
    width: int = 0


@dataclass
class InjectionSubResult(_SubResult):
    result: Optional[ImageNoiseInjectionResult] = None


@dataclass
class DenoisingSubResult(_SubResult):
    result: Optional[ImageDenoisingResult] = None


# ==================================================================
# Full experiment result
# ==================================================================


@dataclass
class ImageExperimentResult:
    """
    Complete, typed result from run_image_experiment().

    The frontend renders this object directly — no dict parsing.
    """

    config: ImageExperimentConfig

    generation: GenerationSubResult = field(default_factory=GenerationSubResult)
    injection: InjectionSubResult = field(default_factory=InjectionSubResult)
    denoising: DenoisingSubResult = field(default_factory=DenoisingSubResult)

    # Convenience references (set after all stages succeed)
    clean_image: Optional[np.ndarray] = None
    noisy_image: Optional[np.ndarray] = None
    cleaned_image: Optional[np.ndarray] = None

    quality_comparison: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return (
            self.generation.success
            and self.injection.success
            and self.denoising.success
        )

    @property
    def errors(self) -> list[str]:
        errs = []
        for sub in (self.generation, self.injection, self.denoising):
            if sub.error:
                errs.append(sub.error)
        return errs


# ==================================================================
# Orchestrator
# ==================================================================


# Map generator image_type names (aliases users might type)
_IMAGE_TYPE_ALIASES: dict[str, str] = {t: t for t in IMAGE_TYPES}
# No aliases needed currently; all match directly.


def run_image_experiment(
    config: Optional[ImageExperimentConfig] = None,
) -> ImageExperimentResult:
    """
    Run the full image experiment pipeline.

    Parameters
    ----------
    config : ImageExperimentConfig or None (uses defaults)

    Returns
    -------
    ImageExperimentResult
        Always returns a result even on partial failure.
        Check result.success and result.errors.
    """
    cfg = config or ImageExperimentConfig()
    result = ImageExperimentResult(config=cfg)
    total_t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Stage A: Generate clean synthetic image
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        image_type = _IMAGE_TYPE_ALIASES.get(cfg.image_type, cfg.image_type)
        if image_type not in IMAGE_TYPES:
            raise ValueError(
                f"Unknown image_type '{cfg.image_type}'. Valid: {IMAGE_TYPES}"
            )
        dataset_cfg = ImageDatasetConfig(
            image_types=[image_type],
            samples_per_type=1,
            height=cfg.image_size,
            width=cfg.image_size,
            seed=cfg.seed,
        )
        ds = SyntheticImageDataset(dataset_cfg)
        samples = ds.generate()
        sample = samples[0]
        clean_img = sample.image
        gen_t = time.perf_counter() - t0
        result.generation = GenerationSubResult(
            success=True,
            image=clean_img,
            image_type=image_type,
            height=sample.meta.height,
            width=sample.meta.width,
            timing_s=gen_t,
        )
        result.clean_image = clean_img
    except Exception as exc:
        result.generation = GenerationSubResult(
            success=False,
            error=f"Stage A (generate): {exc}\n{traceback.format_exc()}",
            timing_s=time.perf_counter() - t0,
        )
        result.timing["total_s"] = time.perf_counter() - total_t0
        return result

    # ------------------------------------------------------------------
    # Stage B: Inject noise
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        inj = inject_image_noise(
            clean_img,
            noise_type=cfg.noise_type,
            target_psnr_db=cfg.target_psnr_db,
            density=cfg.sp_density,
            variance=cfg.speckle_variance,
            freq_x=cfg.periodic_freq_x,
            freq_y=cfg.periodic_freq_y,
            periodic_amplitude=cfg.periodic_amplitude,
            uniform_amplitude=cfg.uniform_amplitude,
            seed=cfg.seed,
        )
        inj_t = time.perf_counter() - t0
        result.injection = InjectionSubResult(
            success=True, result=inj, timing_s=inj_t
        )
        result.noisy_image = inj.noisy_image
    except Exception as exc:
        result.injection = InjectionSubResult(
            success=False,
            error=f"Stage B (inject): {exc}\n{traceback.format_exc()}",
            timing_s=time.perf_counter() - t0,
        )
        result.timing["total_s"] = time.perf_counter() - total_t0
        return result

    # ------------------------------------------------------------------
    # Stage C: Denoise
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        method = cfg.denoising_method
        if method == "auto":
            method = recommend_method(cfg.noise_type)

        den_result = ImageDenoiser(
            method=method,
            kernel_size=cfg.kernel_size,
            sigma=cfg.sigma,
            wavelet=cfg.wavelet,
            wavelet_level=cfg.wavelet_level,
            d=cfg.bilateral_d,
            sigma_color=cfg.bilateral_sigma_color,
            sigma_space=cfg.bilateral_sigma_space,
            nlm_h=cfg.nlm_h,
        ).denoise(
            inj.noisy_image,
            clean_reference=clean_img,
            noise_type=cfg.noise_type,
        )
        den_t = time.perf_counter() - t0
        result.denoising = DenoisingSubResult(
            success=True, result=den_result, timing_s=den_t
        )
        result.cleaned_image = den_result.cleaned_image
    except Exception as exc:
        result.denoising = DenoisingSubResult(
            success=False,
            error=f"Stage C (denoise): {exc}\n{traceback.format_exc()}",
            timing_s=time.perf_counter() - t0,
        )
        result.timing["total_s"] = time.perf_counter() - total_t0
        return result

    # ------------------------------------------------------------------
    # Stage D: Quality comparison
    # ------------------------------------------------------------------
    try:
        result.quality_comparison = compare_image_quality(
            clean_img, inj.noisy_image, den_result.cleaned_image
        )
    except Exception as exc:
        result.quality_comparison = {"error": str(exc)}

    # ------------------------------------------------------------------
    # Timings
    # ------------------------------------------------------------------
    result.timing = {
        "generation_s": result.generation.timing_s,
        "injection_s": result.injection.timing_s,
        "denoising_s": result.denoising.timing_s,
        "total_s": time.perf_counter() - total_t0,
    }

    return result
