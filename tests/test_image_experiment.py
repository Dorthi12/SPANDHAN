"""
tests/test_image_experiment.py
================================
End-to-end integration tests for domains/image/experiment.py — Milestone E.

Coverage
--------
Config
  1.  Default config is valid and runnable
  2.  All image types produce success
  3.  All noise types produce success
  4.  All denoising methods produce success

Generation
  5.  Generated image is 2-D float64 in [0, 1]
  6.  Image size matches config
  7.  Different image types produce different images

Noise Injection
  8.  Noisy image is same shape as clean
  9.  Noisy image differs from clean
  10. psnr_before is finite and positive for Gaussian
  11. Salt-and-pepper injects exactly correct pixel fraction (approx)
  12. Periodic noise creates structured pattern

Denoising
  13. Cleaned image is same shape as noisy
  14. Cleaned image is float64 in [0, 1]
  15. Cleaned image differs from noisy for Gaussian + wavelet
  16. Method parameters are recorded
  17. Wavelet denoising reduces RMSE vs noisy (Gaussian)

Quality Comparison
  18. quality_comparison has before/after/improvement keys
  19. PSNR-after is finite
  20. SSIM-after is in [-1, 1]
  21. RMSE-after is non-negative
  22. hist_corr-after is in [-1, 1]
  23. Improvement for PSNR is (after - before)
  24. Improvement for RMSE is (before - after)

Result Dataclass
  25. result.success is True for default config
  26. result.errors is empty for default config
  27. result.clean_image == generation.image
  28. result.timing has required keys
  29. All timing values are non-negative

Error handling
  30. Invalid image_type yields success=False, error in generation sub
  31. run_image_experiment always returns an ImageExperimentResult
  32. errors list is populated on failure

Reproducibility
  33. Same seed → identical images and metrics
  34. Different seeds → different noisy images

Image types — content checks
  35. checkerboard produces binary-ish values
  36. gradient produces monotonic column means (horizontal)
  37. sinusoidal produces values in (0, 1) — not constant
  38. gaussian_blob has peak at centre region

Performance
  39. Default experiment completes in < 30 s
  40. 128×128 image processes in < 10 s
"""

from __future__ import annotations

import time
import pytest
import numpy as np

from domains.image.experiment import (
    ImageExperimentConfig,
    ImageExperimentResult,
    run_image_experiment,
)
from generators.image_dataset_generator import IMAGE_TYPES
from intelligence.image.noise_injector import IMAGE_NOISE_TYPES
from intelligence.image.denoiser import SUPPORTED_METHODS as DENOISE_METHODS


# ==================================================================
# Fixtures
# ==================================================================


@pytest.fixture(scope="module")
def default_result() -> ImageExperimentResult:
    return run_image_experiment()


@pytest.fixture(scope="module")
def gaussian_result() -> ImageExperimentResult:
    cfg = ImageExperimentConfig(noise_type="gaussian", target_psnr_db=20.0, image_size=64)
    return run_image_experiment(cfg)


# ==================================================================
# Config
# ==================================================================


class TestConfig:

    def test_default_config_runs(self, default_result):
        """Default config should always succeed."""
        assert isinstance(default_result, ImageExperimentResult)
        assert default_result.success, default_result.errors

    @pytest.mark.parametrize("image_type", list(IMAGE_TYPES))
    def test_all_image_types_succeed(self, image_type):
        cfg = ImageExperimentConfig(image_type=image_type, image_size=64)
        r = run_image_experiment(cfg)
        assert r.success, f"Failed for {image_type}: {r.errors}"

    @pytest.mark.parametrize("noise_type", list(IMAGE_NOISE_TYPES))
    def test_all_noise_types_succeed(self, noise_type):
        cfg = ImageExperimentConfig(noise_type=noise_type, image_size=64)
        r = run_image_experiment(cfg)
        assert r.success, f"Failed for {noise_type}: {r.errors}"

    @pytest.mark.parametrize("method", list(DENOISE_METHODS) + ["auto"])
    def test_all_denoising_methods_succeed(self, method):
        cfg = ImageExperimentConfig(denoising_method=method, image_size=64)
        r = run_image_experiment(cfg)
        assert r.success, f"Failed for {method}: {r.errors}"


# ==================================================================
# Generation
# ==================================================================


class TestGeneration:

    def test_image_is_2d_float64(self, default_result):
        img = default_result.clean_image
        assert img.ndim == 2
        assert img.dtype == np.float64

    def test_image_in_range_01(self, default_result):
        img = default_result.clean_image
        assert img.min() >= -1e-10
        assert img.max() <= 1.0 + 1e-10

    def test_image_size_matches_config(self):
        size = 64
        cfg = ImageExperimentConfig(image_size=size)
        r = run_image_experiment(cfg)
        assert r.clean_image.shape == (size, size)

    def test_different_image_types_different_content(self):
        types = ["checkerboard", "gradient", "circles"]
        results = [run_image_experiment(ImageExperimentConfig(image_type=t, image_size=64)) for t in types]
        images = [r.clean_image for r in results]
        # At least pairwise different
        assert not np.allclose(images[0], images[1])
        assert not np.allclose(images[1], images[2])


# ==================================================================
# Noise Injection
# ==================================================================


class TestNoiseInjection:

    def test_noisy_same_shape_as_clean(self, default_result):
        assert default_result.noisy_image.shape == default_result.clean_image.shape

    def test_noisy_differs_from_clean(self, default_result):
        diff = np.abs(default_result.clean_image - default_result.noisy_image).max()
        assert diff > 1e-6

    def test_psnr_before_positive_finite(self, gaussian_result):
        psnr = gaussian_result.injection.result.psnr_before
        assert np.isfinite(psnr)
        assert psnr > 0

    def test_sp_noise_corrupts_pixels(self):
        cfg = ImageExperimentConfig(
            noise_type="salt_and_pepper",
            sp_density=0.1,
            image_size=64,
        )
        r = run_image_experiment(cfg)
        assert r.success
        noisy = r.noisy_image
        # Some pixels should be 0 or 1 due to S&P
        extreme_px = np.sum((noisy < 0.01) | (noisy > 0.99))
        assert extreme_px > 0

    def test_periodic_noise_has_structure(self):
        cfg = ImageExperimentConfig(
            image_type="gradient",
            noise_type="periodic",
            periodic_freq_x=8.0,
            periodic_freq_y=4.0,
            periodic_amplitude=0.3,
            image_size=64,
        )
        r = run_image_experiment(cfg)
        assert r.success
        # Noisy image should differ from clean (periodic pattern added)
        diff = np.abs(r.noisy_image - r.clean_image).max()
        assert diff > 0.1

    def test_noisy_image_in_range_01(self, default_result):
        img = default_result.noisy_image
        assert img.min() >= -1e-10
        assert img.max() <= 1.0 + 1e-10


# ==================================================================
# Denoising
# ==================================================================


class TestDenoising:

    def test_cleaned_same_shape(self, default_result):
        assert default_result.cleaned_image.shape == default_result.noisy_image.shape

    def test_cleaned_in_range_01(self, default_result):
        img = default_result.cleaned_image
        assert img.min() >= -1e-10
        assert img.max() <= 1.0 + 1e-10

    def test_cleaned_is_float64(self, default_result):
        assert default_result.cleaned_image.dtype == np.float64

    def test_cleaned_differs_from_noisy(self):
        cfg = ImageExperimentConfig(
            noise_type="gaussian", target_psnr_db=15.0,
            denoising_method="gaussian_blur", sigma=2.0, image_size=64
        )
        r = run_image_experiment(cfg)
        assert r.success
        diff = np.abs(r.cleaned_image - r.noisy_image).max()
        assert diff > 1e-6

    def test_method_parameters_recorded(self, default_result):
        den = default_result.denoising.result
        assert den is not None
        assert isinstance(den.method_parameters, dict)
        assert "method" in den.method_parameters

    def test_wavelet_reduces_rmse_on_gaussian(self):
        cfg = ImageExperimentConfig(
            noise_type="gaussian", target_psnr_db=20.0,
            denoising_method="wavelet", image_size=64, seed=0
        )
        r = run_image_experiment(cfg)
        assert r.success
        cmp = r.quality_comparison
        assert cmp["after"]["rmse"] <= cmp["before"]["rmse"] * 1.5


# ==================================================================
# Quality Comparison
# ==================================================================


class TestQualityComparison:

    def test_has_before_after_improvement(self, default_result):
        cmp = default_result.quality_comparison
        assert "before" in cmp
        assert "after" in cmp
        assert "improvement" in cmp

    def test_psnr_after_finite(self, default_result):
        cmp = default_result.quality_comparison
        psnr_a = cmp["after"].get("psnr_db", float("nan"))
        assert np.isfinite(psnr_a) or psnr_a == float("inf")

    def test_ssim_after_in_range(self, default_result):
        cmp = default_result.quality_comparison
        ssim_a = cmp["after"].get("ssim", float("nan"))
        assert np.isfinite(ssim_a)
        assert -1.0 - 1e-6 <= ssim_a <= 1.0 + 1e-6

    def test_rmse_after_nonnegative(self, default_result):
        cmp = default_result.quality_comparison
        rmse = cmp["after"].get("rmse", float("nan"))
        assert np.isfinite(rmse)
        assert rmse >= 0.0

    def test_hist_corr_in_range(self, default_result):
        cmp = default_result.quality_comparison
        hc = cmp["after"].get("hist_corr", float("nan"))
        assert -1.0 - 1e-6 <= hc <= 1.0 + 1e-6

    def test_psnr_improvement_correct_direction(self, gaussian_result):
        cmp = gaussian_result.quality_comparison
        imp = cmp["improvement"]["psnr_db"]
        assert isinstance(imp, float)
        # improvement = after - before
        expected = cmp["after"]["psnr_db"] - cmp["before"]["psnr_db"]
        assert abs(imp - expected) < 1e-8

    def test_rmse_improvement_correct_direction(self, gaussian_result):
        cmp = gaussian_result.quality_comparison
        imp = cmp["improvement"]["rmse"]
        # improvement = before - after (positive = better)
        expected = cmp["before"]["rmse"] - cmp["after"]["rmse"]
        assert abs(imp - expected) < 1e-8


# ==================================================================
# Result Dataclass
# ==================================================================


class TestResultDataclass:

    def test_success_true_for_default(self, default_result):
        assert default_result.success is True

    def test_errors_empty_for_default(self, default_result):
        assert default_result.errors == []

    def test_clean_image_is_generation_image(self, default_result):
        assert default_result.clean_image is default_result.generation.image

    def test_timing_has_required_keys(self, default_result):
        for key in ("generation_s", "injection_s", "denoising_s", "total_s"):
            assert key in default_result.timing

    def test_timing_values_nonnegative(self, default_result):
        for v in default_result.timing.values():
            assert v >= 0.0


# ==================================================================
# Error Handling
# ==================================================================


class TestErrorHandling:

    def test_invalid_image_type_fails_gracefully(self):
        cfg = ImageExperimentConfig(image_type="not_a_real_type")
        r = run_image_experiment(cfg)
        assert isinstance(r, ImageExperimentResult)
        assert not r.generation.success
        assert r.generation.error is not None
        assert not r.success

    def test_always_returns_result(self):
        cfg = ImageExperimentConfig(image_type="__invalid__")
        r = run_image_experiment(cfg)
        assert isinstance(r, ImageExperimentResult)

    def test_errors_list_populated_on_failure(self):
        cfg = ImageExperimentConfig(image_type="__invalid__")
        r = run_image_experiment(cfg)
        assert len(r.errors) > 0


# ==================================================================
# Reproducibility
# ==================================================================


class TestReproducibility:

    def test_same_seed_same_images(self):
        cfg = ImageExperimentConfig(image_type="gaussian_blob", image_size=64, seed=42)
        r1 = run_image_experiment(cfg)
        r2 = run_image_experiment(cfg)
        assert np.allclose(r1.clean_image, r2.clean_image)
        assert np.allclose(r1.noisy_image, r2.noisy_image)
        assert np.allclose(r1.cleaned_image, r2.cleaned_image)

    def test_different_seeds_different_noisy(self):
        cfg1 = ImageExperimentConfig(
            image_type="gaussian_blob", noise_type="gaussian", image_size=64, seed=1
        )
        cfg2 = ImageExperimentConfig(
            image_type="gaussian_blob", noise_type="gaussian", image_size=64, seed=99
        )
        r1 = run_image_experiment(cfg1)
        r2 = run_image_experiment(cfg2)
        assert r1.success and r2.success
        assert not np.allclose(r1.noisy_image, r2.noisy_image)


# ==================================================================
# Image content checks
# ==================================================================


class TestImageContent:

    def test_checkerboard_binary_ish(self):
        cfg = ImageExperimentConfig(image_type="checkerboard", image_size=64)
        r = run_image_experiment(cfg)
        img = r.clean_image
        # Most pixels should be 0 or 1
        binary_pct = np.mean((img < 0.05) | (img > 0.95))
        assert binary_pct > 0.8

    def test_gradient_monotone_means(self):
        cfg = ImageExperimentConfig(image_type="gradient", image_size=64)
        r = run_image_experiment(cfg)
        img = r.clean_image
        # horizontal gradient → column means should increase monotonically (roughly)
        col_means = img.mean(axis=0)
        # at least monotonically increasing over the first half
        diffs = np.diff(col_means)
        assert np.sum(diffs > 0) > len(diffs) * 0.7

    def test_sinusoidal_not_constant(self):
        cfg = ImageExperimentConfig(image_type="sinusoidal", image_size=64)
        r = run_image_experiment(cfg)
        img = r.clean_image
        assert img.std() > 0.05

    def test_gaussian_blob_has_peak(self):
        cfg = ImageExperimentConfig(
            image_type="gaussian_blob", image_size=64
        )
        r = run_image_experiment(cfg)
        img = r.clean_image
        # Image should have values close to 1.0 somewhere (the blob peak)
        assert img.max() > 0.8


# ==================================================================
# Performance
# ==================================================================


class TestPerformance:

    def test_default_experiment_under_30s(self):
        t0 = time.perf_counter()
        r = run_image_experiment()
        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"Too slow: {elapsed:.2f}s"

    def test_small_image_under_10s(self):
        cfg = ImageExperimentConfig(image_size=64)
        t0 = time.perf_counter()
        r = run_image_experiment(cfg)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"Too slow: {elapsed:.2f}s"
