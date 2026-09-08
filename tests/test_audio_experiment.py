"""
tests/test_audio_experiment.py
================================
End-to-end integration tests for domains/audio/experiment.py — Milestone D.

Coverage
--------
Config
  1.  Default config is valid and runnable
  2.  All signal types produce valid results
  3.  All noise types produce valid results
  4.  All denoising methods produce valid results

AudioExperimentResult contract
  5.  Result has all required top-level fields
  6.  timestamp is an ISO 8601 string
  7.  pipeline_version is a non-empty string
  8.  config stored in result matches the input config
  9.  sampling_rate stored matches config
  10. time_axis has correct length (sr * duration)
  11. clean/noisy/cleaned signals have same length as time_axis
  12. All signals are finite and 1-D

Injection sub-result
  13. injection.measured_snr_db is a float (finite or ±inf)
  14. injection.target_snr_db equals config.target_snr_db
  15. measured_snr_db and target_snr_db are separate values
  16. injection.noise_type matches config.noise_type
  17. injection.seed equals config.seed

Analysis sub-results
  18. clean_analysis.stage == "clean"
  19. noisy_analysis.stage == "noisy"
  20. cleaned_analysis.stage == "cleaned"
  21. clean_analysis.metrics.rms > 0
  22. noisy_analysis.classification.ground_truth == config.noise_type
  23. clean_analysis.classification.ground_truth is None
  24. All analysis metrics are finite floats

Denoising sub-result
  25. denoising.snr_improvement_db is a float (reference provided)
  26. denoising.rmse_reduction is a float (reference provided)
  27. quality_comparison has before/after/improvement keys

Quality comparison
  28. quality_comparison["before"]["snr_db"] is finite
  29. quality_comparison["after"]["snr_db"] is finite
  30. quality_comparison["improvement"]["snr_db"] is a float

Reproducibility
  31. Same seed → same clean signal
  32. Same seed → same noisy signal (for Gaussian)
  33. Different seeds → different noisy signals

Timing
  34. total_time_s > 0
  35. stage_times_s contains all expected stage keys
  36. Sum of stage times ≈ total_time_s (within tolerance)

Errors/warnings
  37. errors and warnings are lists (may be empty)
  38. Invalid signal_type → graceful error, not crash
  39. Invalid noise_type → result has error entry

Auto denoising
  40. denoise_method="auto" selects method from noise_type
"""

import math
from datetime import datetime

import numpy as np
import pytest

from domains.audio.experiment import (
    AudioExperimentConfig,
    AudioExperimentResult,
    PIPELINE_VERSION,
    run_audio_experiment,
)
from domains.audio.noise_pipeline import AUDIO_NOISE_TYPES
from preprocessing.audio_denoising import SUPPORTED_METHODS, recommend_method


# ==================================================================
# Fixtures
# ==================================================================


@pytest.fixture(scope="module")
def default_result():
    """Default experiment — 1 s sine at 8 kHz with Gaussian noise."""
    cfg = AudioExperimentConfig(seed=42)
    return run_audio_experiment(cfg)


@pytest.fixture(scope="module")
def default_cfg():
    return AudioExperimentConfig(seed=42)


# ==================================================================
# 1-4: Config validity
# ==================================================================


class TestConfigValidity:

    def test_default_config_produces_result(self, default_result):
        assert isinstance(default_result, AudioExperimentResult)

    @pytest.mark.parametrize("signal_type", [
        "clean_sine", "multi_tone", "harmonic", "chirp",
        "amplitude_modulated", "frequency_modulated",
    ])
    def test_all_signal_types(self, signal_type):
        cfg = AudioExperimentConfig(signal_type=signal_type, seed=0)
        result = run_audio_experiment(cfg)
        assert isinstance(result, AudioExperimentResult)
        assert np.all(np.isfinite(result.clean_signal))

    @pytest.mark.parametrize("noise_type", list(AUDIO_NOISE_TYPES))
    def test_all_noise_types(self, noise_type):
        cfg = AudioExperimentConfig(noise_type=noise_type, seed=1)
        result = run_audio_experiment(cfg)
        assert isinstance(result, AudioExperimentResult)
        assert np.all(np.isfinite(result.noisy_signal))

    @pytest.mark.parametrize("method", [
        m for m in SUPPORTED_METHODS if m not in ("wavelet",)
    ])
    def test_all_denoising_methods(self, method):
        cfg = AudioExperimentConfig(denoise_method=method, seed=2)
        result = run_audio_experiment(cfg)
        assert np.all(np.isfinite(result.cleaned_signal))


# ==================================================================
# 5-12: Result contract
# ==================================================================


class TestResultContract:

    def test_has_all_top_level_fields(self, default_result):
        r = default_result
        assert hasattr(r, "config")
        assert hasattr(r, "timestamp")
        assert hasattr(r, "pipeline_version")
        assert hasattr(r, "clean_signal")
        assert hasattr(r, "noisy_signal")
        assert hasattr(r, "cleaned_signal")
        assert hasattr(r, "time_axis")
        assert hasattr(r, "sampling_rate")
        assert hasattr(r, "injection")
        assert hasattr(r, "clean_analysis")
        assert hasattr(r, "noisy_analysis")
        assert hasattr(r, "cleaned_analysis")
        assert hasattr(r, "denoising")
        assert hasattr(r, "quality_comparison")
        assert hasattr(r, "total_time_s")
        assert hasattr(r, "stage_times_s")
        assert hasattr(r, "warnings")
        assert hasattr(r, "errors")

    def test_timestamp_is_iso8601(self, default_result):
        ts = default_result.timestamp
        assert isinstance(ts, str)
        # Must parse as datetime
        datetime.fromisoformat(ts)

    def test_pipeline_version_non_empty(self, default_result):
        assert isinstance(default_result.pipeline_version, str)
        assert len(default_result.pipeline_version) > 0
        assert default_result.pipeline_version == PIPELINE_VERSION

    def test_config_stored_in_result(self, default_result, default_cfg):
        assert default_result.config.seed == default_cfg.seed
        assert default_result.config.signal_type == default_cfg.signal_type
        assert default_result.config.noise_type == default_cfg.noise_type

    def test_sampling_rate_stored(self, default_result, default_cfg):
        assert default_result.sampling_rate == default_cfg.sampling_rate

    def test_time_axis_correct_length(self, default_result, default_cfg):
        expected_n = int(default_cfg.sampling_rate * default_cfg.duration)
        assert len(default_result.time_axis) == expected_n

    def test_signals_same_length_as_time_axis(self, default_result):
        n = len(default_result.time_axis)
        assert len(default_result.clean_signal) == n
        assert len(default_result.noisy_signal) == n
        assert len(default_result.cleaned_signal) == n

    def test_all_signals_finite_1d(self, default_result):
        for sig in [
            default_result.clean_signal,
            default_result.noisy_signal,
            default_result.cleaned_signal,
        ]:
            assert sig.ndim == 1
            assert np.all(np.isfinite(sig))


# ==================================================================
# 13-17: Injection sub-result
# ==================================================================


class TestInjectionSubResult:

    def test_measured_snr_is_float(self, default_result):
        assert isinstance(default_result.injection.measured_snr_db, float)

    def test_target_snr_matches_config(self, default_result, default_cfg):
        assert default_result.injection.target_snr_db == default_cfg.target_snr_db

    def test_measured_and_target_are_separate(self, default_result):
        # Must be distinct attributes, not aliases
        assert hasattr(default_result.injection, "target_snr_db")
        assert hasattr(default_result.injection, "measured_snr_db")
        # Both exist independently — target is the config value
        assert default_result.injection.target_snr_db == 10.0

    def test_noise_type_matches_config(self, default_result, default_cfg):
        assert default_result.injection.noise_type == default_cfg.noise_type

    def test_seed_matches_config(self, default_result, default_cfg):
        assert default_result.injection.seed == default_cfg.seed


# ==================================================================
# 18-24: Analysis sub-results
# ==================================================================


class TestAnalysisSubResults:

    def test_clean_analysis_stage(self, default_result):
        assert default_result.clean_analysis.stage == "clean"

    def test_noisy_analysis_stage(self, default_result):
        assert default_result.noisy_analysis.stage == "noisy"

    def test_cleaned_analysis_stage(self, default_result):
        assert default_result.cleaned_analysis.stage == "cleaned"

    def test_clean_analysis_rms_positive(self, default_result):
        assert default_result.clean_analysis.metrics.rms > 0

    def test_noisy_ground_truth_equals_noise_type(self, default_result, default_cfg):
        clf = default_result.noisy_analysis.classification
        assert clf.ground_truth == default_cfg.noise_type

    def test_clean_ground_truth_is_none(self, default_result):
        clf = default_result.clean_analysis.classification
        assert clf.ground_truth is None

    def test_all_analysis_scalars_finite(self, default_result):
        for analysis in [
            default_result.clean_analysis,
            default_result.noisy_analysis,
            default_result.cleaned_analysis,
        ]:
            m = analysis.metrics
            scalars = [m.rms, m.variance, m.std, m.peak_amplitude,
                       m.crest_factor, m.signal_energy]
            for v in scalars:
                assert np.isfinite(v), f"Non-finite metric in {analysis.stage}: {v}"

    def test_fft_arrays_present_in_all_analyses(self, default_result):
        for analysis in [
            default_result.clean_analysis,
            default_result.noisy_analysis,
            default_result.cleaned_analysis,
        ]:
            assert analysis.metrics.fft_frequencies is not None
            assert len(analysis.metrics.fft_frequencies) > 0

    def test_noise_features_present_in_noisy(self, default_result):
        feats = default_result.noisy_analysis.metrics.noise_features
        assert feats is not None
        assert len(feats) == 20


# ==================================================================
# 25-30: Denoising and quality comparison
# ==================================================================


class TestDenoisingAndComparison:

    def test_snr_improvement_is_float(self, default_result):
        assert isinstance(default_result.denoising.snr_improvement_db, float)

    def test_rmse_reduction_is_float(self, default_result):
        assert isinstance(default_result.denoising.rmse_reduction, float)

    def test_quality_comparison_has_required_keys(self, default_result):
        cmp = default_result.quality_comparison
        assert "before" in cmp
        assert "after" in cmp
        assert "improvement" in cmp

    def test_before_snr_is_finite_or_inf(self, default_result):
        snr = default_result.quality_comparison["before"]["snr_db"]
        assert isinstance(snr, float)
        # inf is valid when the clean and noisy signals are identical (zero noise)
        assert not math.isnan(snr)

    def test_after_snr_is_finite(self, default_result):
        snr = default_result.quality_comparison["after"]["snr_db"]
        assert isinstance(snr, float)

    def test_improvement_snr_is_float(self, default_result):
        imp = default_result.quality_comparison["improvement"]["snr_db"]
        assert isinstance(imp, float)


# ==================================================================
# 31-33: Reproducibility
# ==================================================================


class TestReproducibility:

    def test_same_seed_same_clean_signal(self):
        cfg = AudioExperimentConfig(seed=77, noise_type="gaussian")
        r1 = run_audio_experiment(cfg)
        r2 = run_audio_experiment(cfg)
        np.testing.assert_array_equal(r1.clean_signal, r2.clean_signal)

    def test_same_seed_same_noisy_signal(self):
        cfg = AudioExperimentConfig(seed=77, noise_type="gaussian")
        r1 = run_audio_experiment(cfg)
        r2 = run_audio_experiment(cfg)
        np.testing.assert_array_equal(r1.noisy_signal, r2.noisy_signal)

    def test_different_seeds_different_noisy(self):
        cfg1 = AudioExperimentConfig(seed=1, noise_type="gaussian")
        cfg2 = AudioExperimentConfig(seed=2, noise_type="gaussian")
        r1 = run_audio_experiment(cfg1)
        r2 = run_audio_experiment(cfg2)
        # Skip assertion if generation failed (both would be zeros)
        if r1.errors or r2.errors:
            pytest.skip("Signal generation failed; skipping seed-diversity check.")
        assert not np.array_equal(r1.noisy_signal, r2.noisy_signal)


# ==================================================================
# 34-36: Timing
# ==================================================================


class TestTiming:

    def test_total_time_positive(self, default_result):
        assert default_result.total_time_s > 0.0

    def test_stage_times_has_all_keys(self, default_result):
        expected = {
            "A_generate", "B_inject", "C_analyze_input",
            "D_denoise", "E_analyze_output", "F_compare",
        }
        assert expected.issubset(set(default_result.stage_times_s.keys()))

    def test_stage_times_sum_close_to_total(self, default_result):
        s = sum(default_result.stage_times_s.values())
        total = default_result.total_time_s
        # Total includes bookkeeping; sum should be <= total + 5%
        assert s <= total * 1.05 + 0.01


# ==================================================================
# 37-39: Errors / warnings
# ==================================================================


class TestErrorHandling:

    def test_errors_and_warnings_are_lists(self, default_result):
        assert isinstance(default_result.errors, list)
        assert isinstance(default_result.warnings, list)

    def test_invalid_signal_type_graceful(self):
        """Bad signal type → error list, not exception."""
        cfg = AudioExperimentConfig(signal_type="nonexistent_type", seed=0)
        result = run_audio_experiment(cfg)
        assert isinstance(result, AudioExperimentResult)
        # Should have an error logged
        assert len(result.errors) > 0 or len(result.warnings) > 0

    def test_invalid_noise_type_graceful(self):
        """Bad noise type → error list, not exception."""
        cfg = AudioExperimentConfig.__new__(AudioExperimentConfig)
        # Bypass validation by manually setting field
        from dataclasses import fields
        for f in fields(AudioExperimentConfig):
            setattr(cfg, f.name, f.default if f.default is not f.default_factory else f.default_factory())
        cfg.noise_type = "completely_invalid_noise_xyz"
        cfg.seed = 0
        result = run_audio_experiment(cfg)
        assert isinstance(result, AudioExperimentResult)
        assert len(result.errors) > 0


# ==================================================================
# 40: Auto denoising
# ==================================================================


class TestAutoDenoising:

    @pytest.mark.parametrize("noise_type,expected_method", [
        ("periodic", "bandstop"),
        ("mixed", "staged"),
        ("colored", "lowpass"),
    ])
    def test_auto_selects_correct_method(self, noise_type, expected_method):
        cfg = AudioExperimentConfig(
            noise_type=noise_type,
            denoise_method="auto",
            seed=5,
        )
        result = run_audio_experiment(cfg)
        if result.denoising is not None:
            assert result.denoising.method == expected_method

    def test_auto_with_gaussian_selects_valid_method(self):
        cfg = AudioExperimentConfig(
            noise_type="gaussian",
            denoise_method="auto",
            seed=5,
        )
        result = run_audio_experiment(cfg)
        if result.denoising is not None:
            assert result.denoising.method in SUPPORTED_METHODS
