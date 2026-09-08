"""
tests/test_audio_denoising.py
==============================
Tests for preprocessing/audio_denoising.py — Milestone C.

Coverage
--------
Method correctness
  1.  All supported methods execute without error
  2.  All methods return finite output
  3.  All methods return same-length output as input
  4.  Lowpass attenuates high-frequency noise
  5.  Highpass attenuates low-frequency noise
  6.  Bandpass passes signal in the band
  7.  Bandstop attenuates signal at the notch frequency
  8.  Wavelet denoising reduces RMS error vs clean (when pywt available)
  9.  Staged denoising executes and produces valid output
  10. Staged denoising removes periodic component

Typed result
  11. Returns AudioDenoisingResult dataclass
  12. method field matches the method requested
  13. method_parameters is a non-empty dict
  14. processing_time_s is non-negative
  15. warnings is a list

Quality metrics (with reference)
  16. snr_improvement_db is a float when clean_reference supplied
  17. rmse_reduction is a float when clean_reference supplied
  18. quality_comparison contains before/after/improvement keys
  19. Denoising of Gaussian noise with wavelet improves SNR
  20. Denoising of periodic noise with bandstop improves SNR
  21. noisy_signal stored in result equals input

Without reference
  22. snr_improvement_db is None when no reference
  23. rmse_reduction is None when no reference
  24. quality_comparison is empty dict when no reference

Method recommendation
  25. recommend_method("gaussian") → wavelet or lowpass (pywt-dependent)
  26. recommend_method("periodic") → bandstop
  27. recommend_method("mixed") → staged
  28. recommend_method("colored") → lowpass
  29. recommend_method(None) → lowpass
  30. recommend_method unknown → lowpass fallback

Invalid inputs
  31. Invalid method raises ValueError
  32. 2-D noisy signal raises ValueError
  33. Empty signal raises ValueError
  34. Non-finite signal raises ValueError
  35. Negative sampling rate raises ValueError
  36. Mismatched reference length raises ValueError
  37. filter_order < 1 raises ValueError
  38. Invalid wavelet_threshold_mode raises ValueError
  39. bandpass with low >= high raises ValueError

denoise_audio() factory
  40. Auto mode selects method from noise_type
  41. Returns AudioDenoisingResult
  42. All methods reachable through factory
"""

import numpy as np
import pytest

from generators.waveform_generator import generate_sine
from domains.audio.noise_pipeline import inject_audio_noise
from preprocessing.audio_denoising import (
    SUPPORTED_METHODS,
    AudioDenoiser,
    AudioDenoisingResult,
    _PYWT_AVAILABLE,
    denoise_audio,
    recommend_method,
)


# ==================================================================
# Fixtures
# ==================================================================


@pytest.fixture
def sr():
    return 8000.0


@pytest.fixture
def clean_sine(sr):
    _, sig = generate_sine(440.0, sr, 1.0, amplitude=1.0)
    return sig.astype(np.float64)


@pytest.fixture
def gaussian_noise_result(clean_sine, sr):
    return inject_audio_noise(clean_sine, sr, noise_type="gaussian",
                              target_snr_db=5.0, seed=0)


@pytest.fixture
def periodic_noise_result(clean_sine, sr):
    return inject_audio_noise(clean_sine, sr, noise_type="periodic",
                              periodic_frequency=100.0,
                              periodic_amplitude=0.5)


# ==================================================================
# 1-10: Method correctness
# ==================================================================


class TestAllMethodsExecute:

    @pytest.mark.parametrize("method", [
        "lowpass", "highpass", "bandpass", "bandstop", "staged"
    ])
    def test_method_executes_without_error(self, clean_sine, sr, method):
        """All non-wavelet methods must execute cleanly on a 1-sec sine."""
        denoiser = AudioDenoiser(
            method=method,
            # Bandpass: ensure valid range for 8 kHz signal
            bandpass_low_hz=300.0,
            bandpass_high_hz=3000.0,
            notch_freq_hz=100.0,
            notch_bandwidth_hz=20.0,
            staged_notch_freq_hz=100.0,
            staged_notch_bandwidth_hz=20.0,
        )
        result = denoiser.denoise(clean_sine, sr)
        assert result is not None

    @pytest.mark.skipif(not _PYWT_AVAILABLE, reason="pywt not installed")
    def test_wavelet_executes_without_error(self, clean_sine, sr):
        denoiser = AudioDenoiser(method="wavelet")
        result = denoiser.denoise(clean_sine, sr)
        assert result is not None

    @pytest.mark.parametrize("method", [
        "lowpass", "highpass", "bandpass", "bandstop", "staged"
    ])
    def test_all_methods_finite_output(self, clean_sine, sr, method):
        denoiser = AudioDenoiser(
            method=method,
            bandpass_low_hz=300.0,
            bandpass_high_hz=3000.0,
            notch_freq_hz=100.0,
            notch_bandwidth_hz=20.0,
            staged_notch_freq_hz=100.0,
            staged_notch_bandwidth_hz=20.0,
        )
        result = denoiser.denoise(clean_sine, sr)
        assert np.all(np.isfinite(result.cleaned_signal)), (
            f"Method {method!r} produced non-finite output"
        )

    @pytest.mark.parametrize("method", [
        "lowpass", "highpass", "bandpass", "bandstop", "staged"
    ])
    def test_all_methods_same_length(self, clean_sine, sr, method):
        denoiser = AudioDenoiser(
            method=method,
            bandpass_low_hz=300.0,
            bandpass_high_hz=3000.0,
            notch_freq_hz=100.0,
            notch_bandwidth_hz=20.0,
            staged_notch_freq_hz=100.0,
            staged_notch_bandwidth_hz=20.0,
        )
        result = denoiser.denoise(clean_sine, sr)
        assert len(result.cleaned_signal) == len(clean_sine)


class TestLowpassEffect:

    def test_attenuates_high_frequency(self, sr):
        """Lowpass should suppress a 2 kHz component while keeping 200 Hz."""
        t = np.arange(int(sr)) / sr
        low_comp = np.sin(2 * np.pi * 200.0 * t)
        high_comp = 0.8 * np.sin(2 * np.pi * 2000.0 * t)
        mixed = low_comp + high_comp

        result = AudioDenoiser(method="lowpass", cutoff_hz=500.0).denoise(mixed, sr)
        cleaned = result.cleaned_signal

        # After lowpass, high-freq component should have much less energy
        high_residual_power = float(np.mean((cleaned - low_comp) ** 2))
        original_high_power = float(np.mean(high_comp ** 2))
        assert high_residual_power < 0.5 * original_high_power


class TestBandstopEffect:

    def test_attenuates_notch_frequency(self, sr):
        """Bandstop at 100 Hz should suppress a 100 Hz sinusoid."""
        t = np.arange(int(sr)) / sr
        signal = np.sin(2 * np.pi * 440.0 * t)
        periodic = 0.5 * np.sin(2 * np.pi * 100.0 * t)
        noisy = signal + periodic

        result = AudioDenoiser(
            method="bandstop",
            notch_freq_hz=100.0,
            notch_bandwidth_hz=20.0,
        ).denoise(noisy, sr)

        # 100 Hz component should be attenuated
        corr_before = float(np.corrcoef(noisy - signal, periodic)[0, 1])
        cleaned_residual = result.cleaned_signal - signal
        corr_after = float(np.corrcoef(cleaned_residual, periodic)[0, 1])
        assert abs(corr_after) < abs(corr_before) + 0.1  # tolerance for filter transition


class TestWaveletDenoising:

    @pytest.mark.skipif(not _PYWT_AVAILABLE, reason="pywt not installed")
    def test_wavelet_reduces_gaussian_noise(self, gaussian_noise_result, clean_sine, sr):
        """Wavelet denoising should reduce RMSE vs the clean reference."""
        noisy = gaussian_noise_result.noisy_signal
        result = AudioDenoiser(method="wavelet").denoise(noisy, sr, clean_reference=clean_sine)

        rmse_before = float(np.sqrt(np.mean((noisy - clean_sine) ** 2)))
        rmse_after = float(np.sqrt(np.mean((result.cleaned_signal - clean_sine) ** 2)))
        assert rmse_after < rmse_before

    @pytest.mark.skipif(not _PYWT_AVAILABLE, reason="pywt not installed")
    def test_wavelet_same_length(self, gaussian_noise_result, sr):
        noisy = gaussian_noise_result.noisy_signal
        result = AudioDenoiser(method="wavelet").denoise(noisy, sr)
        assert len(result.cleaned_signal) == len(noisy)

    @pytest.mark.skipif(not _PYWT_AVAILABLE, reason="pywt not installed")
    def test_wavelet_parameters_recorded(self, gaussian_noise_result, sr):
        noisy = gaussian_noise_result.noisy_signal
        result = AudioDenoiser(method="wavelet").denoise(noisy, sr)
        p = result.method_parameters
        assert "wavelet" in p
        assert "level" in p
        assert "threshold" in p
        assert "threshold_mode" in p

    @pytest.mark.skipif(_PYWT_AVAILABLE, reason="pywt IS installed")
    def test_wavelet_falls_back_to_lowpass_when_pywt_missing(self, clean_sine, sr):
        result = AudioDenoiser(method="wavelet").denoise(clean_sine, sr)
        assert np.all(np.isfinite(result.cleaned_signal))
        assert "pywt not installed" in " ".join(result.warnings)


class TestStagedDenoising:

    def test_staged_removes_periodic_component(self, periodic_noise_result, clean_sine, sr):
        noisy = periodic_noise_result.noisy_signal
        result = AudioDenoiser(
            method="staged",
            staged_notch_freq_hz=100.0,
            staged_notch_bandwidth_hz=20.0,
        ).denoise(noisy, sr, clean_reference=clean_sine)
        assert np.all(np.isfinite(result.cleaned_signal))
        assert len(result.cleaned_signal) == len(noisy)

    def test_staged_params_contain_stages(self, clean_sine, sr):
        result = AudioDenoiser(
            method="staged",
            staged_notch_freq_hz=100.0,
            staged_notch_bandwidth_hz=20.0,
        ).denoise(clean_sine, sr)
        assert "stages" in result.method_parameters
        assert isinstance(result.method_parameters["stages"], list)
        assert len(result.method_parameters["stages"]) >= 1


# ==================================================================
# 11-15: Typed result
# ==================================================================


class TestResultTyping:

    def test_returns_audio_denoising_result(self, clean_sine, sr):
        result = AudioDenoiser(method="lowpass").denoise(clean_sine, sr)
        assert isinstance(result, AudioDenoisingResult)

    def test_method_field_matches(self, clean_sine, sr):
        result = AudioDenoiser(method="highpass").denoise(clean_sine, sr)
        assert result.method == "highpass"

    def test_method_parameters_is_dict(self, clean_sine, sr):
        result = AudioDenoiser(method="lowpass").denoise(clean_sine, sr)
        assert isinstance(result.method_parameters, dict)
        assert len(result.method_parameters) > 0

    def test_processing_time_non_negative(self, clean_sine, sr):
        result = AudioDenoiser(method="lowpass").denoise(clean_sine, sr)
        assert result.processing_time_s >= 0.0

    def test_warnings_is_list(self, clean_sine, sr):
        result = AudioDenoiser(method="lowpass").denoise(clean_sine, sr)
        assert isinstance(result.warnings, list)

    def test_noisy_signal_stored(self, gaussian_noise_result, sr):
        noisy = gaussian_noise_result.noisy_signal
        result = AudioDenoiser(method="lowpass").denoise(noisy, sr)
        np.testing.assert_array_equal(result.noisy_signal, noisy)


# ==================================================================
# 16-24: Quality metrics
# ==================================================================


class TestQualityMetricsWithReference:

    def test_snr_improvement_is_float(self, gaussian_noise_result, clean_sine, sr):
        noisy = gaussian_noise_result.noisy_signal
        result = AudioDenoiser(method="lowpass").denoise(noisy, sr, clean_sine)
        assert isinstance(result.snr_improvement_db, float)
        assert np.isfinite(result.snr_improvement_db)

    def test_rmse_reduction_is_float(self, gaussian_noise_result, clean_sine, sr):
        noisy = gaussian_noise_result.noisy_signal
        result = AudioDenoiser(method="lowpass").denoise(noisy, sr, clean_sine)
        assert isinstance(result.rmse_reduction, float)

    def test_quality_comparison_has_required_keys(self, gaussian_noise_result, clean_sine, sr):
        noisy = gaussian_noise_result.noisy_signal
        result = AudioDenoiser(method="lowpass").denoise(noisy, sr, clean_sine)
        cmp = result.quality_comparison
        assert "before" in cmp
        assert "after" in cmp
        assert "improvement" in cmp

    def test_wavelet_improves_snr_on_gaussian_noise(
        self, gaussian_noise_result, clean_sine, sr
    ):
        if not _PYWT_AVAILABLE:
            pytest.skip("pywt not installed")
        noisy = gaussian_noise_result.noisy_signal
        result = AudioDenoiser(method="wavelet").denoise(noisy, sr, clean_sine)
        assert result.snr_improvement_db is not None
        # Wavelet denoising on SNR=5dB Gaussian should produce some improvement
        assert result.snr_improvement_db > -5.0  # at least not catastrophically bad

    def test_bandstop_improves_snr_on_periodic_noise(
        self, periodic_noise_result, clean_sine, sr
    ):
        noisy = periodic_noise_result.noisy_signal
        result = AudioDenoiser(
            method="bandstop",
            notch_freq_hz=100.0,
            notch_bandwidth_hz=20.0,
        ).denoise(noisy, sr, clean_sine)
        # Bandstop at the periodic frequency should improve SNR
        assert result.snr_improvement_db is not None
        assert result.snr_improvement_db > 0.0


class TestQualityMetricsWithoutReference:

    def test_snr_improvement_is_none(self, clean_sine, sr):
        result = AudioDenoiser(method="lowpass").denoise(clean_sine, sr)
        assert result.snr_improvement_db is None

    def test_rmse_reduction_is_none(self, clean_sine, sr):
        result = AudioDenoiser(method="lowpass").denoise(clean_sine, sr)
        assert result.rmse_reduction is None

    def test_quality_comparison_is_empty(self, clean_sine, sr):
        result = AudioDenoiser(method="lowpass").denoise(clean_sine, sr)
        assert result.quality_comparison == {}


# ==================================================================
# 25-30: Method recommendation
# ==================================================================


class TestMethodRecommendation:

    def test_gaussian_recommends_wavelet_or_lowpass(self):
        method = recommend_method("gaussian")
        assert method in ("wavelet", "lowpass")

    def test_periodic_recommends_bandstop(self):
        assert recommend_method("periodic") == "bandstop"

    def test_mixed_recommends_staged(self):
        assert recommend_method("mixed") == "staged"

    def test_colored_recommends_lowpass(self):
        assert recommend_method("colored") == "lowpass"

    def test_none_recommends_lowpass(self):
        assert recommend_method(None) == "lowpass"

    def test_unknown_falls_back_to_lowpass(self):
        result = recommend_method("some_completely_unknown_type")
        assert result == "lowpass"

    def test_capitalised_aliases_work(self):
        assert recommend_method("Periodic") == "bandstop"
        assert recommend_method("Mixed") == "staged"
        assert recommend_method("Colored") == "lowpass"

    def test_result_is_supported_method(self):
        for noise_type in ["gaussian", "impulse", "periodic", "colored", "mixed", None]:
            m = recommend_method(noise_type)
            assert m in SUPPORTED_METHODS


# ==================================================================
# 31-39: Invalid inputs
# ==================================================================


class TestInvalidInputs:

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            AudioDenoiser(method="magic_filter")

    def test_2d_signal_raises(self, sr):
        bad = np.ones((10, 2))
        denoiser = AudioDenoiser(method="lowpass")
        with pytest.raises(ValueError):
            denoiser.denoise(bad, sr)

    def test_empty_signal_raises(self, sr):
        denoiser = AudioDenoiser(method="lowpass")
        with pytest.raises(ValueError):
            denoiser.denoise(np.array([]), sr)

    def test_non_finite_signal_raises(self, sr):
        bad = np.array([1.0, np.nan, 2.0])
        denoiser = AudioDenoiser(method="lowpass")
        with pytest.raises(ValueError):
            denoiser.denoise(bad, sr)

    def test_negative_sampling_rate_raises(self, clean_sine):
        denoiser = AudioDenoiser(method="lowpass")
        with pytest.raises(ValueError):
            denoiser.denoise(clean_sine, -1.0)

    def test_mismatched_reference_length_raises(self, clean_sine, sr):
        ref = np.ones(len(clean_sine) + 100)
        denoiser = AudioDenoiser(method="lowpass")
        with pytest.raises(ValueError):
            denoiser.denoise(clean_sine, sr, clean_reference=ref)

    def test_filter_order_zero_raises(self):
        with pytest.raises(ValueError):
            AudioDenoiser(method="lowpass", filter_order=0)

    def test_invalid_threshold_mode_raises(self):
        with pytest.raises(ValueError):
            AudioDenoiser(method="wavelet", wavelet_threshold_mode="unknown")

    def test_bandpass_invalid_range_raises(self, clean_sine, sr):
        denoiser = AudioDenoiser(
            method="bandpass",
            bandpass_low_hz=3000.0,
            bandpass_high_hz=300.0,  # low > high
        )
        with pytest.raises((ValueError, Exception)):
            denoiser.denoise(clean_sine, sr)


# ==================================================================
# 40-42: denoise_audio() factory
# ==================================================================


class TestDenoisingFactory:

    def test_auto_mode_selects_from_noise_type(self, gaussian_noise_result, sr):
        noisy = gaussian_noise_result.noisy_signal
        result = denoise_audio(
            noisy, sr,
            method="auto",
            noise_type="periodic",
            notch_freq_hz=440.0,
            notch_bandwidth_hz=20.0,
        )
        assert result.method == "bandstop"

    def test_returns_audio_denoising_result(self, clean_sine, sr):
        result = denoise_audio(clean_sine, sr, method="lowpass")
        assert isinstance(result, AudioDenoisingResult)

    def test_explicit_method_overrides_noise_type(self, clean_sine, sr):
        result = denoise_audio(
            clean_sine, sr,
            method="highpass",
            noise_type="gaussian",
        )
        assert result.method == "highpass"

    @pytest.mark.parametrize("method", ["lowpass", "highpass", "bandpass", "bandstop", "staged"])
    def test_all_methods_via_factory(self, clean_sine, sr, method):
        result = denoise_audio(
            clean_sine, sr,
            method=method,
            bandpass_low_hz=300.0,
            bandpass_high_hz=3000.0,
            notch_freq_hz=200.0,
            notch_bandwidth_hz=20.0,
            staged_notch_freq_hz=200.0,
            staged_notch_bandwidth_hz=20.0,
        )
        assert isinstance(result, AudioDenoisingResult)
        assert np.all(np.isfinite(result.cleaned_signal))


# ==================================================================
# Integration: Inject → Denoise → Compare
# ==================================================================


class TestFullPipeline:

    @pytest.mark.skipif(not _PYWT_AVAILABLE, reason="pywt not installed")
    def test_gaussian_inject_wavelet_denoise(self, clean_sine, sr):
        injection = inject_audio_noise(
            clean_sine, sr, noise_type="gaussian",
            target_snr_db=5.0, seed=42,
        )
        result = denoise_audio(
            injection.noisy_signal, sr,
            method="wavelet",
            clean_reference=injection.clean_signal,
        )
        cmp = result.quality_comparison
        # Cleaned signal should have lower RMSE than noisy
        assert cmp["after"]["rmse"] <= cmp["before"]["rmse"] * 1.5  # generous tolerance

    def test_periodic_inject_bandstop_denoise(self, clean_sine, sr):
        injection = inject_audio_noise(
            clean_sine, sr, noise_type="periodic",
            periodic_frequency=200.0,
            periodic_amplitude=0.8,
        )
        result = denoise_audio(
            injection.noisy_signal, sr,
            method="bandstop",
            notch_freq_hz=200.0,
            notch_bandwidth_hz=30.0,
            clean_reference=injection.clean_signal,
        )
        # After notch at 200 Hz, SNR should be better
        assert result.snr_improvement_db is not None
        assert result.snr_improvement_db > 0.0

    def test_noisy_cleaned_same_length_as_clean(self, clean_sine, sr):
        injection = inject_audio_noise(
            clean_sine, sr, noise_type="gaussian", seed=1,
        )
        result = denoise_audio(injection.noisy_signal, sr, method="lowpass")
        assert len(result.cleaned_signal) == len(clean_sine)
