import numpy as np
import pytest

from intelligence.noise.analyzer import analyze_noise


def test_analyze_noise_basic_metrics():
    noise = np.ones(1000)
    result = analyze_noise(noise, 1000)

    assert np.isclose(result["mean"], 1.0)
    assert np.isclose(result["rms"], 1.0)
    assert np.isclose(result["variance"], 0.0)
    assert np.isclose(result["std"], 0.0)
    assert np.isclose(result["peak"], 1.0)
    assert np.isclose(result["peak_to_peak"], 0.0)
    assert np.isclose(result["crest_factor"], 1.0)


def test_zero_crossing_rate():
    noise = np.array([1.0, -1.0, 1.0, -1.0, 1.0])

    result = analyze_noise(noise, 1000)

    assert np.isclose(result["zero_crossing_rate"], 1.0)


def test_periodic_noise_dominant_frequency():
    sampling_rate = 1000
    duration = 2.0

    time = np.arange(
        0,
        duration,
        1 / sampling_rate,
    )

    noise = np.sin(2 * np.pi * 50 * time)

    result = analyze_noise(noise, sampling_rate)

    assert abs(result["dominant_frequency"] - 50.0) <= 1.0


def test_spectral_metrics_exist():
    noise = np.random.default_rng(42).normal(
        0,
        1,
        2000,
    )

    result = analyze_noise(noise, 1000)

    assert result["frequency_axis"].ndim == 1
    assert result["psd"].ndim == 1
    assert len(result["frequency_axis"]) == len(result["psd"])
    assert result["spectral_power"] >= 0.0


def test_invalid_noise():
    noise = np.array([1.0, np.nan, 2.0])

    with pytest.raises(ValueError):
        analyze_noise(noise, 1000)


def test_invalid_sampling_rate():
    noise = np.ones(100)

    with pytest.raises(ValueError):
        analyze_noise(noise, 0)

    with pytest.raises(ValueError):
        analyze_noise(noise, -1000)


def test_empty_noise():
    with pytest.raises(ValueError):
        analyze_noise(np.array([]), 1000)