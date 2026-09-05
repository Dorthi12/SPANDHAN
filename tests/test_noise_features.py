import numpy as np
import pytest

from intelligence.noise.features import extract_noise_features


EXPECTED_FEATURES = {
    "rms",
    "variance",
    "std",
    "kurtosis",
    "skewness",
    "crest_factor",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_flatness",
    "spectral_entropy",
    "spectral_rolloff",
    "mains_band_energy",
    "high_band_energy",
    "snr_db",
}


def test_all_14_features_are_returned():
    rng = np.random.default_rng(42)

    signal = rng.normal(0, 1, 2000)

    result = extract_noise_features(
        signal,
        1000,
    )

    assert set(result.keys()) == EXPECTED_FEATURES


def test_basic_statistical_features():
    signal = np.ones(1000)

    result = extract_noise_features(
        signal,
        1000,
    )

    assert np.isclose(result["rms"], 1.0)
    assert np.isclose(result["variance"], 0.0)
    assert np.isclose(result["std"], 0.0)
    assert np.isclose(result["crest_factor"], 1.0)


def test_snr_is_nan_when_not_provided():
    signal = np.random.default_rng(42).normal(
        size=1000
    )

    result = extract_noise_features(
        signal,
        1000,
    )

    assert np.isnan(result["snr_db"])


def test_snr_can_be_supplied():
    signal = np.random.default_rng(42).normal(
        size=1000
    )

    result = extract_noise_features(
        signal,
        1000,
        snr_db=15.5,
    )

    assert np.isclose(result["snr_db"], 15.5)


def test_periodic_signal_has_mains_band_energy():
    fs = 1000

    time = np.arange(2000) / fs

    signal = np.sin(
        2 * np.pi * 50 * time
    )

    result = extract_noise_features(
        signal,
        fs,
    )

    assert result["mains_band_energy"] > 0


def test_invalid_signal():
    signal = np.array([
        1.0,
        np.nan,
        2.0,
    ])

    with pytest.raises(ValueError):
        extract_noise_features(
            signal,
            1000,
        )


def test_invalid_sampling_rate():
    signal = np.ones(100)

    with pytest.raises(ValueError):
        extract_noise_features(
            signal,
            0,
        )


def test_invalid_snr():
    signal = np.ones(100)

    with pytest.raises(ValueError):
        extract_noise_features(
            signal,
            1000,
            snr_db=np.inf,
        )