import numpy as np
import pytest

from intelligence.noise.features import extract_noise_features


EXPECTED_FEATURES = {
    # Time-domain statistics
    "rms",
    "variance",
    "std",
    "kurtosis",
    "skewness",
    "crest_factor",
    "zero_crossing_rate",
    # Spectral features
    "spectral_centroid",
    "spectral_flatness",
    "spectral_entropy",
    "spectral_rolloff",
    "mains_band_energy",
    "high_band_energy",
    # SNR
    "snr_db",
    # New discriminative features
    "impulse_count",
    "peak_count_rate",
    "energy_ratio_first_half",
    "spectral_variance",
    "low_band_energy",
    "mid_band_energy",
}


def test_all_20_features_are_returned():
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


def test_impulse_count_groups_nearby_threshold_crossings():
    """One multi-sample spike should count as one impulse event."""

    from intelligence.noise.features import extract_noise_features

    signal = np.zeros(1000, dtype=float)

    # Three samples belonging to one impulse.
    signal[100:103] = 10.0

    # Another separated impulse.
    signal[500:503] = -10.0

    features = extract_noise_features(
        signal,
        sampling_rate=1000.0,
    )

    assert features["impulse_count"] == 2.0


def test_impulse_count_zero_for_no_threshold_crossings():
    """A normal low-amplitude signal should have no impulse events."""

    from intelligence.noise.features import extract_noise_features

    signal = np.sin(
        np.linspace(
            0,
            4 * np.pi,
            1000,
        )
    )

    features = extract_noise_features(
        signal,
        sampling_rate=1000.0,
    )

    assert features["impulse_count"] == 0.0