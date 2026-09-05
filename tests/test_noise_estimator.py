import numpy as np
import pytest

from intelligence.noise.estimator import estimate_noise


def test_estimate_noise():
    clean = np.ones(1000)
    noise = np.full(1000, 0.1)
    noisy = clean + noise

    result = estimate_noise(
        clean,
        noisy,
    )

    assert np.allclose(
        result["noise"],
        noise,
    )

    assert np.isclose(
        result["signal_power"],
        1.0,
    )

    assert np.isclose(
        result["noise_power"],
        0.01,
    )

    assert np.isclose(
        result["snr_db"],
        20.0,
    )


def test_zero_noise():
    clean = np.sin(
        2 * np.pi * 50 * np.arange(1000) / 1000
    )

    result = estimate_noise(
        clean,
        clean.copy(),
    )

    assert np.isclose(
        result["noise_power"],
        0.0,
    )

    assert np.isinf(
        result["snr_db"]
    )


def test_different_lengths():
    clean = np.ones(100)
    noisy = np.ones(200)

    with pytest.raises(ValueError):
        estimate_noise(clean, noisy)


def test_invalid_clean_signal():
    clean = np.array([1.0, np.nan, 2.0])
    noisy = np.ones(3)

    with pytest.raises(ValueError):
        estimate_noise(clean, noisy)


def test_invalid_noisy_signal():
    clean = np.ones(3)
    noisy = np.array([1.0, np.inf, 2.0])

    with pytest.raises(ValueError):
        estimate_noise(clean, noisy)