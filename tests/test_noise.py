"""
Test Noise module.
"""

import numpy as np

from generators.noise_generator import (
    add_gaussian_noise,
    add_impulse_noise,
    add_periodic_noise,
    add_colored_noise,
    add_mixed_noise,
)


def create_test_signal():
    fs = 1000
    t = np.arange(1000) / fs

    return np.sin(2 * np.pi * 50 * t)


def test_gaussian_noise():

    signal = create_test_signal()

    noisy = add_gaussian_noise(
        signal,
        snr_db=10,
        random_state=42,
    )

    assert noisy.shape == signal.shape
    assert np.isfinite(noisy).all()
    assert not np.array_equal(noisy, signal)


def test_impulse_noise():

    signal = create_test_signal()

    noisy = add_impulse_noise(
        signal,
        probability=0.05,
        random_state=42,
    )

    assert noisy.shape == signal.shape
    assert np.isfinite(noisy).all()
    assert not np.array_equal(noisy, signal)


def test_periodic_noise():

    signal = create_test_signal()

    noisy = add_periodic_noise(
        signal,
        sampling_rate=1000,
        frequency=60,
        amplitude=0.2,
    )

    assert noisy.shape == signal.shape
    assert np.isfinite(noisy).all()


def test_colored_noise():

    signal = create_test_signal()

    noisy = add_colored_noise(
        signal,
        color="pink",
        strength=0.1,
        random_state=42,
    )

    assert noisy.shape == signal.shape
    assert np.isfinite(noisy).all()
    assert not np.array_equal(noisy, signal)


def test_mixed_noise():

    signal = create_test_signal()

    noisy = add_mixed_noise(
        signal,
        sampling_rate=1000,
        snr_db=15,
        periodic_frequency=60,
        random_state=42,
    )

    assert noisy.shape == signal.shape
    assert np.isfinite(noisy).all()
    assert not np.array_equal(noisy, signal)