"""
Test Generators module.
"""

import numpy as np

from generators.waveform_generator import (
    generate_sine,
    generate_multi_tone,
    generate_square,
    generate_sawtooth,
    generate_impulse,
)


def test_sine_generator():

    t, signal = generate_sine(
        frequency=100,
        sampling_rate=1000,
        duration=1.0,
    )

    assert len(t) == 1000
    assert len(signal) == 1000
    assert np.isfinite(signal).all()


def test_multi_tone_generator():

    t, signal = generate_multi_tone(
        frequencies=[50, 100],
        sampling_rate=1000,
        duration=1.0,
    )

    assert len(signal) == 1000
    assert np.isfinite(signal).all()


def test_square_generator():

    t, signal = generate_square(
        frequency=10,
        sampling_rate=1000,
        duration=1.0,
    )

    assert len(signal) == 1000
    assert set(np.unique(signal)).issubset({-1.0, 1.0})


def test_sawtooth_generator():

    t, signal = generate_sawtooth(
        frequency=10,
        sampling_rate=1000,
        duration=1.0,
    )

    assert len(signal) == 1000
    assert np.isfinite(signal).all()


def test_impulse_generator():

    t, signal = generate_impulse(
        sampling_rate=1000,
        duration=1.0,
        position=100,
        amplitude=5.0,
    )

    assert len(signal) == 1000
    assert signal[100] == 5.0
    assert np.count_nonzero(signal) == 1