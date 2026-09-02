"""
Test Dsp module.
"""

import numpy as np

from dsp.fft import compute_fft
from dsp.psd import compute_psd


def test_psd_detects_sine_frequency():

    sampling_rate = 1000
    frequency = 100

    t = np.arange(2000) / sampling_rate

    signal_data = np.sin(
        2 * np.pi * frequency * t
    )

    frequencies, psd = compute_psd(
        signal_data,
        sampling_rate,
        nperseg=1000,
    )

    peak_index = np.argmax(psd)

    detected_frequency = frequencies[
        peak_index
    ]

    assert np.isclose(
        detected_frequency,
        frequency,
        atol=2.0,
    )


def test_psd_returns_valid_arrays():

    signal_data = np.random.default_rng(
        42
    ).normal(
        size=1000
    )

    frequencies, psd = compute_psd(
        signal_data,
        sampling_rate=1000,
    )

    assert len(frequencies) == len(psd)
    assert len(frequencies) > 0
    assert np.isfinite(frequencies).all()
    assert np.isfinite(psd).all()


def test_psd_is_nonnegative():

    signal_data = np.random.default_rng(
        42
    ).normal(
        size=1000
    )

    _, psd = compute_psd(
        signal_data,
        sampling_rate=1000,
    )

    assert np.all(psd >= 0)


def test_psd_rejects_invalid_sampling_rate():

    signal_data = np.ones(100)

    try:
        compute_psd(
            signal_data,
            sampling_rate=0,
        )
        assert False
    except ValueError:
        assert True
        
def test_fft_detects_sine_frequency():

    sampling_rate = 1000
    duration = 1.0
    frequency = 100

    t = np.arange(
        int(sampling_rate * duration)
    ) / sampling_rate

    signal = np.sin(
        2 * np.pi * frequency * t
    )

    frequencies, magnitude = compute_fft(
        signal,
        sampling_rate,
    )

    peak_index = np.argmax(magnitude)

    detected_frequency = frequencies[
        peak_index
    ]

    assert np.isclose(
        detected_frequency,
        frequency,
        atol=1.0,
    )


def test_fft_returns_one_sided_spectrum():

    signal = np.ones(1000)

    frequencies, magnitude = compute_fft(
        signal,
        sampling_rate=1000,
    )

    assert len(frequencies) == 501
    assert len(magnitude) == 501


def test_fft_frequency_range():

    signal = np.zeros(1000)

    frequencies, _ = compute_fft(
        signal,
        sampling_rate=1000,
    )

    assert np.isclose(
        frequencies[0],
        0.0,
    )

    assert np.isclose(
        frequencies[-1],
        500.0,
    )


def test_fft_rejects_invalid_sampling_rate():

    signal = np.ones(100)

    try:
        compute_fft(
            signal,
            sampling_rate=0,
        )
        assert False
    except ValueError:
        assert True