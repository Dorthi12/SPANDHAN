"""
Test Dsp module.
"""

import numpy as np

from dsp.fft import compute_fft
from dsp.psd import compute_psd
from dsp.stft import compute_stft
from dsp.correlation import (
    compute_cross_correlation,
    find_correlation_peak,
)


def test_cross_correlation_identical_signals():

    signal_data = np.sin(
        2 * np.pi * 20 * np.arange(1000) / 1000
    )

    lags, correlation = compute_cross_correlation(
        signal_data,
        signal_data,
    )

    peak_lag, peak_value = find_correlation_peak(
        lags,
        correlation,
    )

    assert peak_lag == 0
    assert np.isclose(
        peak_value,
        1.0,
        atol=1e-10,
    )

def test_cross_correlation_detects_shift():

    base = np.zeros(1000)

    base[200:250] = np.hanning(50)

    shifted = np.zeros(1000)

    shifted[225:275] = np.hanning(50)

    lags, correlation = compute_cross_correlation(
        base,
        shifted,
    )

    peak_lag, _ = find_correlation_peak(
        lags,
        correlation,
    )

    assert abs(peak_lag) == 25

    
def test_cross_correlation_output_lengths():

    signal_a = np.ones(100)
    signal_b = np.ones(50)

    # Use non-constant signals because constant signals
    # have zero variance after mean removal.
    signal_a = np.sin(
        2 * np.pi * 10 * np.arange(100) / 100
    )

    signal_b = np.sin(
        2 * np.pi * 10 * np.arange(50) / 100
    )

    lags, correlation = compute_cross_correlation(
        signal_a,
        signal_b,
    )

    assert len(lags) == 149
    assert len(correlation) == 149


def test_cross_correlation_rejects_empty_signal():

    try:
        compute_cross_correlation(
            np.array([]),
            np.ones(10),
        )
        assert False
    except ValueError:
        assert True

def test_stft_detects_sine_frequency():

    sampling_rate = 1000
    frequency = 100

    t = np.arange(5000) / sampling_rate

    signal_data = np.sin(
        2 * np.pi * frequency * t
    )

    frequencies, times, magnitude = compute_stft(
        signal_data,
        sampling_rate=sampling_rate,
        nperseg=500,
        noverlap=250,
    )

    assert magnitude.shape == (
        len(frequencies),
        len(times),
    )

    peak_frequency_index = np.argmax(
        np.mean(magnitude, axis=1)
    )

    detected_frequency = frequencies[
        peak_frequency_index
    ]

    assert np.isclose(
        detected_frequency,
        frequency,
        atol=2.0,
    )


def test_stft_returns_valid_output():

    signal_data = np.random.default_rng(
        42
    ).normal(
        size=2000
    )

    frequencies, times, magnitude = compute_stft(
        signal_data,
        sampling_rate=1000,
        nperseg=256,
    )

    assert len(frequencies) > 0
    assert len(times) > 0

    assert magnitude.ndim == 2

    assert np.isfinite(frequencies).all()
    assert np.isfinite(times).all()
    assert np.isfinite(magnitude).all()


def test_stft_rejects_invalid_overlap():

    signal_data = np.ones(1000)

    try:
        compute_stft(
            signal_data,
            sampling_rate=1000,
            nperseg=256,
            noverlap=256,
        )
        assert False
    except ValueError:
        assert True


def test_stft_rejects_invalid_sampling_rate():

    signal_data = np.ones(1000)

    try:
        compute_stft(
            signal_data,
            sampling_rate=0,
        )
        assert False
    except ValueError:
        assert True

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