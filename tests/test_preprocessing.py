"""
Test Preprocessing module.
"""

import numpy as np

from preprocessing.dc_removal import remove_dc
from preprocessing.detrending import detrend_signal
from preprocessing.normalization import normalize_signal
from preprocessing.resampling import resample_signal

def test_resample_signal_downsampling():

    signal = np.sin(
        2 * np.pi * 100 * np.arange(1000) / 1000
    )

    processed = resample_signal(
        signal,
        original_rate=1000,
        target_rate=500,
    )

    assert len(processed) == 500
    assert np.isfinite(processed).all()


def test_resample_signal_upsampling():

    signal = np.sin(
        2 * np.pi * 100 * np.arange(500) / 500
    )

    processed = resample_signal(
        signal,
        original_rate=500,
        target_rate=1000,
    )

    assert len(processed) == 1000
    assert np.isfinite(processed).all()


def test_resample_same_rate():

    signal = np.arange(
        100,
        dtype=np.float64,
    )

    processed = resample_signal(
        signal,
        original_rate=1000,
        target_rate=1000,
    )

    assert np.array_equal(processed, signal)
    
def test_peak_normalization():

    signal = np.array([
        -2.0,
        1.0,
        4.0,
        -3.0,
    ])

    processed = normalize_signal(
        signal,
        method="peak",
    )

    assert np.isclose(
        np.max(np.abs(processed)),
        1.0,
    )

    assert processed.shape == signal.shape
    assert np.isfinite(processed).all()


def test_zscore_normalization():

    signal = np.arange(
        1.0,
        101.0,
    )

    processed = normalize_signal(
        signal,
        method="zscore",
    )

    assert np.isclose(
        np.mean(processed),
        0.0,
        atol=1e-10,
    )

    assert np.isclose(
        np.std(processed),
        1.0,
        atol=1e-10,
    )


def test_normalization_rejects_empty():

    signal = np.array([])

    try:
        normalize_signal(signal)
        assert False
    except ValueError:
        assert True

def test_remove_dc():

    signal = np.array([
        3.0,
        4.0,
        5.0,
        6.0,
    ])

    processed = remove_dc(signal)

    assert np.isclose(np.mean(processed), 0.0)

    assert processed.shape == signal.shape

    assert np.isfinite(processed).all()


def test_remove_dc_preserves_shape():

    signal = np.linspace(
        1.0,
        10.0,
        1000,
    )

    processed = remove_dc(signal)

    assert processed.shape == signal.shape


def test_remove_dc_rejects_empty_signal():

    signal = np.array([])

    try:
        remove_dc(signal)
        assert False
    except ValueError:
        assert True

def test_detrend_signal():

    x = np.linspace(0.0, 1.0, 1000)

    trend = 5.0 * x
    oscillation = np.sin(2 * np.pi * 10 * x)

    signal = trend + oscillation

    processed = detrend_signal(signal)

    assert processed.shape == signal.shape
    assert np.isfinite(processed).all()

    # The linear trend should be substantially reduced.
    fitted_slope = np.polyfit(x, processed, 1)[0]

    assert abs(fitted_slope) < 0.1


def test_detrend_signal_preserves_samples():

    signal = np.arange(500, dtype=np.float64)

    processed = detrend_signal(signal)

    assert len(processed) == len(signal)


def test_detrend_signal_rejects_empty():

    signal = np.array([])

    try:
        detrend_signal(signal)
        assert False
    except ValueError:
        assert True