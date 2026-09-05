import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from dsp.fft import compute_fft
from dsp.results_adapters import fft_to_result
from dsp.results import DSPResult
from visualization.fft_plot import plot_fft
from dsp.music import estimate_music_frequencies
from dsp.results_adapters import music_to_result
from dsp.esprit import estimate_esprit_frequencies
from dsp.results_adapters import esprit_to_result
from dsp.results_adapters import (
    fft_to_result,
    music_to_result,
    esprit_to_result,
)


def create_comparison_results():
    fs = 1000
    t = np.arange(5000) / fs

    rng = np.random.default_rng(42)

    signal = (
        np.sin(2 * np.pi * 50 * t)
        + 0.8 * np.sin(2 * np.pi * 51 * t)
        + 0.005 * rng.standard_normal(len(t))
    )

    fft_frequencies, fft_magnitude = compute_fft(
        signal,
        fs,
    )

    fft_result = fft_to_result(
        fft_frequencies,
        fft_magnitude,
        fs,
    )

    music_raw = estimate_music_frequencies(
        signal,
        sampling_rate=fs,
        model_order=80,
        num_sources=2,
        nfft=16384,
    )

    music_result = music_to_result(music_raw)

    esprit_raw = estimate_esprit_frequencies(
        signal,
        sampling_rate=fs,
        model_order=80,
        num_sources=2,
    )

    esprit_result = esprit_to_result(esprit_raw)

    return (
        fft_result,
        music_result,
        esprit_result,
    )


def test_plot_frequency_comparison():
    from visualization.comparison import plot_frequency_comparison

    fft_result, music_result, esprit_result = (
        create_comparison_results()
    )

    figure, axis = plot_frequency_comparison(
        fft_result,
        music_result,
        esprit_result,
        frequency_range=(45, 55),
    )

    assert figure is not None
    assert axis is not None

    assert axis.get_xlabel() == "Frequency (Hz)"

    assert len(axis.lines) >= 3

    figure.clf()


def test_comparison_contains_esprit_estimates():
    from visualization.comparison import plot_frequency_comparison

    fft_result, music_result, esprit_result = (
        create_comparison_results()
    )

    figure, axis = plot_frequency_comparison(
        fft_result,
        music_result,
        esprit_result,
        frequency_range=(45, 55),
    )

    vertical_lines = [
        line
        for line in axis.lines
        if line.get_linestyle() == "--"
    ]

    assert len(vertical_lines) == len(
        esprit_result.frequencies
    )

    figure.clf()


def test_comparison_rejects_invalid_frequency_range():
    from visualization.comparison import plot_frequency_comparison

    fft_result, music_result, esprit_result = (
        create_comparison_results()
    )

    with pytest.raises(ValueError):
        plot_frequency_comparison(
            fft_result,
            music_result,
            esprit_result,
            frequency_range=(55, 45),
        )


def test_comparison_rejects_wrong_fft_method():
    from visualization.comparison import plot_frequency_comparison

    fft_result, music_result, esprit_result = (
        create_comparison_results()
    )

    fft_result.method = "MUSIC"

    with pytest.raises(ValueError):
        plot_frequency_comparison(
            fft_result,
            music_result,
            esprit_result,
        )

def create_esprit_result():
    fs = 1000
    t = np.arange(2000) / fs

    signal = (
        np.sin(2 * np.pi * 50 * t)
        + 0.8 * np.sin(2 * np.pi * 120 * t)
    )

    raw_result = estimate_esprit_frequencies(
        signal,
        sampling_rate=fs,
        model_order=30,
        num_sources=2,
    )

    return esprit_to_result(raw_result)


def test_plot_esprit_returns_figure():
    from visualization.esprit_plot import plot_esprit

    result = create_esprit_result()

    figure, axis = plot_esprit(result)

    assert figure is not None
    assert axis is not None
    assert axis.get_xlabel() == "Estimated Component"
    assert axis.get_ylabel() == "Frequency (Hz)"

    figure.clf()


def test_plot_esprit_contains_estimates():
    from visualization.esprit_plot import plot_esprit

    result = create_esprit_result()

    figure, axis = plot_esprit(result)

    assert len(axis.lines) >= len(result.frequencies)

    figure.clf()


def test_plot_esprit_frequency_limit():
    from visualization.esprit_plot import plot_esprit

    result = create_esprit_result()

    figure, axis = plot_esprit(
        result,
        max_frequency=100,
    )

    y_data = axis.lines[0].get_ydata()

    assert np.max(y_data) <= 100

    figure.clf()


def test_plot_esprit_rejects_wrong_method():
    from visualization.esprit_plot import plot_esprit

    result = create_esprit_result()
    result.method = "MUSIC"

    with pytest.raises(ValueError):
        plot_esprit(result)


def test_plot_esprit_rejects_empty_result():
    from visualization.esprit_plot import plot_esprit
    from dsp.results import FrequencyResult

    result = FrequencyResult(
        method="ESPRIT",
        frequencies=[],
    )

    with pytest.raises(ValueError):
        plot_esprit(result)

def create_music_result():
    fs = 1000
    t = np.arange(2000) / fs

    signal = (
        np.sin(2 * np.pi * 50 * t)
        + 0.8 * np.sin(2 * np.pi * 120 * t)
    )

    raw_result = estimate_music_frequencies(
        signal,
        sampling_rate=fs,
        model_order=30,
        num_sources=2,
    )

    return music_to_result(raw_result)


def test_plot_music_returns_figure():
    from visualization.music_plot import plot_music

    result = create_music_result()

    figure, axis = plot_music(result)

    assert figure is not None
    assert axis is not None
    assert axis.get_xlabel() == "Frequency (Hz)"
    assert axis.get_ylabel() == "Normalized Pseudospectrum"

    figure.clf()


def test_plot_music_contains_spectrum():
    from visualization.music_plot import plot_music

    result = create_music_result()

    figure, axis = plot_music(result)

    assert len(axis.lines) >= 1

    figure.clf()


def test_plot_music_marks_estimated_frequencies():
    from visualization.music_plot import plot_music

    result = create_music_result()

    figure, axis = plot_music(result)

    # One spectrum line + one vertical line per estimate.
    assert len(axis.lines) >= 1 + len(result.frequencies)

    figure.clf()


def test_plot_music_rejects_wrong_method():
    from visualization.music_plot import plot_music

    result = create_music_result()
    result.method = "FFT"

    with pytest.raises(ValueError):
        plot_music(result)


def test_plot_music_rejects_missing_data():
    from visualization.music_plot import plot_music
    from dsp.results import FrequencyResult

    result = FrequencyResult(
        method="MUSIC",
        frequencies=[50.0, 120.0],
    )

    with pytest.raises(ValueError):
        plot_music(result)

def create_fft_result():
    fs = 1000
    t = np.arange(1000) / fs

    signal = np.sin(2 * np.pi * 50 * t)

    frequencies, magnitude = compute_fft(
        signal,
        fs,
    )

    return fft_to_result(
        frequencies,
        magnitude,
        fs,
    )


def test_plot_fft_returns_figure():
    result = create_fft_result()

    figure, axis = plot_fft(result)

    assert figure is not None
    assert axis is not None
    assert axis.get_xlabel() == "Frequency (Hz)"
    assert axis.get_ylabel() == "Magnitude"

    figure.clf()


def test_plot_fft_frequency_limit():
    result = create_fft_result()

    figure, axis = plot_fft(
        result,
        max_frequency=100,
    )

    x_data = axis.lines[0].get_xdata()

    assert np.max(x_data) <= 100

    figure.clf()


def test_plot_fft_rejects_wrong_result_type():
    result = DSPResult(
        method="MUSIC",
    )

    with pytest.raises(ValueError):
        plot_fft(result)


def test_plot_fft_rejects_missing_data():
    result = DSPResult(
        method="FFT",
    )

    with pytest.raises(ValueError):
        plot_fft(result)


def test_plot_fft_rejects_invalid_frequency_limit():
    result = create_fft_result()

    with pytest.raises(ValueError):
        plot_fft(
            result,
            max_frequency=0,
        )