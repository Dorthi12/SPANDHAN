import numpy as np

from dsp.fft import compute_fft
from dsp.music import estimate_music_frequencies
from dsp.esprit import estimate_esprit_frequencies
from dsp.results_adapters import (
    fft_to_result,
    music_to_result,
    esprit_to_result,
)
from dsp.results import DSPResult, FrequencyResult


def test_fft_adapter():
    fs = 1000
    t = np.arange(1000) / fs
    signal = np.sin(2 * np.pi * 50 * t)

    frequencies, magnitude = compute_fft(signal, fs)

    result = fft_to_result(
        frequencies,
        magnitude,
        fs,
    )

    assert isinstance(result, DSPResult)
    assert result.method == "FFT"
    assert result.parameters["sampling_rate"] == fs
    assert np.array_equal(
        result.get("frequency_axis"),
        frequencies,
    )


def test_music_adapter():
    fs = 1000
    t = np.arange(2000) / fs
    signal = (
        np.sin(2 * np.pi * 50 * t)
        + 0.8 * np.sin(2 * np.pi * 120 * t)
    )

    raw_result = estimate_music_frequencies(
        signal,
        fs,
        model_order=30,
        num_sources=2,
    )

    result = music_to_result(raw_result)

    assert isinstance(result, FrequencyResult)
    assert result.method == "MUSIC"
    assert len(result.frequencies) == 2
    assert result.parameters["model_order"] == 30


def test_esprit_adapter():
    fs = 1000
    t = np.arange(2000) / fs
    signal = (
        np.sin(2 * np.pi * 50 * t)
        + 0.8 * np.sin(2 * np.pi * 120 * t)
    )

    raw_result = estimate_esprit_frequencies(
        signal,
        fs,
        model_order=30,
        num_sources=2,
    )

    result = esprit_to_result(raw_result)

    assert isinstance(result, FrequencyResult)
    assert result.method == "ESPRIT"
    assert len(result.frequencies) == 2
    assert result.parameters["num_sources"] == 2