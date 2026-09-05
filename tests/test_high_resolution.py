import numpy as np

from dsp.fft import compute_fft
from dsp.music import estimate_music_frequencies
from dsp.esprit import estimate_esprit_frequencies


def test_fft_music_esprit_same_signal():
    fs = 1000
    t = np.arange(5000) / fs

    rng = np.random.default_rng(42)

    signal = (
        np.sin(2 * np.pi * 50 * t)
        + 0.8 * np.sin(2 * np.pi * 51 * t)
        + 0.005 * rng.standard_normal(len(t))
    )

    # -------------------------
    # FFT
    # -------------------------
    frequencies, magnitude = compute_fft(signal, fs)

    assert len(frequencies) == len(magnitude)
    assert frequencies[0] == 0
    assert frequencies[-1] <= fs / 2

    # -------------------------
    # MUSIC
    # -------------------------
    music_result = estimate_music_frequencies(
        signal,
        sampling_rate=fs,
        model_order=80,
        num_sources=2,
        nfft=16384,
    )

    music_frequencies = music_result["frequencies"]
    print(f"\nMUSIC frequencies: {music_frequencies}")

    assert len(music_frequencies) == 2

    # Both target frequencies must be detected.
    assert np.any(np.abs(music_frequencies - 50.0) < 1.0), (
        f"MUSIC failed to detect 50 Hz. "
        f"Estimated frequencies: {music_frequencies}"
    )

    assert np.any(np.abs(music_frequencies - 51.0) < 1.0), (
        f"MUSIC failed to detect 51 Hz. "
        f"Estimated frequencies: {music_frequencies}"
    )

    # -------------------------
    # ESPRIT
    # -------------------------
    esprit_result = estimate_esprit_frequencies(
        signal,
        sampling_rate=fs,
        model_order=80,
        num_sources=2,
    )

    esprit_frequencies = esprit_result["frequencies"]
    print(f"ESPRIT frequencies: {esprit_frequencies}")
    
    assert len(esprit_frequencies) == 2

    assert np.any(np.abs(esprit_frequencies - 50.0) < 1.0), (
        f"ESPRIT failed to detect 50 Hz. "
        f"Estimated frequencies: {esprit_frequencies}"
    )

    assert np.any(np.abs(esprit_frequencies - 51.0) < 1.0), (
        f"ESPRIT failed to detect 51 Hz. "
        f"Estimated frequencies: {esprit_frequencies}"
    )