"""
Comparison module.
"""

import numpy as np
import matplotlib.pyplot as plt

from dsp.results import DSPResult, FrequencyResult


def plot_frequency_comparison(
    fft_result: DSPResult,
    music_result: FrequencyResult,
    esprit_result: FrequencyResult,
    frequency_range: tuple[float, float] | None = None,
):
    if not isinstance(fft_result, DSPResult):
        raise TypeError("fft_result must be a DSPResult.")

    if not isinstance(music_result, FrequencyResult):
        raise TypeError("music_result must be a FrequencyResult.")

    if not isinstance(esprit_result, FrequencyResult):
        raise TypeError("esprit_result must be a FrequencyResult.")

    if fft_result.method != "FFT":
        raise ValueError(
            f"Expected FFT result, got {fft_result.method!r}."
        )

    if music_result.method != "MUSIC":
        raise ValueError(
            f"Expected MUSIC result, got {music_result.method!r}."
        )

    if esprit_result.method != "ESPRIT":
        raise ValueError(
            f"Expected ESPRIT result, got {esprit_result.method!r}."
        )

    fft_frequencies = fft_result.get("frequency_axis")
    fft_magnitude = fft_result.get("magnitude")

    music_frequencies = music_result.get("frequency_axis")
    music_spectrum = music_result.get("pseudospectrum")

    if fft_frequencies is None or fft_magnitude is None:
        raise ValueError(
            "FFT result must contain frequency_axis and magnitude."
        )

    if music_frequencies is None or music_spectrum is None:
        raise ValueError(
            "MUSIC result must contain frequency_axis "
            "and pseudospectrum."
        )

    fft_frequencies = np.asarray(
        fft_frequencies,
        dtype=np.float64,
    )

    fft_magnitude = np.asarray(
        fft_magnitude,
        dtype=np.float64,
    )

    music_frequencies = np.asarray(
        music_frequencies,
        dtype=np.float64,
    )

    music_spectrum = np.asarray(
        music_spectrum,
        dtype=np.float64,
    )

    esprit_frequencies = np.asarray(
        esprit_result.frequencies,
        dtype=np.float64,
    )

    if fft_frequencies.ndim != 1:
        raise ValueError("FFT frequencies must be one-dimensional.")

    if fft_magnitude.ndim != 1:
        raise ValueError("FFT magnitude must be one-dimensional.")

    if music_frequencies.ndim != 1:
        raise ValueError("MUSIC frequencies must be one-dimensional.")

    if music_spectrum.ndim != 1:
        raise ValueError(
            "MUSIC pseudospectrum must be one-dimensional."
        )

    if esprit_frequencies.ndim != 1:
        raise ValueError(
            "ESPRIT frequencies must be one-dimensional."
        )

    if len(fft_frequencies) != len(fft_magnitude):
        raise ValueError(
            "FFT frequency and magnitude arrays must have "
            "the same length."
        )

    if len(music_frequencies) != len(music_spectrum):
        raise ValueError(
            "MUSIC frequency and pseudospectrum arrays must "
            "have the same length."
        )

    if len(fft_frequencies) == 0:
        raise ValueError("FFT result cannot be empty.")

    if len(music_frequencies) == 0:
        raise ValueError("MUSIC result cannot be empty.")

    if len(esprit_frequencies) == 0:
        raise ValueError("ESPRIT result cannot be empty.")

    if frequency_range is not None:
        if len(frequency_range) != 2:
            raise ValueError(
                "frequency_range must contain (minimum, maximum)."
            )

        minimum, maximum = map(float, frequency_range)

        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError(
                "Frequency range values must be finite."
            )

        if minimum < 0 or maximum <= minimum:
            raise ValueError(
                "Frequency range must satisfy "
                "0 <= minimum < maximum."
            )

        fft_mask = (
            (fft_frequencies >= minimum)
            & (fft_frequencies <= maximum)
        )

        music_mask = (
            (music_frequencies >= minimum)
            & (music_frequencies <= maximum)
        )

        fft_frequencies = fft_frequencies[fft_mask]
        fft_magnitude = fft_magnitude[fft_mask]

        music_frequencies = music_frequencies[music_mask]
        music_spectrum = music_spectrum[music_mask]

        if len(fft_frequencies) == 0:
            raise ValueError(
                "Frequency range contains no FFT data."
            )

        if len(music_frequencies) == 0:
            raise ValueError(
                "Frequency range contains no MUSIC data."
            )

        esprit_frequencies = esprit_frequencies[
            (esprit_frequencies >= minimum)
            & (esprit_frequencies <= maximum)
        ]

        if len(esprit_frequencies) == 0:
            raise ValueError(
                "Frequency range contains no ESPRIT estimates."
            )

    figure, axis = plt.subplots()

    axis.plot(
        fft_frequencies,
        fft_magnitude,
        label="FFT",
    )

    axis.plot(
        music_frequencies,
        music_spectrum,
        label="MUSIC",
    )

    for frequency in esprit_frequencies:
        axis.axvline(
            frequency,
            linestyle="--",
            linewidth=1.0,
            label="ESPRIT" if frequency == esprit_frequencies[0] else None,
        )

    axis.set_title(
        "High-Resolution Frequency Comparison"
    )
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Normalized Magnitude / Spectrum")
    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    return figure, axis