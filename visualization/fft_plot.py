"""
Fft Plot module.
"""

import numpy as np
import matplotlib.pyplot as plt

from dsp.results import DSPResult


def plot_fft(
    result: DSPResult,
    max_frequency: float | None = None,
):
    if not isinstance(result, DSPResult):
        raise TypeError("result must be a DSPResult.")

    if result.method != "FFT":
        raise ValueError(
            f"Expected an FFT result, got {result.method!r}."
        )

    frequencies = result.get("frequency_axis")
    magnitude = result.get("magnitude")

    if frequencies is None or magnitude is None:
        raise ValueError(
            "FFT result must contain frequency_axis and magnitude."
        )

    frequencies = np.asarray(frequencies, dtype=np.float64)
    magnitude = np.asarray(magnitude, dtype=np.float64)

    if frequencies.ndim != 1 or magnitude.ndim != 1:
        raise ValueError("Frequency and magnitude data must be one-dimensional.")

    if len(frequencies) != len(magnitude):
        raise ValueError(
            "Frequency and magnitude arrays must have the same length."
        )

    if len(frequencies) == 0:
        raise ValueError("FFT result cannot be empty.")

    if not np.all(np.isfinite(frequencies)):
        raise ValueError("Frequency data must be finite.")

    if not np.all(np.isfinite(magnitude)):
        raise ValueError("Magnitude data must be finite.")

    if max_frequency is not None:
        max_frequency = float(max_frequency)

        if not np.isfinite(max_frequency) or max_frequency <= 0:
            raise ValueError(
                "max_frequency must be greater than zero."
            )

        mask = frequencies <= max_frequency

        if not np.any(mask):
            raise ValueError(
                "max_frequency is below the available frequency range."
            )

        frequencies = frequencies[mask]
        magnitude = magnitude[mask]

    figure, axis = plt.subplots()

    axis.plot(frequencies, magnitude)

    axis.set_title("FFT Magnitude Spectrum")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Magnitude")
    axis.grid(True)

    figure.tight_layout()

    return figure, axis