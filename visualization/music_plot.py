"""
Music Plot module.
"""

import numpy as np
import matplotlib.pyplot as plt

from dsp.results import FrequencyResult


def plot_music(result: FrequencyResult):
    if not isinstance(result, FrequencyResult):
        raise TypeError("result must be a FrequencyResult.")

    if result.method != "MUSIC":
        raise ValueError(
            f"Expected a MUSIC result, got {result.method!r}."
        )

    frequencies = result.get("frequency_axis")
    pseudospectrum = result.get("pseudospectrum")

    if frequencies is None or pseudospectrum is None:
        raise ValueError(
            "MUSIC result must contain frequency_axis "
            "and pseudospectrum."
        )

    frequencies = np.asarray(frequencies, dtype=np.float64)
    pseudospectrum = np.asarray(
        pseudospectrum,
        dtype=np.float64,
    )

    if frequencies.ndim != 1 or pseudospectrum.ndim != 1:
        raise ValueError(
            "Frequency and pseudospectrum data must be one-dimensional."
        )

    if len(frequencies) != len(pseudospectrum):
        raise ValueError(
            "Frequency and pseudospectrum arrays must have "
            "the same length."
        )

    if len(frequencies) == 0:
        raise ValueError("MUSIC result cannot be empty.")

    if not np.all(np.isfinite(frequencies)):
        raise ValueError("Frequency data must be finite.")

    if not np.all(np.isfinite(pseudospectrum)):
        raise ValueError(
            "Pseudospectrum data must be finite."
        )

    figure, axis = plt.subplots()

    axis.plot(
        frequencies,
        pseudospectrum,
    )

    # Mark estimated frequencies.
    for frequency in result.frequencies:
        axis.axvline(
            frequency,
            linestyle="--",
            linewidth=1.0,
        )

    axis.set_title("MUSIC Pseudospectrum")
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Normalized Pseudospectrum")
    axis.grid(True)

    figure.tight_layout()

    return figure, axis