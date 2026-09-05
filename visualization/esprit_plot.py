import numpy as np
import matplotlib.pyplot as plt

from dsp.results import FrequencyResult


def plot_esprit(
    result: FrequencyResult,
    max_frequency: float | None = None,
):
    if not isinstance(result, FrequencyResult):
        raise TypeError("result must be a FrequencyResult.")

    if result.method != "ESPRIT":
        raise ValueError(
            f"Expected an ESPRIT result, got {result.method!r}."
        )

    frequencies = np.asarray(
        result.frequencies,
        dtype=np.float64,
    )

    if frequencies.ndim != 1:
        raise ValueError(
            "ESPRIT frequencies must be one-dimensional."
        )

    if frequencies.size == 0:
        raise ValueError(
            "ESPRIT result contains no estimated frequencies."
        )

    if not np.all(np.isfinite(frequencies)):
        raise ValueError(
            "ESPRIT frequencies must be finite."
        )

    if np.any(frequencies < 0):
        raise ValueError(
            "ESPRIT frequencies cannot be negative."
        )

    if max_frequency is not None:
        max_frequency = float(max_frequency)

        if not np.isfinite(max_frequency) or max_frequency <= 0:
            raise ValueError(
                "max_frequency must be greater than zero."
            )

        frequencies = frequencies[
            frequencies <= max_frequency
        ]

        if frequencies.size == 0:
            raise ValueError(
                "max_frequency is below all estimated frequencies."
            )

    frequencies = np.sort(frequencies)

    figure, axis = plt.subplots()

    positions = np.arange(1, len(frequencies) + 1)

    markerline, stemlines, baseline = axis.stem(
        positions,
        frequencies,
    )

    axis.set_title("ESPRIT Frequency Estimates")
    axis.set_xlabel("Estimated Component")
    axis.set_ylabel("Frequency (Hz)")
    axis.grid(True)

    axis.set_xticks(positions)

    figure.tight_layout()

    return figure, axis