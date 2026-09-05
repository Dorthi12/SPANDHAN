from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FrequencyComparison:
    target_frequencies: np.ndarray
    estimated_frequencies: dict[str, np.ndarray] = field(default_factory=dict)
    errors: dict[str, np.ndarray] = field(default_factory=dict)
    resolved: dict[str, bool] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)


def compare_frequency_estimates(
    target_frequencies,
    estimated_frequencies,
    tolerance_hz=1.0,
):
    target_frequencies = np.asarray(
        target_frequencies,
        dtype=np.float64,
    )

    if target_frequencies.ndim != 1:
        raise ValueError("Target frequencies must be one-dimensional.")

    if target_frequencies.size == 0:
        raise ValueError("Target frequencies cannot be empty.")

    if not np.all(np.isfinite(target_frequencies)):
        raise ValueError("Target frequencies must be finite.")

    tolerance_hz = float(tolerance_hz)

    if not np.isfinite(tolerance_hz) or tolerance_hz <= 0:
        raise ValueError("Tolerance must be greater than zero.")

    normalized_estimates = {}
    errors = {}
    resolved = {}

    for method, frequencies in estimated_frequencies.items():
        frequencies = np.asarray(
            frequencies,
            dtype=np.float64,
        )

        if frequencies.ndim != 1:
            raise ValueError(
                f"Estimated frequencies for {method} "
                "must be one-dimensional."
            )

        if not np.all(np.isfinite(frequencies)):
            raise ValueError(
                f"Estimated frequencies for {method} "
                "must be finite."
            )

        frequencies = np.sort(frequencies)
        normalized_estimates[method] = frequencies

        if len(frequencies) != len(target_frequencies):
            errors[method] = np.full(
                len(target_frequencies),
                np.inf,
            )
            resolved[method] = False
            continue

        frequency_errors = np.abs(
            frequencies - np.sort(target_frequencies)
        )

        errors[method] = frequency_errors
        resolved[method] = bool(
            np.all(frequency_errors <= tolerance_hz)
        )

    return FrequencyComparison(
        target_frequencies=np.sort(target_frequencies),
        estimated_frequencies=normalized_estimates,
        errors=errors,
        resolved=resolved,
        parameters={
            "tolerance_hz": tolerance_hz,
        },
    )