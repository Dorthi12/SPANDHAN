"""
Normalization module.
"""

import numpy as np


def normalize_signal(
    signal: np.ndarray,
    method: str = "peak",
) -> np.ndarray:
    """
    Normalize a 1-D signal.

    Supported methods:
        - peak: scale maximum absolute value to 1
        - zscore: zero mean and unit standard deviation
    """

    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    method = method.lower()

    if method == "peak":

        peak = np.max(np.abs(signal))

        if peak == 0:
            raise ValueError(
                "Cannot peak-normalize a zero signal."
            )

        return signal / peak

    if method == "zscore":

        mean = np.mean(signal)
        std = np.std(signal)

        if std == 0:
            raise ValueError(
                "Cannot z-score normalize a constant signal."
            )

        return (signal - mean) / std

    raise ValueError(
        f"Unsupported normalization method: {method}"
    )
