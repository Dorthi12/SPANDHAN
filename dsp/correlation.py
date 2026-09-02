"""
Correlation module.
"""

import numpy as np
from scipy import signal


def compute_cross_correlation(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized cross-correlation between two
    one-dimensional signals.

    Returns
    -------
    lags : np.ndarray
        Sample lags corresponding to the correlation.

    correlation : np.ndarray
        Normalized cross-correlation values.
    """

    signal_a = np.asarray(signal_a, dtype=np.float64)
    signal_b = np.asarray(signal_b, dtype=np.float64)

    if signal_a.ndim != 1:
        raise ValueError("signal_a must be one-dimensional.")

    if signal_b.ndim != 1:
        raise ValueError("signal_b must be one-dimensional.")

    if signal_a.size == 0 or signal_b.size == 0:
        raise ValueError("Signals cannot be empty.")

    if not np.all(np.isfinite(signal_a)):
        raise ValueError(
            "signal_a contains NaN or infinite values."
        )

    if not np.all(np.isfinite(signal_b)):
        raise ValueError(
            "signal_b contains NaN or infinite values."
        )

    signal_a = signal_a - np.mean(signal_a)
    signal_b = signal_b - np.mean(signal_b)

    norm = np.linalg.norm(signal_a) * np.linalg.norm(signal_b)

    if norm == 0:
        raise ValueError(
            "Correlation is undefined for a zero-variance signal."
        )

    correlation = signal.correlate(
        signal_a,
        signal_b,
        mode="full",
        method="auto",
    )

    correlation = correlation / norm

    lags = signal.correlation_lags(
        len(signal_a),
        len(signal_b),
        mode="full",
    )

    return lags, correlation


def find_correlation_peak(
    lags: np.ndarray,
    correlation: np.ndarray,
) -> tuple[int, float]:
    """
    Find the lag corresponding to the maximum absolute
    normalized correlation.
    """

    lags = np.asarray(lags)

    correlation = np.asarray(
        correlation,
        dtype=np.float64,
    )

    if lags.ndim != 1 or correlation.ndim != 1:
        raise ValueError(
            "lags and correlation must be one-dimensional."
        )

    if len(lags) != len(correlation):
        raise ValueError(
            "lags and correlation must have equal length."
        )

    if len(correlation) == 0:
        raise ValueError(
            "Correlation cannot be empty."
        )

    index = np.argmax(np.abs(correlation))

    return int(lags[index]), float(correlation[index])