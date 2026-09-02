"""
Dc Removal module.
"""

import numpy as np


def remove_dc(signal: np.ndarray) -> np.ndarray:
    """
    Remove the mean (DC component) from a 1-D signal.
    """

    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or infinite values.")

    return signal - np.mean(signal)