"""
Detrending module.
"""

import numpy as np
from scipy import signal


def detrend_signal(signal_data: np.ndarray) -> np.ndarray:
    """
    Remove the linear trend from a 1-D signal.
    """

    signal_data = np.asarray(signal_data, dtype=np.float64)

    if signal_data.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if signal_data.size == 0:
        raise ValueError("Signal cannot be empty.")

    if not np.all(np.isfinite(signal_data)):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    return signal.detrend(signal_data, type="linear")