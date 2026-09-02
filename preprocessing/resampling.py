"""
Resampling module.
"""
import numpy as np
from scipy import signal


def resample_signal(
    signal_data: np.ndarray,
    original_rate: float,
    target_rate: float,
) -> np.ndarray:
    """
    Resample a 1-D signal from original_rate to target_rate.
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

    original_rate = float(original_rate)
    target_rate = float(target_rate)

    if not np.isfinite(original_rate) or original_rate <= 0:
        raise ValueError(
            "Original sampling rate must be greater than zero."
        )

    if not np.isfinite(target_rate) or target_rate <= 0:
        raise ValueError(
            "Target sampling rate must be greater than zero."
        )

    if np.isclose(original_rate, target_rate):
        return signal_data.copy()

    ratio = target_rate / original_rate

    # Convert the sampling-rate ratio into integer factors.
    from math import gcd

    original_int = int(round(original_rate))
    target_int = int(round(target_rate))

    divisor = gcd(original_int, target_int)

    up = target_int // divisor
    down = original_int // divisor

    new_length = int(round(len(signal_data) * ratio))

    return signal.resample_poly(
        signal_data,
        up,
        down,
    )[:new_length]