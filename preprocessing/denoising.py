"""
Denoising module.

general- purpose low-pass Butterworth filter
"""

import numpy as np
from scipy import signal


def lowpass_denoise(
    signal_data: np.ndarray,
    sampling_rate: float,
    cutoff_frequency: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth low-pass filter.
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

    sampling_rate = float(sampling_rate)
    cutoff_frequency = float(cutoff_frequency)
    order = int(order)

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError(
            "Sampling rate must be greater than zero."
        )

    nyquist = sampling_rate / 2.0

    if not 0 < cutoff_frequency < nyquist:
        raise ValueError(
            "Cutoff frequency must be between 0 and "
            "the Nyquist frequency."
        )

    if order < 1:
        raise ValueError("Filter order must be at least 1.")

    if len(signal_data) < 3 * order + 1:
        raise ValueError(
            "Signal is too short for the requested filter order."
        )

    normalized_cutoff = cutoff_frequency / nyquist

    sos = signal.butter(
        order,
        normalized_cutoff,
        btype="lowpass",
        output="sos",
    )

    return signal.sosfiltfilt(
        sos,
        signal_data,
    )