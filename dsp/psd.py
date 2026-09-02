"""
Psd module.
"""

import numpy as np
from scipy import signal


def compute_psd(
    signal_data: np.ndarray,
    sampling_rate: float,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Power Spectral Density using Welch's method.

    Returns
    -------
    frequencies : np.ndarray
        Frequency axis in Hz.

    psd : np.ndarray
        Power spectral density.
    """

    signal_data = np.asarray(
        signal_data,
        dtype=np.float64,
    )

    if signal_data.ndim != 1:
        raise ValueError(
            "Signal must be one-dimensional."
        )

    if signal_data.size == 0:
        raise ValueError(
            "Signal cannot be empty."
        )

    if not np.all(np.isfinite(signal_data)):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    sampling_rate = float(sampling_rate)

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError(
            "Sampling rate must be greater than zero."
        )

    if nperseg is not None:

        nperseg = int(nperseg)

        if nperseg < 2:
            raise ValueError(
                "nperseg must be at least 2."
            )

        if nperseg > len(signal_data):
            raise ValueError(
                "nperseg cannot exceed signal length."
            )

    frequencies, psd = signal.welch(
        signal_data,
        fs=sampling_rate,
        nperseg=nperseg,
        scaling="density",
    )

    return frequencies, psd