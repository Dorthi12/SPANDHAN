"""
Stft module.
"""

import numpy as np
from scipy import signal


def compute_stft(
    signal_data: np.ndarray,
    sampling_rate: float,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform.

    Returns
    -------
    frequencies : np.ndarray
        Frequency axis in Hz.

    times : np.ndarray
        Time axis in seconds.

    magnitude : np.ndarray
        Magnitude of the STFT.
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

    nperseg = int(nperseg)

    if nperseg < 2:
        raise ValueError(
            "nperseg must be at least 2."
        )

    if nperseg > len(signal_data):
        raise ValueError(
            "nperseg cannot exceed signal length."
        )

    if noverlap is not None:

        noverlap = int(noverlap)

        if noverlap < 0:
            raise ValueError(
                "noverlap cannot be negative."
            )

        if noverlap >= nperseg:
            raise ValueError(
                "noverlap must be smaller than nperseg."
            )

    frequencies, times, zxx = signal.stft(
        signal_data,
        fs=sampling_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
    )

    magnitude = np.abs(zxx)

    return frequencies, times, magnitude