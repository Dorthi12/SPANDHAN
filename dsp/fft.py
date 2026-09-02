"""
Fft module.
"""

import numpy as np


def compute_fft(
    signal_data: np.ndarray,
    sampling_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the one-sided FFT magnitude spectrum.

    Returns
    -------
    frequencies : np.ndarray
        Frequency axis in Hz.

    magnitude : np.ndarray
        One-sided magnitude spectrum.
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

    n = len(signal_data)

    spectrum = np.fft.rfft(signal_data)

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / sampling_rate,
    )

    magnitude = np.abs(spectrum) / n

    # Convert to the conventional one-sided amplitude
    # spectrum. DC and Nyquist components are not doubled.
    if n > 1:

        if n % 2 == 0:
            magnitude[1:-1] *= 2.0
        else:
            magnitude[1:] *= 2.0

    return frequencies, magnitude