"""
Hilbert module.
"""

import numpy as np
from scipy.signal import hilbert


def compute_hilbert(
    signal_data: np.ndarray,
    sampling_rate: float,
) -> dict[str, np.ndarray]:
    """
    Compute analytic-signal quantities using the Hilbert transform.

    Returns:
        analytic_signal
        envelope
        phase
        instantaneous_frequency
    """

    signal_data = np.asarray(
        signal_data,
        dtype=np.float64,
    )

    if signal_data.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if signal_data.size < 2:
        raise ValueError(
            "Signal must contain at least two samples."
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

    analytic_signal = hilbert(signal_data)

    envelope = np.abs(analytic_signal)

    phase = np.unwrap(
        np.angle(analytic_signal)
    )

    instantaneous_frequency = (
        np.gradient(phase)
        * sampling_rate
        / (2 * np.pi)
    )

    return {
        "analytic_signal": analytic_signal,
        "envelope": envelope,
        "phase": phase,
        "instantaneous_frequency":
            instantaneous_frequency,
    }