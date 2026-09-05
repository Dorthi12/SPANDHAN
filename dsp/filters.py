"""
Filters module.
"""

import numpy as np
from scipy import signal


def _validate_signal(signal_data: np.ndarray) -> np.ndarray:
    signal_data = np.asarray(signal_data, dtype=np.float64)

    if signal_data.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if signal_data.size == 0:
        raise ValueError("Signal cannot be empty.")

    if not np.all(np.isfinite(signal_data)):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    return signal_data


def _validate_sampling_rate(sampling_rate: float) -> float:
    sampling_rate = float(sampling_rate)

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError(
            "Sampling rate must be greater than zero."
        )

    return sampling_rate


def _validate_order(order: int) -> int:
    order = int(order)

    if order < 1:
        raise ValueError(
            "Filter order must be at least 1."
        )

    return order


def apply_lowpass(
    signal_data: np.ndarray,
    sampling_rate: float,
    cutoff_frequency: float,
    order: int = 4,
) -> np.ndarray:

    signal_data = _validate_signal(signal_data)
    sampling_rate = _validate_sampling_rate(sampling_rate)
    order = _validate_order(order)

    nyquist = sampling_rate / 2.0

    if not 0 < cutoff_frequency < nyquist:
        raise ValueError(
            "Cutoff frequency must be between 0 and "
            "the Nyquist frequency."
        )

    sos = signal.butter(
        order,
        cutoff_frequency,
        btype="lowpass",
        fs=sampling_rate,
        output="sos",
    )

    return signal.sosfiltfilt(sos, signal_data)


def apply_highpass(
    signal_data: np.ndarray,
    sampling_rate: float,
    cutoff_frequency: float,
    order: int = 4,
) -> np.ndarray:

    signal_data = _validate_signal(signal_data)
    sampling_rate = _validate_sampling_rate(sampling_rate)
    order = _validate_order(order)

    nyquist = sampling_rate / 2.0

    if not 0 < cutoff_frequency < nyquist:
        raise ValueError(
            "Cutoff frequency must be between 0 and "
            "the Nyquist frequency."
        )

    sos = signal.butter(
        order,
        cutoff_frequency,
        btype="highpass",
        fs=sampling_rate,
        output="sos",
    )

    return signal.sosfiltfilt(sos, signal_data)


def apply_bandpass(
    signal_data: np.ndarray,
    sampling_rate: float,
    low_frequency: float,
    high_frequency: float,
    order: int = 4,
) -> np.ndarray:

    signal_data = _validate_signal(signal_data)
    sampling_rate = _validate_sampling_rate(sampling_rate)
    order = _validate_order(order)

    nyquist = sampling_rate / 2.0

    if not 0 < low_frequency < high_frequency < nyquist:
        raise ValueError(
            "Frequencies must satisfy "
            "0 < low < high < Nyquist."
        )

    sos = signal.butter(
        order,
        [low_frequency, high_frequency],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )

    return signal.sosfiltfilt(sos, signal_data)


def apply_bandstop(
    signal_data: np.ndarray,
    sampling_rate: float,
    low_frequency: float,
    high_frequency: float,
    order: int = 4,
) -> np.ndarray:

    signal_data = _validate_signal(signal_data)
    sampling_rate = _validate_sampling_rate(sampling_rate)
    order = _validate_order(order)

    nyquist = sampling_rate / 2.0

    if not 0 < low_frequency < high_frequency < nyquist:
        raise ValueError(
            "Frequencies must satisfy "
            "0 < low < high < Nyquist."
        )

    sos = signal.butter(
        order,
        [low_frequency, high_frequency],
        btype="bandstop",
        fs=sampling_rate,
        output="sos",
    )

    return signal.sosfiltfilt(sos, signal_data)