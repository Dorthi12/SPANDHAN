"""
Wavelet module.
"""

import numpy as np
import pywt


def _validate_signal(signal_data):
    signal_data = np.asarray(signal_data, dtype=np.float64)

    if signal_data.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")
    if signal_data.size == 0:
        raise ValueError("Signal is empty.")
    if not np.all(np.isfinite(signal_data)):
        raise ValueError("Signal contains NaN or infinite values.")

    return signal_data


def discrete_wavelet_transform(
    signal_data,
    wavelet="db4",
    level=None,
):
    signal_data = _validate_signal(signal_data)

    wavelet_obj = pywt.Wavelet(wavelet)

    max_level = pywt.dwt_max_level(
        len(signal_data),
        wavelet_obj.dec_len,
    )

    if max_level < 1:
        raise ValueError("Signal is too short for wavelet decomposition.")

    if level is None:
        level = min(4, max_level)

    if not isinstance(level, int) or level < 1:
        raise ValueError("Level must be a positive integer.")

    if level > max_level:
        raise ValueError(
            f"Level {level} exceeds maximum allowed level {max_level}."
        )

    coefficients = pywt.wavedec(
        signal_data,
        wavelet,
        level=level,
    )

    return {
        "coefficients": coefficients,
        "approximation": coefficients[0],
        "details": coefficients[1:],
        "wavelet": wavelet,
        "level": level,
    }