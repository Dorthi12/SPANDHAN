"""
Cepstrum module.
"""

import numpy as np


def _validate_signal(signal_data):
    signal_data = np.asarray(signal_data, dtype=np.float64)

    if signal_data.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if signal_data.size == 0:
        raise ValueError("Signal is empty.")

    if not np.all(np.isfinite(signal_data)):
        raise ValueError("Signal contains NaN or infinite values.")

    return signal_data


def compute_real_cepstrum(signal_data, sampling_rate):
    signal_data = _validate_signal(signal_data)

    sampling_rate = float(sampling_rate)

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError(
            "Sampling rate must be finite and greater than zero."
        )

    spectrum = np.fft.fft(signal_data)

    magnitude = np.abs(spectrum)

    # Avoid log(0)
    magnitude = np.maximum(magnitude, np.finfo(float).eps)

    log_magnitude = np.log(magnitude)

    cepstrum = np.fft.ifft(log_magnitude).real

    quefrency = np.arange(len(signal_data)) / sampling_rate

    return quefrency, cepstrum