"""
Estimator module.
"""
import numpy as np


def _validate_signal(signal, name):
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if signal.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(signal)):
        raise ValueError(f"{name} must contain only finite values.")

    return signal


def estimate_noise(
    clean_signal,
    noisy_signal,
):
    clean_signal = _validate_signal(
        clean_signal,
        "clean_signal",
    )

    noisy_signal = _validate_signal(
        noisy_signal,
        "noisy_signal",
    )

    if len(clean_signal) != len(noisy_signal):
        raise ValueError(
            "Clean and noisy signals must have the same length."
        )

    noise = noisy_signal - clean_signal

    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power <= 0:
        snr_db = np.inf
    else:
        snr_db = 10.0 * np.log10(
            signal_power / noise_power
        )

    return {
        "noise": noise,
        "signal_power": float(signal_power),
        "noise_power": float(noise_power),
        "snr_db": float(snr_db),
    }