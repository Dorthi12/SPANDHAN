"""
Analyzer module.
"""

import numpy as np
from scipy.signal import welch


def _validate_noise(noise):
    noise = np.asarray(noise, dtype=np.float64)

    if noise.ndim != 1:
        raise ValueError("Noise signal must be one-dimensional.")

    if noise.size == 0:
        raise ValueError("Noise signal cannot be empty.")

    if not np.all(np.isfinite(noise)):
        raise ValueError("Noise signal must contain only finite values.")

    return noise


def _validate_sampling_rate(sampling_rate):
    sampling_rate = float(sampling_rate)

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be finite and greater than zero.")

    return sampling_rate


def analyze_noise(noise, sampling_rate):
    """
    Characterize a noise signal using time-domain and spectral metrics.

    Returns
    -------
    dict
        Noise characterization metrics and PSD information.
    """

    noise = _validate_noise(noise)
    sampling_rate = _validate_sampling_rate(sampling_rate)

    mean = float(np.mean(noise))
    rms = float(np.sqrt(np.mean(noise ** 2)))
    variance = float(np.var(noise))
    standard_deviation = float(np.std(noise))

    peak = float(np.max(np.abs(noise)))
    peak_to_peak = float(np.ptp(noise))

    if rms > 0:
        crest_factor = float(peak / rms)
    else:
        crest_factor = 0.0

    # Zero-crossing rate.
    if len(noise) > 1:
        signs = np.sign(noise)
        signs[signs == 0] = 1.0
        zero_crossings = np.sum(signs[:-1] != signs[1:])
        zero_crossing_rate = float(zero_crossings / (len(noise) - 1))
    else:
        zero_crossing_rate = 0.0

    # Welch power spectral density.
    nperseg = min(256, len(noise))

    if nperseg >= 2:
        frequencies, psd = welch(
            noise,
            fs=sampling_rate,
            nperseg=nperseg,
        )

        total_power = float(np.trapezoid(psd, frequencies))

        if total_power > 0:
            spectral_centroid = float(
                np.trapezoid(frequencies * psd, frequencies)
                / total_power
            )

            dominant_frequency = float(
                frequencies[np.argmax(psd)]
            )
        else:
            spectral_centroid = 0.0
            dominant_frequency = 0.0
    else:
        frequencies = np.array([], dtype=np.float64)
        psd = np.array([], dtype=np.float64)
        total_power = 0.0
        spectral_centroid = 0.0
        dominant_frequency = 0.0

    return {
        "mean": mean,
        "rms": rms,
        "variance": variance,
        "std": standard_deviation,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_centroid": spectral_centroid,
        "dominant_frequency": dominant_frequency,
        "spectral_power": total_power,
        "frequency_axis": frequencies,
        "psd": psd,
    }