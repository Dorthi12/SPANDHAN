import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
from scipy.signal import welch


def _validate_signal(signal):
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal must contain only finite values.")

    return signal


def _validate_sampling_rate(sampling_rate):
    sampling_rate = float(sampling_rate)

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError(
            "Sampling rate must be finite and greater than zero."
        )

    return sampling_rate


def _spectral_features(signal, sampling_rate):
    frequencies, psd = welch(
        signal,
        fs=sampling_rate,
        nperseg=min(256, len(signal)),
    )

    total_power = np.trapezoid(psd, frequencies)

    if total_power <= 0:
        return {
            "spectral_centroid": 0.0,
            "spectral_flatness": 0.0,
            "spectral_entropy": 0.0,
            "spectral_rolloff": 0.0,
            "mains_band_energy": 0.0,
            "high_band_energy": 0.0,
        }

    # Spectral centroid
    spectral_centroid = (
        np.trapezoid(frequencies * psd, frequencies)
        / total_power
    )

    # Spectral flatness
    positive_psd = psd[psd > 0]

    if len(positive_psd) == 0:
        spectral_flatness = 0.0
    else:
        geometric_mean = np.exp(
            np.mean(np.log(positive_psd))
        )
        arithmetic_mean = np.mean(positive_psd)

        spectral_flatness = (
            geometric_mean / arithmetic_mean
            if arithmetic_mean > 0
            else 0.0
        )

    # Spectral entropy
    probability = psd / np.sum(psd)

    probability = probability[probability > 0]

    spectral_entropy = float(
        -np.sum(probability * np.log2(probability))
    )

    # Normalize entropy to [0, 1]
    if len(probability) > 1:
        spectral_entropy /= np.log2(len(probability))

    # 85% spectral rolloff
    cumulative_power = np.cumsum(psd)

    threshold = 0.85 * cumulative_power[-1]

    rolloff_index = np.searchsorted(
        cumulative_power,
        threshold,
    )

    rolloff_index = min(
        rolloff_index,
        len(frequencies) - 1,
    )

    spectral_rolloff = frequencies[rolloff_index]

    # Mains-frequency region: 45–55 Hz.
    # This is a FEATURE BAND, not an assumption about
    # the system's fundamental frequency.
    mains_mask = (
        (frequencies >= 45.0)
        & (frequencies <= 55.0)
    )

    mains_band_energy = np.trapezoid(
        psd[mains_mask],
        frequencies[mains_mask],
    ) if np.any(mains_mask) else 0.0

    # High-frequency region: upper 25% of Nyquist.
    nyquist = sampling_rate / 2.0

    high_band_mask = frequencies >= 0.75 * nyquist

    high_band_energy = np.trapezoid(
        psd[high_band_mask],
        frequencies[high_band_mask],
    ) if np.any(high_band_mask) else 0.0

    return {
        "spectral_centroid": float(spectral_centroid),
        "spectral_flatness": float(spectral_flatness),
        "spectral_entropy": float(spectral_entropy),
        "spectral_rolloff": float(spectral_rolloff),
        "mains_band_energy": float(mains_band_energy),
        "high_band_energy": float(high_band_energy),
    }


def extract_noise_features(
    signal,
    sampling_rate,
    snr_db=None,
):
    """
    Extract the project's fixed 14-feature vector.

    Parameters
    ----------
    signal : array-like
        Signal to characterize.

    sampling_rate : float
        Sampling frequency in Hz.

    snr_db : float or None
        Independently estimated SNR, when available.

    Returns
    -------
    dict
        Named feature dictionary.
    """

    signal = _validate_signal(signal)
    sampling_rate = _validate_sampling_rate(sampling_rate)

    mean = np.mean(signal)

    rms = np.sqrt(np.mean(signal ** 2))
    variance = np.var(signal)
    standard_deviation = np.std(signal)

    if np.allclose(signal, signal[0]):
        kurtosis_value = 0.0
        skewness_value = 0.0
    else:
        kurtosis_value = kurtosis(
            signal,
            fisher=True,
            bias=False,
        )
        skewness_value = skew(
            signal,
            bias=False,
        )
    peak = np.max(np.abs(signal))

    crest_factor = (
        peak / rms
        if rms > 0
        else 0.0
    )

    if len(signal) > 1:
        centered = signal - mean
        signs = np.sign(centered)

        signs[signs == 0] = 1.0

        zero_crossings = np.sum(
            signs[:-1] != signs[1:]
        )

        zero_crossing_rate = (
            zero_crossings / (len(signal) - 1)
        )
    else:
        zero_crossing_rate = 0.0

    spectral = _spectral_features(
        signal,
        sampling_rate,
    )

    if snr_db is None:
        snr_value = np.nan
    else:
        snr_value = float(snr_db)

        if not np.isfinite(snr_value):
            raise ValueError(
                "snr_db must be finite when provided."
            )

    return {
        "rms": float(rms),
        "variance": float(variance),
        "std": float(standard_deviation),
        "kurtosis": float(kurtosis_value),
        "skewness": float(skewness_value),
        "crest_factor": float(crest_factor),
        "zero_crossing_rate": float(zero_crossing_rate),
        "spectral_centroid": spectral[
            "spectral_centroid"
        ],
        "spectral_flatness": spectral[
            "spectral_flatness"
        ],
        "spectral_entropy": spectral[
            "spectral_entropy"
        ],
        "spectral_rolloff": spectral[
            "spectral_rolloff"
        ],
        "mains_band_energy": spectral[
            "mains_band_energy"
        ],
        "high_band_energy": spectral[
            "high_band_energy"
        ],
        "snr_db": snr_value,
    }