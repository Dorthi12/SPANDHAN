import numpy as np
from scipy.signal import welch


def _validate_signal(signal, name):
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if signal.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(signal)):
        raise ValueError(f"{name} must contain only finite values.")

    return signal


def _validate_sampling_rate(sampling_rate):
    sampling_rate = float(sampling_rate)

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError(
            "Sampling rate must be finite and greater than zero."
        )

    return sampling_rate


def _calculate_snr(reference, signal):
    error = signal - reference

    signal_power = np.mean(reference ** 2)
    error_power = np.mean(error ** 2)

    if error_power <= 0:
        return np.inf

    if signal_power <= 0:
        return -np.inf

    return float(10.0 * np.log10(signal_power / error_power))


def _calculate_rmse(reference, signal):
    error = signal - reference
    return float(np.sqrt(np.mean(error ** 2)))


def _calculate_correlation(reference, signal):
    reference_centered = reference - np.mean(reference)
    signal_centered = signal - np.mean(signal)

    reference_norm = np.linalg.norm(reference_centered)
    signal_norm = np.linalg.norm(signal_centered)

    if reference_norm == 0 or signal_norm == 0:
        return 0.0

    return float(
        np.dot(reference_centered, signal_centered)
        / (reference_norm * signal_norm)
    )


def _calculate_energy(signal):
    return float(np.sum(signal ** 2))


def _calculate_spectral_distortion(
    reference,
    signal,
    sampling_rate,
):
    nperseg = min(256, len(reference))

    if nperseg < 2:
        return 0.0

    frequencies_ref, psd_ref = welch(
        reference,
        fs=sampling_rate,
        nperseg=nperseg,
    )

    frequencies_sig, psd_sig = welch(
        signal,
        fs=sampling_rate,
        nperseg=nperseg,
    )

    if not np.array_equal(frequencies_ref, frequencies_sig):
        psd_sig = np.interp(
            frequencies_ref,
            frequencies_sig,
            psd_sig,
        )

    reference_power = np.trapezoid(psd_ref, frequencies_ref)

    if reference_power <= 0:
        return 0.0

    difference_power = np.trapezoid(
        (psd_ref - psd_sig) ** 2,
        frequencies_ref,
    )

    return float(
        np.sqrt(difference_power / reference_power)
    )


def compare_signal_quality(
    reference_signal,
    before_signal,
    after_signal,
    sampling_rate,
):
    """
    Compare signal quality before and after processing.

    Parameters
    ----------
    reference_signal : array-like
        Clean/reference signal.

    before_signal : array-like
        Signal before denoising or processing.

    after_signal : array-like
        Signal after denoising or processing.

    sampling_rate : float
        Sampling frequency in Hz.

    Returns
    -------
    dict
        Before/after quality metrics and improvement values.
    """

    reference_signal = _validate_signal(
        reference_signal,
        "reference_signal",
    )

    before_signal = _validate_signal(
        before_signal,
        "before_signal",
    )

    after_signal = _validate_signal(
        after_signal,
        "after_signal",
    )

    sampling_rate = _validate_sampling_rate(sampling_rate)

    if len(reference_signal) != len(before_signal):
        raise ValueError(
            "Reference and before signals must have the same length."
        )

    if len(reference_signal) != len(after_signal):
        raise ValueError(
            "Reference and after signals must have the same length."
        )

    before_snr = _calculate_snr(
        reference_signal,
        before_signal,
    )

    after_snr = _calculate_snr(
        reference_signal,
        after_signal,
    )

    snr_improvement = after_snr - before_snr

    before_rmse = _calculate_rmse(
        reference_signal,
        before_signal,
    )

    after_rmse = _calculate_rmse(
        reference_signal,
        after_signal,
    )

    before_correlation = _calculate_correlation(
        reference_signal,
        before_signal,
    )

    after_correlation = _calculate_correlation(
        reference_signal,
        after_signal,
    )

    reference_energy = _calculate_energy(reference_signal)
    before_energy = _calculate_energy(before_signal)
    after_energy = _calculate_energy(after_signal)

    before_energy_error = (
        abs(before_energy - reference_energy)
        / reference_energy
        if reference_energy > 0
        else 0.0
    )

    after_energy_error = (
        abs(after_energy - reference_energy)
        / reference_energy
        if reference_energy > 0
        else 0.0
    )

    before_spectral_distortion = _calculate_spectral_distortion(
        reference_signal,
        before_signal,
        sampling_rate,
    )

    after_spectral_distortion = _calculate_spectral_distortion(
        reference_signal,
        after_signal,
        sampling_rate,
    )

    return {
    "reference_energy": float(reference_energy),

    "before": {
        "snr_db": float(before_snr),
        "rmse": before_rmse,
        "correlation": before_correlation,
        "energy": before_energy,
        "energy_error": float(before_energy_error),
        "spectral_distortion": before_spectral_distortion,
    },

    "after": {
        "snr_db": float(after_snr),
        "rmse": after_rmse,
        "correlation": after_correlation,
        "energy": after_energy,
        "energy_error": float(after_energy_error),
        "spectral_distortion": after_spectral_distortion,
    },

    "improvement": {
        "snr_db": float(snr_improvement),
        "rmse_reduction": float(before_rmse - after_rmse),
        "correlation_change": float(
            after_correlation - before_correlation
        ),
        "energy_error_reduction": float(
            before_energy_error - after_energy_error
        ),
        "spectral_distortion_reduction": float(
            before_spectral_distortion
            - after_spectral_distortion
        ),
    },
}