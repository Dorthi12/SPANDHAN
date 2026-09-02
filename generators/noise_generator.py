"""
Noise Generator module.
"""

import numpy as np


def _validate_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or infinite values.")

    return signal


def _validate_snr(snr_db: float) -> float:
    snr_db = float(snr_db)

    if not np.isfinite(snr_db):
        raise ValueError("SNR must be finite.")

    return snr_db


def add_gaussian_noise(
    signal: np.ndarray,
    snr_db: float,
    random_state: int | None = None,
) -> np.ndarray:

    signal = _validate_signal(signal)
    snr_db = _validate_snr(snr_db)

    rng = np.random.default_rng(random_state)

    signal_power = np.mean(signal ** 2)

    if signal_power == 0:
        raise ValueError("Cannot determine SNR for a zero-power signal.")

    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_power),
        size=len(signal),
    )

    return signal + noise


def add_impulse_noise(
    signal: np.ndarray,
    probability: float = 0.01,
    amplitude_factor: float = 5.0,
    random_state: int | None = None,
) -> np.ndarray:

    signal = _validate_signal(signal)

    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    if amplitude_factor < 0:
        raise ValueError("Amplitude factor cannot be negative.")

    rng = np.random.default_rng(random_state)

    noisy_signal = signal.copy()

    mask = rng.random(len(signal)) < probability

    signal_std = np.std(signal)

    if signal_std == 0:
        signal_std = 1.0

    impulses = rng.choice(
        [-1.0, 1.0],
        size=np.count_nonzero(mask),
    )

    noisy_signal[mask] += (
        impulses
        * amplitude_factor
        * signal_std
    )

    return noisy_signal


def add_periodic_noise(
    signal: np.ndarray,
    sampling_rate: float,
    frequency: float,
    amplitude: float,
) -> np.ndarray:

    signal = _validate_signal(signal)

    sampling_rate = float(sampling_rate)
    frequency = float(frequency)
    amplitude = float(amplitude)

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than zero.")

    if frequency < 0:
        raise ValueError("Frequency cannot be negative.")

    if amplitude < 0:
        raise ValueError("Amplitude cannot be negative.")

    t = np.arange(len(signal)) / sampling_rate

    noise = amplitude * np.sin(
        2 * np.pi * frequency * t
    )

    return signal + noise


def add_colored_noise(
    signal: np.ndarray,
    color: str = "pink",
    strength: float = 0.1,
    random_state: int | None = None,
) -> np.ndarray:

    signal = _validate_signal(signal)

    if strength < 0:
        raise ValueError("Strength cannot be negative.")

    color = color.lower()

    if color not in {"pink", "brown"}:
        raise ValueError(
            "Color must be either 'pink' or 'brown'."
        )

    rng = np.random.default_rng(random_state)

    white_noise = rng.normal(
        0.0,
        1.0,
        len(signal),
    )

    spectrum = np.fft.rfft(white_noise)
    frequencies = np.fft.rfftfreq(len(signal))

    frequencies[0] = frequencies[1] if len(frequencies) > 1 else 1.0

    if color == "pink":
        shaping = 1.0 / np.sqrt(frequencies)
    else:
        shaping = 1.0 / frequencies

    colored_spectrum = spectrum * shaping

    colored_noise = np.fft.irfft(
        colored_spectrum,
        n=len(signal),
    )

    std = np.std(colored_noise)

    if std > 0:
        colored_noise /= std

    signal_std = np.std(signal)

    if signal_std == 0:
        signal_std = 1.0

    colored_noise *= signal_std * strength

    return signal + colored_noise


def add_mixed_noise(
    signal: np.ndarray,
    sampling_rate: float,
    snr_db: float = 15.0,
    periodic_frequency: float = 50.0,
    periodic_amplitude: float = 0.1,
    impulse_probability: float = 0.005,
    random_state: int | None = None,
) -> np.ndarray:

    signal = _validate_signal(signal)

    noisy_signal = add_gaussian_noise(
        signal,
        snr_db=snr_db,
        random_state=random_state,
    )

    noisy_signal = add_periodic_noise(
        noisy_signal,
        sampling_rate=sampling_rate,
        frequency=periodic_frequency,
        amplitude=periodic_amplitude,
    )

    noisy_signal = add_impulse_noise(
        noisy_signal,
        probability=impulse_probability,
        random_state=random_state,
    )

    return noisy_signal