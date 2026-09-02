"""
Waveform Generator module.
"""

import numpy as np


def _validate_sampling_rate(sampling_rate: float) -> float:
    sampling_rate = float(sampling_rate)

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than zero.")

    return sampling_rate


def _validate_duration(duration: float) -> float:
    duration = float(duration)

    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be greater than zero.")

    return duration


def _time_vector(sampling_rate: float, duration: float) -> np.ndarray:
    sampling_rate = _validate_sampling_rate(sampling_rate)
    duration = _validate_duration(duration)

    num_samples = int(round(sampling_rate * duration))

    if num_samples < 2:
        raise ValueError("Signal must contain at least two samples.")

    return np.arange(num_samples) / sampling_rate


def generate_sine(
    frequency: float,
    sampling_rate: float,
    duration: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:

    if frequency < 0:
        raise ValueError("Frequency cannot be negative.")

    if amplitude < 0:
        raise ValueError("Amplitude cannot be negative.")

    t = _time_vector(sampling_rate, duration)

    signal = amplitude * np.sin(
        2 * np.pi * frequency * t + phase
    )

    return t, signal


def generate_multi_tone(
    frequencies: list[float],
    sampling_rate: float,
    duration: float,
    amplitudes: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:

    if not frequencies:
        raise ValueError("At least one frequency is required.")

    if any(f < 0 for f in frequencies):
        raise ValueError("Frequencies cannot be negative.")

    t = _time_vector(sampling_rate, duration)

    if amplitudes is None:
        amplitudes = [1.0] * len(frequencies)

    if len(amplitudes) != len(frequencies):
        raise ValueError(
            "Frequencies and amplitudes must have the same length."
        )

    signal = np.zeros_like(t)

    for frequency, amplitude in zip(frequencies, amplitudes):
        if amplitude < 0:
            raise ValueError("Amplitudes cannot be negative.")

        signal += amplitude * np.sin(
            2 * np.pi * frequency * t
        )

    return t, signal


def generate_square(
    frequency: float,
    sampling_rate: float,
    duration: float,
    amplitude: float = 1.0,
    duty_cycle: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:

    if frequency <= 0:
        raise ValueError("Frequency must be greater than zero.")

    if amplitude < 0:
        raise ValueError("Amplitude cannot be negative.")

    if not 0 < duty_cycle < 1:
        raise ValueError("Duty cycle must be between 0 and 1.")

    t = _time_vector(sampling_rate, duration)

    phase = (frequency * t) % 1.0

    signal = np.where(
        phase < duty_cycle,
        amplitude,
        -amplitude,
    )

    return t, signal


def generate_sawtooth(
    frequency: float,
    sampling_rate: float,
    duration: float,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:

    if frequency <= 0:
        raise ValueError("Frequency must be greater than zero.")

    if amplitude < 0:
        raise ValueError("Amplitude cannot be negative.")

    t = _time_vector(sampling_rate, duration)

    phase = (frequency * t) % 1.0

    signal = amplitude * (2.0 * phase - 1.0)

    return t, signal


def generate_impulse(
    sampling_rate: float,
    duration: float,
    position: int = 0,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:

    t = _time_vector(sampling_rate, duration)

    if not 0 <= position < len(t):
        raise ValueError(
            f"Position must be between 0 and {len(t) - 1}."
        )

    if amplitude < 0:
        raise ValueError("Amplitude cannot be negative.")

    signal = np.zeros_like(t)
    signal[position] = amplitude

    return t, signal