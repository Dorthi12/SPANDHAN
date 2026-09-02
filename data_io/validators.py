"""
Validators module.
"""

from pathlib import Path

import numpy as np


SUPPORTED_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".csv",
    ".mat",
    ".txt",
}


def validate_file_path(file_path: str | Path) -> Path:
    """Validate that the supplied file exists and has a supported extension."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Signal file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return path


def validate_signal(signal: np.ndarray) -> np.ndarray:
    """Validate and normalize a signal array."""

    if not isinstance(signal, np.ndarray):
        signal = np.asarray(signal)

    if signal.size == 0:
        raise ValueError("Signal is empty.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or infinite values.")

    if signal.ndim > 2:
        raise ValueError(
            f"Unsupported signal dimensions: {signal.shape}"
        )

    return signal.astype(np.float64, copy=False)


def validate_sampling_rate(sampling_rate: float) -> float:
    """Validate a sampling rate."""

    sampling_rate = float(sampling_rate)

    if not np.isfinite(sampling_rate):
        raise ValueError("Sampling rate must be finite.")

    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be greater than zero.")

    return sampling_rate