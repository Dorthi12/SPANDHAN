"""
Signal Loader module.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from core.session import SignalData
from data_io.validators import (
    validate_file_path,
    validate_sampling_rate,
    validate_signal,
)


def load_signal(
    file_path: str | Path,
    sampling_rate: float | None = None,
    domain: str = "general",
) -> SignalData:
    """
    Load a signal from WAV, CSV, MAT, or TXT.

    Parameters
    ----------
    file_path:
        Path to input file.

    sampling_rate:
        Sampling rate for formats that do not contain it.

    domain:
        Spandhan analysis domain.

    Returns
    -------
    SignalData
        Normalized signal representation.
    """

    path = validate_file_path(file_path)

    extension = path.suffix.lower()

    if extension in {".wav", ".flac", ".mp3"}:
        return _load_audio(path, domain)

    if extension == ".csv":
        return _load_csv(path, sampling_rate, domain)

    if extension == ".mat":
        return _load_mat(path, sampling_rate, domain)

    if extension == ".txt":
        return _load_txt(path, sampling_rate, domain)

    raise ValueError(f"Unsupported file extension: {extension}")


def _load_audio(path: Path, domain: str) -> SignalData:
    import soundfile as sf

    signal, fs = sf.read(path)

    signal = np.asarray(signal)

    # Convert stereo/multichannel audio to mono.
    if signal.ndim == 2:
        signal = np.mean(signal, axis=1)

    signal = validate_signal(signal)
    fs = validate_sampling_rate(fs)

    return SignalData(
        signal=signal,
        sampling_rate=fs,
        filename=path.name,
        domain=domain,
        source=str(path),
        channels=1,
    )


def _load_csv(
    path: Path,
    sampling_rate: float | None,
    domain: str,
) -> SignalData:

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError("CSV file is empty.")

    numeric_columns = data.select_dtypes(
        include=[np.number]
    ).columns

    if len(numeric_columns) == 0:
        raise ValueError(
            "CSV contains no numeric columns."
        )

    # For V1, use the first numeric column as the signal.
    signal = data[numeric_columns[0]].to_numpy()

    if sampling_rate is None:
        raise ValueError(
            "Sampling rate must be supplied for CSV files."
        )

    signal = validate_signal(signal)
    fs = validate_sampling_rate(sampling_rate)

    return SignalData(
        signal=signal,
        sampling_rate=fs,
        filename=path.name,
        domain=domain,
        source=str(path),
        channels=1,
    )


def _load_mat(
    path: Path,
    sampling_rate: float | None,
    domain: str,
) -> SignalData:

    data = loadmat(path)

    candidates = []

    for key, value in data.items():

        if key.startswith("__"):
            continue

        if isinstance(value, np.ndarray) and np.issubdtype(
            value.dtype,
            np.number,
        ):
            if value.size > 1:
                candidates.append((key, value))

    if not candidates:
        raise ValueError(
            "MAT file contains no suitable numeric signal."
        )

    _, signal = max(
        candidates,
        key=lambda item: item[1].size
    )

    signal = np.squeeze(signal)

    if signal.ndim != 1:
        raise ValueError(
            f"MAT signal must be one-dimensional. "
            f"Received shape: {signal.shape}"
        )

    if sampling_rate is None:
        raise ValueError(
            "Sampling rate must be supplied for MAT files."
        )

    signal = validate_signal(signal)
    fs = validate_sampling_rate(sampling_rate)

    return SignalData(
        signal=signal,
        sampling_rate=fs,
        filename=path.name,
        domain=domain,
        source=str(path),
        channels=1,
    )


def _load_txt(
    path: Path,
    sampling_rate: float | None,
    domain: str,
) -> SignalData:

    signal = np.loadtxt(path)

    signal = np.squeeze(signal)

    if sampling_rate is None:
        raise ValueError(
            "Sampling rate must be supplied for TXT files."
        )

    signal = validate_signal(signal)

    if signal.ndim != 1:
        raise ValueError(
            "TXT signal must contain one signal column."
        )

    fs = validate_sampling_rate(sampling_rate)

    return SignalData(
        signal=signal,
        sampling_rate=fs,
        filename=path.name,
        domain=domain,
        source=str(path),
        channels=1,
    )