"""
Signal Loader module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

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
    file_path: Union[str, Path],
    sampling_rate: Optional[float] = None,
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
    try:
        import soundfile as sf
        signal, fs = sf.read(path)
    except (ImportError, Exception):
        from scipy.io import wavfile
        fs, signal = wavfile.read(path)
        if signal.dtype == np.int16:
            signal = signal.astype(np.float64) / 32768.0
        elif signal.dtype == np.int32:
            signal = signal.astype(np.float64) / 2147483648.0
        elif signal.dtype == np.uint8:
            signal = (signal.astype(np.float64) - 128.0) / 128.0

    signal = np.asarray(signal, dtype=np.float64)

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
        sampling_rate = 1000.0

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

    # Check for embedded sampling rate in MAT file metadata
    if sampling_rate is None:
        for sr_key in ("sampling_rate", "fs", "Fs", "samplingRate", "sample_rate", "rate", "SAMPLING_RATE"):
            if sr_key in data:
                val = data[sr_key]
                try:
                    if isinstance(val, np.ndarray):
                        val = val.squeeze().item() if val.size == 1 else float(val.ravel()[0])
                    sampling_rate = float(val)
                    break
                except (ValueError, TypeError):
                    pass

    # Fallback to standard 1000.0 Hz if sampling rate is still not determined
    if sampling_rate is None:
        sampling_rate = 1000.0

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
        sampling_rate = 1000.0

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