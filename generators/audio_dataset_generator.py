"""
generators/audio_dataset_generator.py
======================================
Synthetic audio dataset generator for Spandhan.

Milestone A — Dataset generation only.
No noise injection, no analysis, no frontend changes.

Design principles
-----------------
* Reuses existing generators.waveform_generator functions wherever
  possible (generate_sine, generate_multi_tone).
* Adds only what is genuinely missing: chirp, AM, FM, harmonic.
* Every sample carries complete, typed metadata so the pipeline can
  always reconstruct the experiment.
* SyntheticAudioDataset exposes a stable interface that can later be
  replaced by RealAudioDataset without modifying callers.
* No modifications to any existing module.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ------------------------------------------------------------------
# Reuse existing waveform generators
# ------------------------------------------------------------------
from generators.waveform_generator import (
    generate_sine,
    generate_multi_tone,
)

# ------------------------------------------------------------------
# Supported signal types
# ------------------------------------------------------------------
AUDIO_SIGNAL_TYPES = (
    "clean_sine",
    "multi_tone",
    "harmonic",
    "chirp",
    "amplitude_modulated",
    "frequency_modulated",
)

# Default output directory
DEFAULT_AUDIO_DATASET_DIR = Path("datasets/synthetic/audio")


# ==================================================================
# Data structures
# ==================================================================


@dataclass
class AudioSampleMetadata:
    """
    Complete, typed metadata for one generated audio sample.

    This is the contract between the generator and all downstream
    pipeline stages.  Every field must be JSON-serialisable.
    """

    sample_id: str
    domain: str                        # always "audio"
    signal_type: str
    sampling_rate: float
    duration: float
    num_samples: int
    seed: Optional[int]

    # Per-type parameters (not all fields are populated for every type)
    frequency: Optional[float] = None          # clean_sine, AM carrier, FM carrier
    amplitude: Optional[float] = None          # clean_sine
    frequencies: Optional[list[float]] = None  # multi_tone, harmonic
    amplitudes: Optional[list[float]] = None   # multi_tone, harmonic
    num_harmonics: Optional[int] = None        # harmonic
    fundamental: Optional[float] = None        # harmonic
    harmonic_decay: Optional[float] = None     # harmonic
    chirp_f0: Optional[float] = None           # chirp start frequency
    chirp_f1: Optional[float] = None           # chirp end frequency
    modulation_frequency: Optional[float] = None   # AM/FM
    modulation_depth: Optional[float] = None       # AM
    modulation_index: Optional[float] = None       # FM


@dataclass
class AudioSample:
    """
    One generated audio sample: raw signal + typed metadata.

    signal  — 1-D float64 NumPy array, never normalised here.
               Callers may normalise in their own pre-processing step.
    t       — Corresponding time vector (seconds).
    meta    — AudioSampleMetadata instance.
    """

    signal: np.ndarray
    t: np.ndarray
    meta: AudioSampleMetadata


@dataclass
class AudioDatasetConfig:
    """
    Fully-specified configuration for one dataset generation run.

    All fields are plain Python scalars so the config can be
    serialised to JSON without a custom encoder.
    """

    signal_types: list[str] = field(
        default_factory=lambda: list(AUDIO_SIGNAL_TYPES)
    )
    samples_per_type: int = 5
    sampling_rate: float = 44100.0
    duration: float = 2.0

    # Sine defaults
    frequency: float = 440.0
    amplitude: float = 1.0

    # Multi-tone
    secondary_frequencies: list[float] = field(
        default_factory=lambda: [880.0]
    )

    # Harmonic
    fundamental: float = 220.0
    num_harmonics: int = 5
    harmonic_decay: float = 0.5   # amplitude of k-th harmonic = A * decay^(k-1)

    # Chirp
    chirp_f0: float = 100.0
    chirp_f1: float = 2000.0

    # Amplitude modulation
    am_carrier_freq: float = 440.0
    am_modulation_freq: float = 5.0
    am_modulation_depth: float = 0.5   # 0 – 1

    # Frequency modulation
    fm_carrier_freq: float = 440.0
    fm_modulation_freq: float = 5.0
    fm_modulation_index: float = 2.0   # peak deviation = index * modulation_freq

    seed: Optional[int] = 42


# ==================================================================
# Low-level signal generators
# (only types NOT already in waveform_generator.py)
# ==================================================================


def _time_vector(sampling_rate: float, duration: float) -> np.ndarray:
    """Return a time vector consistent with the existing waveform_generator."""
    num_samples = int(round(sampling_rate * duration))
    if num_samples < 2:
        raise ValueError("Signal must contain at least two samples.")
    return np.arange(num_samples) / sampling_rate


def _generate_harmonic(
    fundamental: float,
    sampling_rate: float,
    duration: float,
    num_harmonics: int,
    amplitude: float = 1.0,
    harmonic_decay: float = 0.5,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Harmonic signal: sum of sinusoids at integer multiples of fundamental.

    Amplitude of the k-th harmonic (k=1 is fundamental) is:
        A * harmonic_decay^(k-1)
    """
    if fundamental <= 0:
        raise ValueError("Fundamental frequency must be positive.")
    if num_harmonics < 1:
        raise ValueError("num_harmonics must be at least 1.")
    if harmonic_decay < 0 or harmonic_decay > 1:
        raise ValueError("harmonic_decay must be in [0, 1].")

    frequencies = [fundamental * k for k in range(1, num_harmonics + 1)]
    amplitudes = [amplitude * (harmonic_decay ** (k - 1)) for k in range(1, num_harmonics + 1)]

    t = _time_vector(sampling_rate, duration)
    signal = np.zeros_like(t)
    for f, a in zip(frequencies, amplitudes):
        signal += a * np.sin(2 * np.pi * f * t)

    return t, signal


def _generate_chirp(
    f0: float,
    f1: float,
    sampling_rate: float,
    duration: float,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear-sweep chirp from f0 Hz to f1 Hz over the given duration.

    Uses the standard instantaneous phase formula:
        phi(t) = 2*pi * (f0*t + (f1-f0)/(2*T) * t^2)
    """
    if f0 < 0 or f1 < 0:
        raise ValueError("Chirp frequencies must be non-negative.")
    if amplitude < 0:
        raise ValueError("Amplitude cannot be negative.")

    t = _time_vector(sampling_rate, duration)
    T = duration
    inst_phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * T) * t ** 2)
    signal = amplitude * np.sin(inst_phase)
    return t, signal


def _generate_amplitude_modulated(
    carrier_freq: float,
    modulation_freq: float,
    modulation_depth: float,
    sampling_rate: float,
    duration: float,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Amplitude-modulated signal.

    x(t) = A * [1 + m * cos(2*pi*f_mod*t)] * cos(2*pi*f_c*t)

    where m is modulation_depth in [0, 1].
    """
    if carrier_freq <= 0:
        raise ValueError("Carrier frequency must be positive.")
    if modulation_freq < 0:
        raise ValueError("Modulation frequency must be non-negative.")
    if not 0 <= modulation_depth <= 1:
        raise ValueError("modulation_depth must be in [0, 1].")

    t = _time_vector(sampling_rate, duration)
    envelope = 1.0 + modulation_depth * np.cos(2 * np.pi * modulation_freq * t)
    signal = amplitude * envelope * np.cos(2 * np.pi * carrier_freq * t)
    return t, signal


def _generate_frequency_modulated(
    carrier_freq: float,
    modulation_freq: float,
    modulation_index: float,
    sampling_rate: float,
    duration: float,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Frequency-modulated signal.

    x(t) = A * cos(2*pi*f_c*t + beta * sin(2*pi*f_mod*t))

    where beta = modulation_index (peak frequency deviation / f_mod).
    """
    if carrier_freq <= 0:
        raise ValueError("Carrier frequency must be positive.")
    if modulation_freq < 0:
        raise ValueError("Modulation frequency must be non-negative.")
    if modulation_index < 0:
        raise ValueError("modulation_index must be non-negative.")

    t = _time_vector(sampling_rate, duration)
    inst_phase = (
        2 * np.pi * carrier_freq * t
        + modulation_index * np.sin(2 * np.pi * modulation_freq * t)
    )
    signal = amplitude * np.cos(inst_phase)
    return t, signal


# ==================================================================
# Per-sample generator dispatcher
# ==================================================================


def _generate_one_sample(
    signal_type: str,
    config: AudioDatasetConfig,
    rng: np.random.Generator,
    sample_index: int,
) -> AudioSample:
    """
    Generate a single audio sample of the given type.

    Randomises parameters within sensible ranges using rng so that
    every sample in a type is meaningfully different while still being
    controlled by the global seed.
    """

    sample_id = str(uuid.uuid4())
    sr = config.sampling_rate
    dur = config.duration
    seed_snapshot = int(rng.integers(0, 2 ** 31))  # per-sample seed for metadata

    meta_kwargs: dict[str, Any] = {
        "sample_id": sample_id,
        "domain": "audio",
        "signal_type": signal_type,
        "sampling_rate": sr,
        "duration": dur,
        "num_samples": int(round(sr * dur)),
        "seed": seed_snapshot,
    }

    if signal_type == "clean_sine":
        # Vary frequency and amplitude slightly across samples
        freq = config.frequency * rng.uniform(0.8, 1.2)
        amp = config.amplitude * rng.uniform(0.7, 1.0)
        t, signal = generate_sine(
            frequency=freq,
            sampling_rate=sr,
            duration=dur,
            amplitude=amp,
        )
        meta_kwargs.update(frequency=float(freq), amplitude=float(amp))

    elif signal_type == "multi_tone":
        base = config.frequency * rng.uniform(0.8, 1.2)
        secondary = [f * rng.uniform(0.9, 1.1) for f in config.secondary_frequencies]
        all_freqs = [base] + secondary
        amps = [rng.uniform(0.5, 1.0) for _ in all_freqs]
        t, signal = generate_multi_tone(
            frequencies=all_freqs,
            sampling_rate=sr,
            duration=dur,
            amplitudes=amps,
        )
        meta_kwargs.update(
            frequencies=[float(f) for f in all_freqs],
            amplitudes=[float(a) for a in amps],
        )

    elif signal_type == "harmonic":
        fundamental = config.fundamental * rng.uniform(0.8, 1.2)
        num_h = config.num_harmonics
        decay = config.harmonic_decay * rng.uniform(0.8, 1.0)
        amp = rng.uniform(0.7, 1.0)
        t, signal = _generate_harmonic(
            fundamental=fundamental,
            sampling_rate=sr,
            duration=dur,
            num_harmonics=num_h,
            amplitude=amp,
            harmonic_decay=decay,
        )
        freqs = [fundamental * k for k in range(1, num_h + 1)]
        amps_list = [amp * (decay ** (k - 1)) for k in range(1, num_h + 1)]
        meta_kwargs.update(
            fundamental=float(fundamental),
            num_harmonics=num_h,
            harmonic_decay=float(decay),
            frequencies=[float(f) for f in freqs],
            amplitudes=[float(a) for a in amps_list],
        )

    elif signal_type == "chirp":
        f0 = config.chirp_f0 * rng.uniform(0.8, 1.2)
        f1 = config.chirp_f1 * rng.uniform(0.8, 1.2)
        amp = rng.uniform(0.7, 1.0)
        t, signal = _generate_chirp(
            f0=f0,
            f1=f1,
            sampling_rate=sr,
            duration=dur,
            amplitude=amp,
        )
        meta_kwargs.update(
            chirp_f0=float(f0),
            chirp_f1=float(f1),
            amplitude=float(amp),
        )

    elif signal_type == "amplitude_modulated":
        carrier = config.am_carrier_freq * rng.uniform(0.9, 1.1)
        mod_f = config.am_modulation_freq * rng.uniform(0.8, 1.2)
        depth = config.am_modulation_depth * rng.uniform(0.8, 1.0)
        amp = rng.uniform(0.7, 1.0)
        t, signal = _generate_amplitude_modulated(
            carrier_freq=carrier,
            modulation_freq=mod_f,
            modulation_depth=depth,
            sampling_rate=sr,
            duration=dur,
            amplitude=amp,
        )
        meta_kwargs.update(
            frequency=float(carrier),
            amplitude=float(amp),
            modulation_frequency=float(mod_f),
            modulation_depth=float(depth),
        )

    elif signal_type == "frequency_modulated":
        carrier = config.fm_carrier_freq * rng.uniform(0.9, 1.1)
        mod_f = config.fm_modulation_freq * rng.uniform(0.8, 1.2)
        index = config.fm_modulation_index * rng.uniform(0.8, 1.2)
        amp = rng.uniform(0.7, 1.0)
        t, signal = _generate_frequency_modulated(
            carrier_freq=carrier,
            modulation_freq=mod_f,
            modulation_index=index,
            sampling_rate=sr,
            duration=dur,
            amplitude=amp,
        )
        meta_kwargs.update(
            frequency=float(carrier),
            amplitude=float(amp),
            modulation_frequency=float(mod_f),
            modulation_index=float(index),
        )

    else:
        raise ValueError(f"Unsupported signal type: {signal_type!r}")

    # Validate result — never store NaN/Inf
    if not np.all(np.isfinite(signal)):
        raise RuntimeError(
            f"Generated signal for type '{signal_type}' contains non-finite values."
        )

    return AudioSample(
        signal=np.asarray(signal, dtype=np.float64),
        t=np.asarray(t, dtype=np.float64),
        meta=AudioSampleMetadata(**meta_kwargs),
    )


# ==================================================================
# Dataset class — stable interface for future replacement
# ==================================================================


class SyntheticAudioDataset:
    """
    Synthetic audio dataset.

    Interface contract
    ------------------
    generate() -> list[AudioSample]
        Generate all samples according to config.
    save(directory)
        Persist signals (.npy) and metadata (.json) to disk.
    load(directory) [classmethod]
        Restore previously saved dataset.
    metadata() -> list[dict]
        Return metadata for all samples as plain dicts.

    Future-compatibility note
    -------------------------
    This class can be replaced with ``RealAudioDataset`` that reads
    from a real corpus by implementing the same four methods.
    The rest of the pipeline never calls a synthetic-specific API.
    """

    def __init__(self, config: Optional[AudioDatasetConfig] = None):
        self.config = config or AudioDatasetConfig()
        self._samples: list[AudioSample] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self) -> list[AudioSample]:
        """
        Generate all samples deterministically from config.seed.

        Returns the list of AudioSample objects and caches them
        internally so save() / metadata() work without re-generating.
        """
        config = self.config
        self._validate_config(config)

        rng = np.random.default_rng(config.seed)
        samples: list[AudioSample] = []

        for signal_type in config.signal_types:
            if signal_type not in AUDIO_SIGNAL_TYPES:
                raise ValueError(
                    f"Unknown signal type: {signal_type!r}. "
                    f"Supported: {AUDIO_SIGNAL_TYPES}"
                )
            for i in range(config.samples_per_type):
                sample = _generate_one_sample(
                    signal_type=signal_type,
                    config=config,
                    rng=rng,
                    sample_index=i,
                )
                samples.append(sample)

        self._samples = samples
        return samples

    def save(self, directory: Optional[Path | str] = None) -> Path:
        """
        Save all samples to disk.

        Directory layout
        ----------------
        <directory>/
            <sample_id>.npy          — float64 signal array
            <sample_id>_t.npy        — float64 time vector
            <sample_id>_meta.json    — AudioSampleMetadata as JSON
            dataset_config.json      — AudioDatasetConfig

        Returns the resolved output directory path.
        """
        if not self._samples:
            raise RuntimeError(
                "No samples to save. Call generate() first."
            )

        out_dir = Path(directory) if directory else DEFAULT_AUDIO_DATASET_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        ordered_ids: list[str] = []
        for sample in self._samples:
            sid = sample.meta.sample_id
            ordered_ids.append(sid)
            np.save(out_dir / f"{sid}.npy", sample.signal)
            np.save(out_dir / f"{sid}_t.npy", sample.t)
            meta_path = out_dir / f"{sid}_meta.json"
            meta_path.write_text(
                json.dumps(asdict(sample.meta), indent=2),
                encoding="utf-8",
            )

        # Write an ordered index so load() can restore generation order.
        index_path = out_dir / "dataset_index.json"
        index_path.write_text(
            json.dumps({"ordered_ids": ordered_ids}, indent=2),
            encoding="utf-8",
        )

        config_path = out_dir / "dataset_config.json"
        config_path.write_text(
            json.dumps(asdict(self.config), indent=2),
            encoding="utf-8",
        )

        return out_dir

    @classmethod
    def load(cls, directory: Path | str) -> "SyntheticAudioDataset":
        """
        Restore a previously saved dataset from disk.

        Reads every *_meta.json sidecar in the directory,
        loads the corresponding .npy files, and re-populates _samples.
        """
        in_dir = Path(directory)
        if not in_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {in_dir}")

        # Restore config if present
        config_path = in_dir / "dataset_config.json"
        config = AudioDatasetConfig()
        if config_path.exists():
            raw_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config = AudioDatasetConfig(**raw_cfg)

        dataset = cls(config=config)
        samples: list[AudioSample] = []

        # Use the ordered index if present (preserves generation order).
        index_path = in_dir / "dataset_index.json"
        if index_path.exists():
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            ordered_ids = index_data["ordered_ids"]
        else:
            # Fallback: alphabetical sort (reproducible but not order-preserving)
            ordered_ids = [
                p.stem.replace("_meta", "")
                for p in sorted(in_dir.glob("*_meta.json"))
            ]

        for sid in ordered_ids:
            meta_path = in_dir / f"{sid}_meta.json"
            raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta = AudioSampleMetadata(**raw_meta)
            signal = np.load(in_dir / f"{sid}.npy")
            t_path = in_dir / f"{sid}_t.npy"
            t = np.load(t_path) if t_path.exists() else np.arange(len(signal)) / meta.sampling_rate
            samples.append(AudioSample(signal=signal, t=t, meta=meta))

        dataset._samples = samples
        return dataset

    def metadata(self) -> list[dict[str, Any]]:
        """Return metadata for all samples as plain JSON-serialisable dicts."""
        return [asdict(s.meta) for s in self._samples]

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)

    def __getitem__(self, index: int) -> AudioSample:
        return self._samples[index]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_config(config: AudioDatasetConfig) -> None:
        if config.samples_per_type < 1:
            raise ValueError("samples_per_type must be at least 1.")
        if config.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive.")
        if config.duration <= 0:
            raise ValueError("duration must be positive.")
        if not config.signal_types:
            raise ValueError("signal_types must not be empty.")
        if config.num_harmonics < 1:
            raise ValueError("num_harmonics must be at least 1.")
        if not 0 <= config.harmonic_decay <= 1:
            raise ValueError("harmonic_decay must be in [0, 1].")
        if not 0 <= config.am_modulation_depth <= 1:
            raise ValueError("am_modulation_depth must be in [0, 1].")


# ==================================================================
# Convenience factory — mirrors the build_noise_dataset() pattern
# ==================================================================


def build_audio_dataset(
    signal_types: Optional[list[str]] = None,
    samples_per_type: int = 5,
    sampling_rate: float = 44100.0,
    duration: float = 2.0,
    seed: int = 42,
) -> SyntheticAudioDataset:
    """
    Convenience factory for building a synthetic audio dataset.

    Mirrors the ``build_noise_dataset()`` pattern used by the
    intelligence.noise module so that callers have a consistent API.

    Parameters
    ----------
    signal_types:
        List of signal type names to generate.  Defaults to all
        supported types.
    samples_per_type:
        Number of samples to generate per type.
    sampling_rate:
        Audio sampling rate in Hz.
    duration:
        Duration of each sample in seconds.
    seed:
        Random seed for full reproducibility.

    Returns
    -------
    SyntheticAudioDataset
        A dataset object with _samples populated.
        Call .save() to persist to disk.
    """
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if samples_per_type < 1:
        raise ValueError("samples_per_type must be at least 1.")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")
    if duration <= 0:
        raise ValueError("duration must be positive.")

    config = AudioDatasetConfig(
        signal_types=signal_types or list(AUDIO_SIGNAL_TYPES),
        samples_per_type=samples_per_type,
        sampling_rate=sampling_rate,
        duration=duration,
        seed=seed,
    )

    dataset = SyntheticAudioDataset(config=config)
    dataset.generate()
    return dataset
