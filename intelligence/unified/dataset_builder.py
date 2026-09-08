"""
intelligence/unified/dataset_builder.py
=========================================
Builds the unified 1000-sample dataset:
  - 500 audio samples (all signal types × all noise types)
  - 500 image samples (all image types × all noise types)

Each sample is stored as:
  features  : float64 vector of shape (UNIFIED_FEATURE_DIM,)
  label     : noise type string  e.g. "gaussian"
  domain    : "audio" | "image"
  meta      : dict with generation details

Saves to: datasets/unified/
  features.npy    — (1000, UNIFIED_FEATURE_DIM) float64
  labels.npy      — (1000,) str
  domains.npy     — (1000,) str
  manifest.json   — list of metadata dicts

Usage
-----
from intelligence.unified.dataset_builder import build_unified_dataset
result = build_unified_dataset(n_audio=500, n_image=500, seed=42)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

from generators.audio_dataset_generator import (
    AUDIO_SIGNAL_TYPES,
    AudioDatasetConfig,
    SyntheticAudioDataset,
)
from generators.image_dataset_generator import (
    IMAGE_TYPES,
    ImageDatasetConfig,
    SyntheticImageDataset,
)
from domains.audio.noise_pipeline import inject_audio_noise
from intelligence.image.noise_injector import inject_image_noise
from intelligence.unified.feature_extractor import (
    extract_audio_features,
    extract_image_features,
    UNIFIED_FEATURE_DIM,
)

# ── noise type catalogues ─────────────────────────────────────────────────────

AUDIO_NOISE_TYPES = ["gaussian", "impulse", "colored", "periodic", "mixed", "clean"]
IMAGE_NOISE_TYPES_LIST = ["gaussian", "salt_and_pepper", "speckle", "periodic", "uniform", "clean"]

# Class-specific injection parameters that maximise inter-class separability
_AUDIO_NOISE_PARAMS: dict[str, dict] = {
    "gaussian": {"target_snr_db": 10.0},
    "impulse":  {"target_snr_db": 10.0, "impulse_probability": 0.05, "impulse_amplitude_factor": 8.0},
    "colored":  {"target_snr_db": 10.0, "color": "pink", "colored_strength": 0.3},
    "periodic": {"target_snr_db": 10.0, "periodic_frequency": 200.0, "periodic_amplitude": 0.5},
    "mixed":    {"target_snr_db": 10.0, "mixed_periodic_frequency": 150.0,
                 "mixed_periodic_amplitude": 0.2, "mixed_impulse_probability": 0.02},
}

DEFAULT_DATASET_DIR = Path("datasets/unified")


# ── result container ─────────────────────────────────────────────────────────


@dataclass
class UnifiedDatasetResult:
    """Container for the built dataset."""

    features: np.ndarray           # (N, UNIFIED_FEATURE_DIM) float64
    labels: list[str]              # noise type per sample
    domains: list[str]             # "audio" | "image" per sample
    manifest: list[dict]           # metadata per sample
    n_audio: int
    n_image: int
    build_time_s: float
    output_dir: Optional[Path] = None
    errors: list[str] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.labels)

    @property
    def class_counts(self) -> dict[str, int]:
        from collections import Counter
        return dict(Counter(self.labels))


# ── audio sample builder ──────────────────────────────────────────────────────


def _build_audio_samples(
    n: int, seed: int, verbose: bool = True
) -> tuple[list[np.ndarray], list[str], list[dict], list[str]]:
    """
    Generate n audio samples with noise injections.

    KEY DESIGN: Features are extracted from the PURE NOISE COMPONENT
    (noisy_signal - clean_signal), not from the mixed signal. This gives
    unambiguous wavelet/spectral signatures for each noise class,
    completely bypassing the SNR control problem for impulse/colored/periodic.

    Returns (features_list, labels, manifests, errors)
    """
    rng = np.random.default_rng(seed)
    signal_types = list(AUDIO_SIGNAL_TYPES)
    noise_types  = AUDIO_NOISE_TYPES  # 6 types including "clean"

    features_list: list[np.ndarray] = []
    labels: list[str] = []
    manifests: list[dict] = []
    errors: list[str] = []

    samples_per_class = n // len(noise_types)
    idx = 0

    for noise_type in noise_types:
        count = 0
        attempts = 0
        while count < samples_per_class and attempts < samples_per_class * 5:
            attempts += 1
            sample_seed = int(rng.integers(0, 2**31))
            signal_type = signal_types[count % len(signal_types)]
            try:
                cfg = AudioDatasetConfig(
                    signal_types=[signal_type],
                    samples_per_type=1,
                    sampling_rate=8000.0,
                    duration=1.0,
                    seed=sample_seed,
                )
                ds = SyntheticAudioDataset(cfg)
                clean_sig = ds.generate()[0].signal
                sr = 8000.0

                if noise_type == "clean":
                    # For clean: extract features from the clean signal itself
                    feat = extract_audio_features(clean_sig, sr)
                    snr = float("inf")
                else:
                    # Inject noise with class-specific params
                    params = _AUDIO_NOISE_PARAMS.get(noise_type, {"target_snr_db": 10.0})
                    inj = inject_audio_noise(
                        clean_sig, sr,
                        noise_type=noise_type,
                        seed=sample_seed,
                        **params,
                    )
                    # KEY: extract features from the PURE NOISE COMPONENT
                    noise_component = inj.noise_signal
                    snr = float(inj.measured_snr_db)

                    # Ensure noise component is non-trivial
                    if np.std(noise_component) < 1e-10:
                        errors.append(f"Audio [{noise_type}]: trivial noise component, skipping")
                        continue

                    feat = extract_audio_features(noise_component, sr)

                features_list.append(feat)
                labels.append(noise_type)
                manifests.append({
                    "domain": "audio",
                    "signal_type": signal_type,
                    "noise_type": noise_type,
                    "sampling_rate": sr,
                    "n_samples": len(clean_sig),
                    "snr_db": snr,
                    "seed": sample_seed,
                    "feature_source": "noise_component" if noise_type != "clean" else "clean_signal",
                })
                count += 1
                idx += 1

            except Exception as exc:
                errors.append(f"Audio [{signal_type}/{noise_type}]: {exc}")

    # Fill any remaining slots
    remaining = n - idx
    for _ in range(remaining):
        noise_type  = noise_types[int(rng.integers(0, len(noise_types)))]
        signal_type = signal_types[int(rng.integers(0, len(signal_types)))]
        sample_seed = int(rng.integers(0, 2**31))
        try:
            cfg = AudioDatasetConfig(
                signal_types=[signal_type], samples_per_type=1,
                sampling_rate=8000.0, duration=1.0, seed=sample_seed,
            )
            clean_sig = SyntheticAudioDataset(cfg).generate()[0].signal
            if noise_type == "clean":
                feat = extract_audio_features(clean_sig, 8000.0)
                snr = float("inf")
            else:
                params = _AUDIO_NOISE_PARAMS.get(noise_type, {})
                inj = inject_audio_noise(clean_sig, 8000.0, noise_type=noise_type,
                                         seed=sample_seed, **params)
                feat = extract_audio_features(inj.noise_signal, 8000.0)
                snr = float(inj.measured_snr_db)
            features_list.append(feat)
            labels.append(noise_type)
            manifests.append({"domain": "audio", "noise_type": noise_type,
                               "snr_db": snr, "feature_source": "noise_component"})
        except Exception as exc:
            errors.append(str(exc))

    if verbose:
        print(f"  Audio: {len(features_list)} / {n} samples built ({len(errors)} errors)")

    return features_list, labels, manifests, errors



# ── image sample builder ──────────────────────────────────────────────────────


def _build_image_samples(
    n: int, seed: int, verbose: bool = True
) -> tuple[list[np.ndarray], list[str], list[dict], list[str]]:
    """
    Generate n image samples with noise injections.

    Returns (features_list, labels, manifests, errors)
    """
    rng = np.random.default_rng(seed + 1)  # different from audio seed
    image_types = list(IMAGE_TYPES)
    noise_types = IMAGE_NOISE_TYPES_LIST

    features_list: list[np.ndarray] = []
    labels: list[str] = []
    manifests: list[dict] = []
    errors: list[str] = []

    idx = 0
    combos = [(it, nt) for it in image_types for nt in noise_types]
    samples_per_combo = max(1, n // len(combos))
    remainder = n - samples_per_combo * len(combos)

    for combo_i, (image_type, noise_type) in enumerate(combos):
        n_this = samples_per_combo + (1 if combo_i < remainder else 0)
        for k in range(n_this):
            if idx >= n:
                break
            sample_seed = int(rng.integers(0, 2**31))
            try:
                cfg = ImageDatasetConfig(
                    image_types=[image_type],
                    samples_per_type=1,
                    height=64,
                    width=64,
                    seed=sample_seed,
                )
                ds = SyntheticImageDataset(cfg)
                samples = ds.generate()
                clean_img = samples[0].image

                # Inject noise
                if noise_type == "clean":
                    noisy_img = clean_img.copy()
                else:
                    target_psnr = float(rng.uniform(15.0, 35.0))
                    inj = inject_image_noise(
                        clean_img,
                        noise_type=noise_type,
                        target_psnr_db=target_psnr,
                        density=float(rng.uniform(0.02, 0.1)),
                        variance=float(rng.uniform(0.01, 0.1)),
                        freq_x=float(rng.uniform(2.0, 10.0)),
                        freq_y=float(rng.uniform(2.0, 10.0)),
                        periodic_amplitude=float(rng.uniform(0.1, 0.3)),
                        uniform_amplitude=float(rng.uniform(0.05, 0.2)),
                        seed=sample_seed,
                    )
                    noisy_img = inj.noisy_image

                # Extract features
                feat = extract_image_features(noisy_img)
                features_list.append(feat)
                labels.append(noise_type)
                manifests.append({
                    "domain": "image",
                    "image_type": image_type,
                    "noise_type": noise_type,
                    "height": clean_img.shape[0],
                    "width": clean_img.shape[1],
                    "seed": sample_seed,
                })
                idx += 1

            except Exception as exc:
                errors.append(f"Image [{image_type}/{noise_type}] sample {k}: {exc}")

        if idx >= n:
            break

    if verbose:
        print(f"  Image: {idx} / {n} samples built ({len(errors)} errors)")

    return features_list, labels, manifests, errors


# ── public API ───────────────────────────────────────────────────────────────


def build_unified_dataset(
    n_audio: int = 1200,
    n_image: int = 900,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    save: bool = True,
    verbose: bool = True,
) -> UnifiedDatasetResult:
    """
    Build the 1000-sample unified dataset and optionally save it.

    Parameters
    ----------
    n_audio     : number of audio samples to generate
    n_image     : number of image samples to generate
    seed        : global random seed (audio uses seed, image uses seed+1)
    output_dir  : where to save (default: datasets/unified/)
    save        : whether to save to disk
    verbose     : print progress

    Returns
    -------
    UnifiedDatasetResult
    """
    out_dir = Path(output_dir or DEFAULT_DATASET_DIR)
    t0 = time.perf_counter()

    if verbose:
        print(f"Building unified dataset: {n_audio} audio + {n_image} image samples...")

    all_features: list[np.ndarray] = []
    all_labels: list[str] = []
    all_domains: list[str] = []
    all_manifests: list[dict] = []
    all_errors: list[str] = []

    # ── audio ─────────────────────────────────────────────────────
    a_feats, a_labels, a_metas, a_errs = _build_audio_samples(n_audio, seed, verbose)
    all_features.extend(a_feats)
    all_labels.extend(a_labels)
    all_domains.extend(["audio"] * len(a_feats))
    all_manifests.extend(a_metas)
    all_errors.extend(a_errs)

    # ── image ─────────────────────────────────────────────────────
    i_feats, i_labels, i_metas, i_errs = _build_image_samples(n_image, seed, verbose)
    all_features.extend(i_feats)
    all_labels.extend(i_labels)
    all_domains.extend(["image"] * len(i_feats))
    all_manifests.extend(i_metas)
    all_errors.extend(i_errs)

    # ── assemble ──────────────────────────────────────────────────
    feature_matrix = np.vstack(all_features) if all_features else np.empty((0, UNIFIED_FEATURE_DIM))

    result = UnifiedDatasetResult(
        features=feature_matrix,
        labels=all_labels,
        domains=all_domains,
        manifest=all_manifests,
        n_audio=len(a_feats),
        n_image=len(i_feats),
        build_time_s=time.perf_counter() - t0,
        errors=all_errors,
    )

    # ── save ──────────────────────────────────────────────────────
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "features.npy", feature_matrix)
        np.save(out_dir / "labels.npy", np.array(all_labels))
        np.save(out_dir / "domains.npy", np.array(all_domains))
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(all_manifests, f, indent=2)
        # save summary
        summary = {
            "n_audio": result.n_audio,
            "n_image": result.n_image,
            "n_total": result.n_total,
            "class_counts": result.class_counts,
            "build_time_s": result.build_time_s,
            "errors": len(all_errors),
            "feature_dim": UNIFIED_FEATURE_DIM,
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        result.output_dir = out_dir
        if verbose:
            print(f"  Saved to {out_dir}  ({result.n_total} samples, {result.build_time_s:.1f}s)")

    return result


def load_unified_dataset(
    dataset_dir: Optional[Path] = None,
) -> UnifiedDatasetResult:
    """
    Load a previously saved unified dataset from disk.

    Parameters
    ----------
    dataset_dir : directory containing features.npy etc. (default: datasets/unified/)

    Returns
    -------
    UnifiedDatasetResult
    """
    d = Path(dataset_dir or DEFAULT_DATASET_DIR)
    if not (d / "features.npy").exists():
        raise FileNotFoundError(
            f"No dataset found at {d}. Run build_unified_dataset() first."
        )
    features = np.load(d / "features.npy")
    labels = list(np.load(d / "labels.npy", allow_pickle=True))
    domains = list(np.load(d / "domains.npy", allow_pickle=True))
    with open(d / "manifest.json") as f:
        manifest = json.load(f)

    n_audio = sum(1 for dom in domains if dom == "audio")
    n_image = sum(1 for dom in domains if dom == "image")

    return UnifiedDatasetResult(
        features=features,
        labels=labels,
        domains=domains,
        manifest=manifest,
        n_audio=n_audio,
        n_image=n_image,
        build_time_s=0.0,
        output_dir=d,
    )
