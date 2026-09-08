"""
scripts/generate_dataset.py
============================
Generate the Spandhan noise-classification dataset and save it to disk.

Usage
-----
    python scripts/generate_dataset.py [--samples N] [--seed S] [--no-wav]

Outputs
-------
    datasets/ml/noise_features.npz        — feature matrix + labels
    datasets/ml/dataset_metadata.json     — class counts, feature names, config
    datasets/synthetic/audio/<class>/     — one .wav per sample (unless --no-wav)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.noise.dataset_builder import build_noise_dataset, NOISE_CLASSES
from intelligence.noise.dataset import split_noise_dataset


def _save_wav(signal: np.ndarray, sampling_rate: float, path: Path) -> None:
    """Save a 1-D float64 array as a 16-bit PCM WAV file (no extra deps)."""
    import struct, wave

    path.parent.mkdir(parents=True, exist_ok=True)
    # Normalise to [-1, 1] then quantise to int16
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal_norm = signal / peak
    else:
        signal_norm = signal

    samples_int16 = (signal_norm * 32767).astype(np.int16)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(int(sampling_rate))
        wf.writeframes(samples_int16.tobytes())


def generate(samples_per_class: int = 50, seed: int = 42, save_wav: bool = True) -> None:
    print("=" * 60)
    print("  SPANDHAN — Dataset Generator")
    print("=" * 60)
    print(f"  Noise classes  : {list(NOISE_CLASSES)}")
    print(f"  Samples/class  : {samples_per_class}")
    print(f"  Total samples  : {len(NOISE_CLASSES) * samples_per_class}")
    print(f"  Seed           : {seed}")
    print(f"  Save WAV files : {save_wav}")
    print()

    # -- 1. Build feature dataset --------------------------------------------
    print(">> Building noise feature dataset ...", flush=True)
    t0 = time.perf_counter()

    dataset = build_noise_dataset(
        samples_per_class=samples_per_class,
        length=2000,
        sampling_rate=1000,
        seed=seed,
    )

    elapsed = time.perf_counter() - t0
    print(f"   Done in {elapsed:.1f}s  |  X shape: {dataset.X.shape}")

    # -- 2. Save .npz feature matrix -----------------------------------------
    ml_dir = PROJECT_ROOT / "datasets" / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)

    npz_path = ml_dir / "noise_features.npz"
    np.savez_compressed(
        npz_path,
        X=dataset.X,
        y=dataset.y,
        feature_names=np.array(dataset.feature_names),
    )
    print("   Saved feature matrix -> " + str(npz_path.relative_to(PROJECT_ROOT)))

    # -- 3. Save metadata JSON -----------------------------------------------
    class_counts = {cls: int(np.sum(dataset.y == cls)) for cls in NOISE_CLASSES}
    metadata = {
        "samples_per_class": samples_per_class,
        "total_samples": len(dataset.y),
        "seed": seed,
        "sampling_rate": 1000,
        "signal_length": 2000,
        "noise_classes": list(NOISE_CLASSES),
        "class_counts": class_counts,
        "feature_names": dataset.feature_names,
        "feature_count": len(dataset.feature_names),
    }

    meta_path = ml_dir / "dataset_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("   Saved metadata       -> " + str(meta_path.relative_to(PROJECT_ROOT)))

    # -- 4. Print split preview ----------------------------------------------
    print()
    print(">> Computing train/val/test split preview ...")
    split = split_noise_dataset(dataset, test_size=0.15, val_size=0.15, random_state=seed)
    print(f"   Train : {split.X_train.shape[0]} samples")
    print(f"   Val   : {split.X_val.shape[0]} samples")
    print(f"   Test  : {split.X_test.shape[0]} samples")

    # -- 5. Save synthetic WAV files -----------------------------------------
    if save_wav:
        print()
        print(">> Generating synthetic WAV files ...")
        from generators.noise_generator import (
            add_gaussian_noise, add_impulse_noise, add_periodic_noise,
            add_colored_noise, add_mixed_noise,
        )
        from generators.waveform_generator import generate_sine

        sr = 1000
        duration = 2.0
        length = int(sr * duration)
        rng = np.random.default_rng(seed + 999)

        noise_fns = {
            "Clean": lambda s: s.copy(),
            "Gaussian": lambda s: add_gaussian_noise(s, snr_db=15.0, random_state=rng),
            "Impulse": lambda s: add_impulse_noise(s, probability=0.005, random_state=rng),
            "Periodic": lambda s: add_periodic_noise(s, sampling_rate=sr, frequency=50.0, amplitude=0.2),
            "Colored": lambda s: add_colored_noise(s, color="pink", strength=0.2, random_state=rng),
            "Mixed": lambda s: add_mixed_noise(s, sampling_rate=sr, snr_db=15.0, periodic_frequency=50.0, random_state=rng),
        }

        wav_base = PROJECT_ROOT / "datasets" / "synthetic" / "audio"
        total_wav = 0

        for cls_name, noise_fn in noise_fns.items():
            cls_dir = wav_base / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)

            for i in range(samples_per_class):
                freq = rng.uniform(80, 400)
                amp = rng.uniform(0.5, 1.0)
                _, clean = generate_sine(frequency=freq, sampling_rate=sr, duration=duration, amplitude=amp)
                noisy = noise_fn(clean)
                wav_path = cls_dir / f"{cls_name.lower()}_{i:03d}.wav"
                _save_wav(noisy, sr, wav_path)
                total_wav += 1

            print(f"   {cls_name:10s} -> {cls_dir.relative_to(PROJECT_ROOT)} ({samples_per_class} files)")

        print(f"   Total WAV files saved: {total_wav}")

    # -- 6. Summary ----------------------------------------------------------
    print()
    print("=" * 60)
    print("  Dataset generation COMPLETE")
    print("  Feature matrix : datasets/ml/noise_features.npz")
    print("  Metadata       : datasets/ml/dataset_metadata.json")
    if save_wav:
        print("  WAV files      : datasets/synthetic/audio/<class>/")
    print("=" * 60)
    print()
    print("  Next step: python scripts/train_model.py")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Spandhan noise dataset")
    parser.add_argument("--samples", type=int, default=50,
                        help="Samples per noise class (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-wav", action="store_true",
                        help="Skip saving synthetic WAV files")
    args = parser.parse_args()

    generate(
        samples_per_class=args.samples,
        seed=args.seed,
        save_wav=not args.no_wav,
    )


if __name__ == "__main__":
    main()
