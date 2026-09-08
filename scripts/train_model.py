"""
scripts/train_model.py
=======================
Load the generated dataset, train the SVM noise classifier,
and save the model to models/noise_classifier.joblib.

Usage
-----
    python scripts/train_model.py [--dataset PATH] [--output PATH] [--seed N]

Prerequisites
-------------
    Run scripts/generate_dataset.py first to produce datasets/ml/noise_features.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intelligence.noise.dataset_builder import NoiseDataset, NOISE_CLASSES
from intelligence.noise.dataset import split_noise_dataset
from intelligence.noise.trainer import train_svm
from intelligence.noise.model_io import save_noise_model


def _load_npz(npz_path: Path) -> NoiseDataset:
    """Load the .npz feature matrix saved by generate_dataset.py."""
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(str)
    feature_names = list(data["feature_names"])

    # Rebuild metadata stubs (not stored in .npz — not needed for training)
    metadata = [{"sample_index": i} for i in range(len(y))]

    return NoiseDataset(X=X, y=y, feature_names=feature_names, metadata=metadata)


def train(
    dataset_path: str = "datasets/ml/noise_features.npz",
    output_path: str = "models/noise_classifier.joblib",
    seed: int = 42,
) -> None:
    print("=" * 60)
    print("  SPANDHAN — Model Trainer")
    print("=" * 60)

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    # -- 1. Load dataset ------------------------------------------------------
    npz_path = PROJECT_ROOT / dataset_path
    if not npz_path.exists():
        print(f"\n  ERROR: Dataset not found at {npz_path}")
        print("  Run: python scripts/generate_dataset.py\n")
        sys.exit(1)

    print(">> Loading dataset from " + dataset_path + " ...")
    dataset = _load_npz(npz_path)
    print(f"   X shape      : {dataset.X.shape}")
    print(f"   Classes      : {sorted(set(dataset.y))}")

    # Class counts
    for cls in NOISE_CLASSES:
        n = int(np.sum(dataset.y == cls))
        print(f"   {cls:12s}: {n} samples")

    # -- 2. Split -------------------------------------------------------------
    print()
    print(">> Splitting dataset (70 / 15 / 15) ...")
    split = split_noise_dataset(dataset, test_size=0.15, val_size=0.15, random_state=seed)
    print(f"   Train : {split.X_train.shape[0]}")
    print(f"   Val   : {split.X_val.shape[0]}")
    print(f"   Test  : {split.X_test.shape[0]}")

    # -- 3. Train -------------------------------------------------------------
    print()
    print(">> Training SVM (GridSearchCV, RBF kernel) ... this may take a minute ...")
    t0 = time.perf_counter()
    result = train_svm(split, random_state=seed)
    elapsed = time.perf_counter() - t0

    print(f"   Done in {elapsed:.1f}s")
    print(f"   Best C         : {result.best_params.get('C')}")
    print(f"   Best gamma     : {result.best_params.get('gamma')}")
    print(f"   Train accuracy : {result.training_accuracy:.4f} ({result.training_accuracy*100:.1f}%)")
    print(f"   Val   accuracy : {result.validation_accuracy:.4f} ({result.validation_accuracy*100:.1f}%)")

    # -- 4. Test set evaluation -----------------------------------------------
    test_acc = float(result.model.score(split.X_test, split.y_test))
    print(f"   Test  accuracy : {test_acc:.4f} ({test_acc*100:.1f}%)")

    # -- 5. Save model --------------------------------------------------------
    print()
    out_path = PROJECT_ROOT / output_path
    print(">> Saving model -> " + output_path + " ...")
    saved = save_noise_model(result, out_path)
    size_kb = saved.stat().st_size / 1024
    print(f"   Saved ({size_kb:.1f} KB)")

    # -- 6. Summary -----------------------------------------------------------
    print()
    print("=" * 60)
    print("  Training COMPLETE")
    print("  Model saved  : " + output_path)
    print(f"  Train acc    : {result.training_accuracy*100:.1f}%")
    print(f"  Val   acc    : {result.validation_accuracy*100:.1f}%")
    print(f"  Test  acc    : {test_acc*100:.1f}%")
    print("=" * 60)
    print()
    print("  Next step: python run.py  →  ML Studio page")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Spandhan noise classifier")
    parser.add_argument("--dataset", default="datasets/ml/noise_features.npz",
                        help="Path to the .npz feature matrix (relative to project root)")
    parser.add_argument("--output", default="models/noise_classifier.joblib",
                        help="Output model path (default: models/noise_classifier.joblib)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    train(dataset_path=args.dataset, output_path=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
