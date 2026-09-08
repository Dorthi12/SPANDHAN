"""
scripts/train_unified_model.py
================================
CLI script — train the unified noise classifier on the saved dataset.

Usage
-----
    python scripts/train_unified_model.py
    python scripts/train_unified_model.py --dataset-dir datasets/unified
    python scripts/train_unified_model.py --n-estimators 200 --test-size 0.2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intelligence.unified.trainer import train_unified_model
from intelligence.unified.dataset_builder import DEFAULT_DATASET_DIR
from intelligence.unified.model_io import DEFAULT_MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Spandhan unified noise classifier."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Path to the unified dataset directory",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Where to save the trained model",
    )
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Force re-generate dataset even if it already exists",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Spandhan — Unified Model Trainer")
    print("=" * 60)
    print(f"  Dataset dir   : {args.dataset_dir}")
    print(f"  Model output  : {args.model_path}")
    print(f"  RF estimators : {args.n_estimators}")
    print(f"  Test split    : {args.test_size:.0%}")
    print("=" * 60)

    # Optionally rebuild dataset first
    if args.rebuild_dataset:
        from intelligence.unified.dataset_builder import build_unified_dataset
        print("\nRebuilding dataset…")
        build_unified_dataset(
            n_audio=500, n_image=500, seed=args.seed,
            output_dir=Path(args.dataset_dir), save=True, verbose=True
        )

    print("\nTraining…")
    bundle = train_unified_model(
        dataset_dir=Path(args.dataset_dir),
        model_path=Path(args.model_path),
        test_size=args.test_size,
        random_state=args.seed,
        n_estimators=args.n_estimators,
        save=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("  Training Results")
    print("=" * 60)
    print(f"  Train accuracy : {bundle.train_accuracy:.4f}")
    print(f"  Val accuracy   : {bundle.val_accuracy:.4f}")
    print(f"  Classes        : {bundle.class_names}")
    print(f"  Train samples  : {bundle.n_train}")
    print(f"  Val samples    : {bundle.n_val}")
    print("\n  Classification Report:")
    for line in bundle.classification_report.strip().splitlines():
        print(f"    {line}")

    if bundle.feature_importances is not None and bundle.feature_names:
        top_n = 10
        idxs = bundle.feature_importances.argsort()[-top_n:][::-1]
        print(f"\n  Top {top_n} most important features:")
        names = bundle.feature_names
        for i in idxs:
            if i < len(names):
                print(f"    {names[i]:20s}: {bundle.feature_importances[i]:.4f}")

    print(f"\n  Model saved -> {args.model_path}")
    print("  App -> Domain -> ML page to run inference on uploaded files.")


if __name__ == "__main__":
    main()
