"""
scripts/create_unified_dataset.py
====================================
CLI script — generate the 1000-sample unified dataset ONCE.

Usage
-----
    python scripts/create_unified_dataset.py
    python scripts/create_unified_dataset.py --n-audio 500 --n-image 500 --seed 42
    python scripts/create_unified_dataset.py --output-dir datasets/unified
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so imports work when called from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intelligence.unified.dataset_builder import build_unified_dataset, DEFAULT_DATASET_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the 1000-sample unified audio+image dataset for Spandhan."
    )
    parser.add_argument("--n-audio", type=int, default=500, help="Number of audio samples")
    parser.add_argument("--n-image", type=int, default=500, help="Number of image samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Output directory (default: datasets/unified/)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Spandhan — Unified Dataset Creator")
    print("=" * 60)
    print(f"  Audio samples : {args.n_audio}")
    print(f"  Image samples : {args.n_image}")
    print(f"  Seed          : {args.seed}")
    print(f"  Output dir    : {args.output_dir}")
    print("=" * 60)

    result = build_unified_dataset(
        n_audio=args.n_audio,
        n_image=args.n_image,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        save=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("  Dataset Summary")
    print("=" * 60)
    print(f"  Total samples : {result.n_total}")
    print(f"  Audio         : {result.n_audio}")
    print(f"  Image         : {result.n_image}")
    print(f"  Feature dim   : {result.features.shape[1]}")
    print(f"  Build time    : {result.build_time_s:.1f}s")
    print(f"  Errors        : {len(result.errors)}")
    print("\n  Class distribution:")
    for cls, cnt in sorted(result.class_counts.items()):
        bar = "#" * (cnt // 5)
        print(f"    {cls:20s}: {cnt:4d}  {bar}")

    if result.errors:
        print(f"\n  First 5 errors:")
        for err in result.errors[:5]:
            print(f"    {err}")

    print("\n  Dataset ready at:", result.output_dir)
    print("  Next: python scripts/train_unified_model.py")


if __name__ == "__main__":
    main()
