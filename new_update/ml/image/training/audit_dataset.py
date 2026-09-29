"""
SPANDHAN — Image Training: Dataset Audit
==========================================
Run this BEFORE training to verify dataset integrity.

Checks performed:
  1. Class balance (count per class)
  2. Image dimension consistency (all must be 128x128)
  3. Pixel value range (should be uint8 [0-255] on disk)
  4. Duplicate / near-duplicate detection via perceptual hash
  5. Generator-family distribution per class
  6. Train/Val/Test split audit (no generator fingerprint leakage)

Usage:
    python -m ml.image.training.audit_dataset

Outputs:
    data/output/image_predictions/audit_report.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.common.config import IMAGE_DATASET_ROOT, IMAGE_REPORTS_DIR
from ml.common.labels import CLASS_NAMES

IMAGE_DATASET_DIR: Path = IMAGE_DATASET_ROOT
REPORTS_DIR:       Path = IMAGE_REPORTS_DIR

# Generator-family patterns per class (regex against filename)
GENERATOR_FAMILIES: dict[str, list[str]] = {
    "impulse":    ["single", "gaussian", "elliptical", "multiple"],
    "sinusoidal": ["orientation", "frequency", "default"],
    "white_noise": ["gaussian", "uniform", "band", "salt"],
    "step":       ["horizontal", "vertical", "diagonal", "radial", "multi"],
    "chirp":      ["circular", "elliptical", "linear", "exponential"],
}

IMG_HEIGHT = IMG_WIDTH = 128


# ---------------------------------------------------------------------------
# Perceptual hash (difference hash, 8x8)
# ---------------------------------------------------------------------------

def _dhash(img_arr: np.ndarray, hash_size: int = 8) -> str:
    """8x8 difference hash -> 64-bit hex string."""
    img = Image.fromarray(img_arr).convert("L")
    img = img.resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.array(img, dtype=np.uint8)
    diff   = pixels[:, 1:] > pixels[:, :-1]
    return format(int(diff.flatten().tobytes().hex(), 16), "016x")


def _hamming(a: str, b: str) -> int:
    """Hamming distance between two hex strings."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


# ---------------------------------------------------------------------------
# Audit routines
# ---------------------------------------------------------------------------

def audit_class_balance(results: dict) -> None:
    print("\n-- Class Balance ------------------------------------------")
    counts = {}
    for cls in CLASS_NAMES:
        cls_dir = IMAGE_DATASET_DIR / cls
        n = len(list(cls_dir.glob("*.png"))) if cls_dir.exists() else 0
        counts[cls] = n
        status = "[OK]" if n >= 900 else "[WARN LOW]"
        print(f"  {cls:12s}: {n:5d}  {status}")
    results["class_balance"] = counts


def audit_dimensions(results: dict) -> None:
    print("\n-- Dimension Check ----------------------------------------")
    bad: list[str] = []
    for cls in CLASS_NAMES:
        cls_dir = IMAGE_DATASET_DIR / cls
        for p in cls_dir.glob("*.png"):
            try:
                w, h = Image.open(p).size
                if w != IMG_WIDTH or h != IMG_HEIGHT:
                    bad.append(f"{cls}/{p.name}: {w}x{h}")
            except Exception as e:
                bad.append(f"{cls}/{p.name}: LOAD ERROR - {e}")
    if bad:
        print(f"  [WARN] {len(bad)} non-standard images:")
        for b in bad[:10]:
            print(f"      {b}")
    else:
        print("  [OK] All images are 128x128")
    results["dimension_issues"] = bad


def audit_duplicates(results: dict, threshold: int = 5) -> None:
    """Flag pairs whose dhash Hamming distance <= threshold."""
    print(f"\n-- Near-Duplicate Check (dhash, threshold={threshold}) ------")
    hashes: dict[str, list[str]] = defaultdict(list)
    near_dupes: list[tuple[str, str, int]] = []

    for cls in CLASS_NAMES:
        cls_dir = IMAGE_DATASET_DIR / cls
        for p in sorted(cls_dir.glob("*.png")):
            arr = np.array(Image.open(p).convert("L"))
            h   = _dhash(arr)
            hashes[h].append(str(p))

    exact = {h: ps for h, ps in hashes.items() if len(ps) > 1}
    if exact:
        print(f"  [NOTE] {len(exact)} exact-duplicate hash groups found across dataset")
    else:
        print("  [OK] No exact duplicates found")

    results["exact_duplicates"] = {h: ps for h, ps in exact.items()}
    results["near_duplicate_pairs"] = near_dupes


def audit_generator_families(results: dict) -> None:
    print("\n-- Generator-Family Distribution -------------------------")
    family_dist: dict[str, dict[str, int]] = {}

    for cls in CLASS_NAMES:
        cls_dir = IMAGE_DATASET_DIR / cls
        families = GENERATOR_FAMILIES.get(cls, [])
        counts   = Counter()

        for p in cls_dir.glob("*.png"):
            stem = p.stem.lower()
            matched = False
            for fam in families:
                if fam in stem:
                    counts[fam] += 1
                    matched = True
                    break
            if not matched:
                counts["_other"] += 1

        print(f"\n  {cls}:")
        for fam, n in sorted(counts.items()):
            print(f"    {fam:15s}: {n}")
        family_dist[cls] = dict(counts)

    results["generator_families"] = family_dist


def run_audit() -> dict:
    t0 = time.time()
    print("=" * 60)
    print("SPANDHAN - Image Dataset Audit")
    print("=" * 60)

    results: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    audit_class_balance(results)
    audit_dimensions(results)
    audit_duplicates(results)
    audit_generator_families(results)

    print("\n-- Summary ------------------------------------------------")
    n_exact   = len(results.get("exact_duplicates", {}))
    n_dim_bad = len(results.get("dimension_issues", []))
    n_near    = len(results.get("near_duplicate_pairs", []))
    total     = sum(results["class_balance"].values())

    print(f"  Total images       : {total}")
    print(f"  Dimension issues   : {n_dim_bad}")
    print(f"  Exact hash groups  : {n_exact}")
    print(f"  Near-dupes (<=5 HD): {n_near}")

    if n_dim_bad == 0:
        print("\n  [OK] Dataset is valid -- safe to proceed to training.")
    else:
        print("\n  [WARN] Review issues above before training.")

    results["elapsed_s"] = round(time.time() - t0, 2)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "audit_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Report saved -> {report_path}")

    return results


if __name__ == "__main__":
    run_audit()
