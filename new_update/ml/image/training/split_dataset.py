"""
SPANDHAN — Image Training: Stratified Dataset Split
=====================================================
Splits (X, y) arrays into TRAIN / VAL / TEST with guaranteed
per-class proportions:

    TRAIN  70%  →  700 per class  →  3500 total
    VAL    15%  →  150 per class  →   750 total
    TEST   15%  →  150 per class  →   750 total

The split is:
  • Stratified  — every class appears in correct proportions
  • Reproducible — controlled by RANDOM_STATE
  • Auditable   — returns index arrays so filenames can be inspected

Usage:
    from ml.image.training.split_dataset import make_splits
    (X_tr, X_val, X_te,
     y_tr, y_val, y_te,
     idx_tr, idx_val, idx_te) = make_splits(X, y)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RANDOM_STATE: int  = 42
TRAIN_RATIO:  float = 0.70
VAL_RATIO:    float = 0.15   # of total
TEST_RATIO:   float = 0.15   # of total


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Returns
    -------
    X_train, X_val, X_test  : feature arrays
    y_train, y_val, y_test  : label arrays
    idx_train, idx_val, idx_test : original indices (for filename auditing)
    """
    n = len(y)
    indices = np.arange(n)

    # Step 1: carve off TEST  (15% of total)
    # Val-ratio relative to the remaining 85%: 15/85 ≈ 0.1765
    val_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

    idx_trainval, idx_test = train_test_split(
        indices,
        test_size=TEST_RATIO,
        stratify=y,
        random_state=random_state,
    )

    # Step 2: carve off VAL from train+val
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_relative,
        stratify=y[idx_trainval],
        random_state=random_state,
    )

    X_train = X[idx_train]
    X_val   = X[idx_val]
    X_test  = X[idx_test]
    y_train = y[idx_train]
    y_val   = y[idx_val]
    y_test  = y[idx_test]

    if verbose:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        from ml.common.labels import ID_TO_CLASS
        print("\n-- Dataset Split ------------------------------------------")
        print(f"  TRAIN : {len(y_train):5d}  ({len(y_train)/n*100:.1f}%)")
        print(f"  VAL   : {len(y_val):5d}  ({len(y_val)/n*100:.1f}%)")
        print(f"  TEST  : {len(y_test):5d}  ({len(y_test)/n*100:.1f}%)")
        print("\n  Per-class breakdown:")
        print(f"  {'Class':12s}  {'Train':>6} {'Val':>6} {'Test':>6}")
        print("  " + "-" * 34)
        for cid, cname in ID_TO_CLASS.items():
            tr = int(np.sum(y_train == cid))
            va = int(np.sum(y_val   == cid))
            te = int(np.sum(y_test  == cid))
            print(f"  {cname:12s}  {tr:>6} {va:>6} {te:>6}")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        idx_train, idx_val, idx_test,
    )


if __name__ == "__main__":
    # Quick smoke-test
    X_path = _ROOT / "data" / "output" / "image_predictions" / "X.npy"
    y_path = _ROOT / "data" / "output" / "image_predictions" / "y.npy"
    X = np.load(str(X_path))
    y = np.load(str(y_path))
    make_splits(X, y)
