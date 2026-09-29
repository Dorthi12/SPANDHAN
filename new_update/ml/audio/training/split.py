"""
SPANDHAN — Training: Train / Val / Test Split
===============================================
Stratified split so every class is proportionally represented in each fold.

Usage (internal):
    from ml.audio.training.split import make_splits
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.common.config import TRAIN_RATIO, VAL_RATIO, RANDOM_STATE


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train / val / test split.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First cut: (train+val) vs test
    test_size = 1.0 - TRAIN_RATIO - VAL_RATIO
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Second cut: train vs val (relative size within tv)
    val_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_relative,
        stratify=y_tv,
        random_state=RANDOM_STATE,
    )

    print(f"Split sizes  —  train: {len(y_train)}  "
          f"val: {len(y_val)}  test: {len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test
