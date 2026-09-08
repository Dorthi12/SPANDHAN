"""
intelligence/unified/model_io.py
===================================
Save and load the unified noise classifier model + metadata.

Model file: models/unified_noise_classifier.joblib
Metadata  : models/unified_noise_classifier_meta.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

DEFAULT_MODEL_PATH = Path("models/unified_noise_classifier.joblib")
DEFAULT_META_PATH  = Path("models/unified_noise_classifier_meta.json")


@dataclass
class UnifiedModelBundle:
    """Everything needed to run inference with the trained model."""

    model: Any                        # sklearn Pipeline or estimator
    feature_names: list[str]
    class_names: list[str]
    train_accuracy: float
    val_accuracy: float
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: str = ""
    feature_importances: Optional[np.ndarray] = None
    n_train: int = 0
    n_val: int = 0
    training_metadata: dict = field(default_factory=dict)


def save_model(
    bundle: UnifiedModelBundle,
    model_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> None:
    """
    Persist model + metadata to disk.

    Parameters
    ----------
    bundle     : trained UnifiedModelBundle
    model_path : .joblib file (default: models/unified_noise_classifier.joblib)
    meta_path  : .json file  (default: models/unified_noise_classifier_meta.json)
    """
    mp = Path(model_path or DEFAULT_MODEL_PATH)
    mm = Path(meta_path or DEFAULT_META_PATH)
    mp.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle.model, mp)

    meta = {
        "feature_names": bundle.feature_names,
        "class_names": bundle.class_names,
        "train_accuracy": float(bundle.train_accuracy),
        "val_accuracy": float(bundle.val_accuracy),
        "n_train": bundle.n_train,
        "n_val": bundle.n_val,
        "classification_report": bundle.classification_report,
        "training_metadata": bundle.training_metadata,
    }
    if bundle.confusion_matrix is not None:
        meta["confusion_matrix"] = bundle.confusion_matrix.tolist()
    if bundle.feature_importances is not None:
        meta["feature_importances"] = bundle.feature_importances.tolist()

    with open(mm, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_model(
    model_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> UnifiedModelBundle:
    """
    Load model + metadata from disk.

    Raises FileNotFoundError if model has not been trained yet.
    """
    mp = Path(model_path or DEFAULT_MODEL_PATH)
    mm = Path(meta_path or DEFAULT_META_PATH)

    if not mp.exists():
        raise FileNotFoundError(
            f"Model not found: {mp}\n"
            "Run  python scripts/train_unified_model.py  first."
        )

    model = joblib.load(mp)
    meta: dict = {}
    if mm.exists():
        with open(mm, "r", encoding="utf-8") as f:
            meta = json.load(f)

    cm = meta.get("confusion_matrix")
    fi = meta.get("feature_importances")

    return UnifiedModelBundle(
        model=model,
        feature_names=meta.get("feature_names", []),
        class_names=meta.get("class_names", []),
        train_accuracy=float(meta.get("train_accuracy", 0.0)),
        val_accuracy=float(meta.get("val_accuracy", 0.0)),
        confusion_matrix=np.array(cm) if cm is not None else None,
        classification_report=meta.get("classification_report", ""),
        feature_importances=np.array(fi) if fi is not None else None,
        n_train=int(meta.get("n_train", 0)),
        n_val=int(meta.get("n_val", 0)),
        training_metadata=meta.get("training_metadata", {}),
    )


def model_exists(model_path: Optional[Path] = None) -> bool:
    """Return True if a trained model file is present on disk."""
    return Path(model_path or DEFAULT_MODEL_PATH).exists()
