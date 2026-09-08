"""
intelligence/unified/model_io.py
===================================
Save and load the domain-routed noise classifier model bundle.

Two separate sklearn Pipelines are stored:
  • audio_model  — trained on audio-only features (37-dim)
  • image_model  — trained on image-only features (30-dim)

Inference is routed to the correct model via the domain tag.

Model file : models/unified_noise_classifier.joblib
Metadata   : models/unified_noise_classifier_meta.json
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
class DomainRoutedBundle:
    """
    Two domain-specific sklearn Pipelines + metadata.

    audio_model is trained ONLY on audio features (indices 0..36).
    image_model is trained ONLY on image features (indices 0..29).
    """

    # Domain-specific models
    audio_model: Any                      # sklearn Pipeline (audio)
    image_model: Any                      # sklearn Pipeline (image)

    # Audio sub-model metadata
    audio_class_names: list[str]
    audio_train_accuracy: float
    audio_val_accuracy: float
    audio_n_train: int
    audio_n_val: int
    audio_classification_report: str = ""
    audio_confusion_matrix: Optional[np.ndarray] = None
    audio_feature_importances: Optional[np.ndarray] = None

    # Image sub-model metadata
    image_class_names: list[str] = field(default_factory=list)
    image_train_accuracy: float = 0.0
    image_val_accuracy: float = 0.0
    image_n_train: int = 0
    image_n_val: int = 0
    image_classification_report: str = ""
    image_confusion_matrix: Optional[np.ndarray] = None
    image_feature_importances: Optional[np.ndarray] = None

    # Combined metadata
    training_metadata: dict = field(default_factory=dict)

    @property
    def val_accuracy(self) -> float:
        """Combined validation accuracy (weighted by sample count)."""
        n = self.audio_n_val + self.image_n_val
        if n == 0:
            return 0.0
        return (
            self.audio_val_accuracy * self.audio_n_val
            + self.image_val_accuracy * self.image_n_val
        ) / n


# Backwards-compat alias
UnifiedModelBundle = DomainRoutedBundle


def save_model(
    bundle: DomainRoutedBundle,
    model_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> None:
    """
    Persist both sub-models + metadata to disk.
    """
    mp = Path(model_path or DEFAULT_MODEL_PATH)
    mm = Path(meta_path or DEFAULT_META_PATH)
    mp.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, mp)

    def _arr(a):
        return a.tolist() if a is not None else None

    meta = {
        "audio_class_names":             bundle.audio_class_names,
        "audio_train_accuracy":          float(bundle.audio_train_accuracy),
        "audio_val_accuracy":            float(bundle.audio_val_accuracy),
        "audio_n_train":                 bundle.audio_n_train,
        "audio_n_val":                   bundle.audio_n_val,
        "audio_classification_report":   bundle.audio_classification_report,
        "audio_confusion_matrix":        _arr(bundle.audio_confusion_matrix),
        "audio_feature_importances":     _arr(bundle.audio_feature_importances),
        "image_class_names":             bundle.image_class_names,
        "image_train_accuracy":          float(bundle.image_train_accuracy),
        "image_val_accuracy":            float(bundle.image_val_accuracy),
        "image_n_train":                 bundle.image_n_train,
        "image_n_val":                   bundle.image_n_val,
        "image_classification_report":   bundle.image_classification_report,
        "image_confusion_matrix":        _arr(bundle.image_confusion_matrix),
        "image_feature_importances":     _arr(bundle.image_feature_importances),
        "training_metadata":             bundle.training_metadata,
    }
    with open(mm, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_model(
    model_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> DomainRoutedBundle:
    """
    Load the domain-routed bundle from disk.

    Raises FileNotFoundError if model has not been trained yet.
    """
    mp = Path(model_path or DEFAULT_MODEL_PATH)

    if not mp.exists():
        raise FileNotFoundError(
            f"Model not found: {mp}\n"
            "Run  python scripts/train_unified_model.py  first."
        )

    bundle = joblib.load(mp)

    # Handle legacy UnifiedModelBundle (single-model) for backward compat
    if not isinstance(bundle, DomainRoutedBundle):
        raise RuntimeError(
            "Stale model file detected. Please retrain:\n"
            "  python scripts/train_unified_model.py"
        )

    return bundle


def model_exists(model_path: Optional[Path] = None) -> bool:
    """Return True if a trained model file is present on disk."""
    return Path(model_path or DEFAULT_MODEL_PATH).exists()
