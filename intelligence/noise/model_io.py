from pathlib import Path
from typing import Any

import joblib
import numpy as np

from intelligence.noise.features import NOISE_FEATURE_NAMES
from intelligence.noise.trainer import NoiseTrainingResult


MODEL_FORMAT_VERSION = "2.0"


def save_noise_model(
    training_result: NoiseTrainingResult,
    model_path: str | Path,
) -> Path:
    """Save a trained noise-classification pipeline and its metadata."""

    if not isinstance(training_result, NoiseTrainingResult):
        raise TypeError(
            "training_result must be a NoiseTrainingResult."
        )

    path = Path(model_path)

    if path.suffix.lower() != ".joblib":
        raise ValueError("Model file must use the .joblib extension.")

    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": MODEL_FORMAT_VERSION,
        "model": training_result.model,
        "feature_names": list(training_result.feature_names),
        "training_accuracy": training_result.training_accuracy,
        "validation_accuracy": training_result.validation_accuracy,
    }

    joblib.dump(payload, path)

    return path


def load_noise_model(
    model_path: str | Path,
) -> NoiseTrainingResult:
    """Load and validate a previously saved noise model."""

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Model path is not a file: {path}"
        )

    if path.suffix.lower() != ".joblib":
        raise ValueError("Model file must use the .joblib extension.")

    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise ValueError(
            "Unable to load the model file."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Invalid model file format.")

    required_keys = {
        "format_version",
        "model",
        "feature_names",
        "training_accuracy",
        "validation_accuracy",
    }

    if not required_keys.issubset(payload):
        raise ValueError(
            "Model file is missing required metadata."
        )

    if payload["format_version"] != MODEL_FORMAT_VERSION:
        raise ValueError(
            "Unsupported model format version."
        )

    feature_names = payload["feature_names"]

    if not isinstance(feature_names, list):
        raise ValueError("Invalid feature-name metadata.")

    if len(feature_names) != len(NOISE_FEATURE_NAMES):
        raise ValueError(
            f"Saved model must contain exactly {len(NOISE_FEATURE_NAMES)} feature names."
        )

    if not np.all(
        np.isfinite(
            [
                payload["training_accuracy"],
                payload["validation_accuracy"],
            ]
        )
    ):
        raise ValueError(
            "Saved accuracy metadata must be finite."
        )

    return NoiseTrainingResult(
        model=payload["model"],
        training_accuracy=float(payload["training_accuracy"]),
        validation_accuracy=float(payload["validation_accuracy"]),
        feature_names=feature_names,
    )