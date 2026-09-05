from pathlib import Path
from typing import Any

import joblib
import numpy as np

from core.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    ML_CLASSES,
    ML_FEATURE_NAMES,
)
from intelligence.noise.features import extract_noise_features
from intelligence.noise.model_io import load_noise_model


class NoiseClassifier:
    """Prediction interface for the trained noise-classification model."""

    def __init__(
        self,
        model: Any | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        self.model = model
        self.confidence_threshold = float(confidence_threshold)

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> "NoiseClassifier":
        """Load a persisted noise model."""

        training_result = load_noise_model(model_path)

        if list(training_result.feature_names) != list(ML_FEATURE_NAMES):
            raise ValueError(
                "Saved model feature names do not match the ML specification."
            )

        return cls(
            model=training_result.model,
            confidence_threshold=confidence_threshold,
        )

    def _validate_features(
        self,
        features: dict[str, float] | np.ndarray,
    ) -> np.ndarray:
        """Validate and normalize the 14-feature input."""

        if isinstance(features, dict):
            if set(features.keys()) != set(ML_FEATURE_NAMES):
                raise ValueError(
                    "Feature dictionary must contain exactly the required "
                    "14 feature names."
                )

            vector = np.asarray(
                [features[name] for name in ML_FEATURE_NAMES],
                dtype=np.float64,
            )

        else:
            vector = np.asarray(
                features,
                dtype=np.float64,
            )

            if vector.ndim == 2 and vector.shape[0] == 1:
                vector = vector[0]

            if vector.ndim != 1:
                raise ValueError(
                    "Feature vector must be one-dimensional."
                )

            if len(vector) != len(ML_FEATURE_NAMES):
                raise ValueError(
                    "Feature vector must contain exactly 14 values."
                )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "Feature vector must contain only finite values."
            )

        return vector

    def _validate_prediction(
        self,
        prediction: str,
    ) -> None:
        if prediction not in ML_CLASSES:
            raise ValueError(
                f"Model returned unsupported class: {prediction}"
            )

    def _get_confidence(
        self,
        feature_vector: np.ndarray,
    ) -> tuple[str, float, np.ndarray, list[str]]:
        """Return predicted class, confidence, probabilities and classes."""

        if self.model is None:
            raise RuntimeError(
                "No model is loaded."
            )

        if not hasattr(self.model, "predict"):
            raise TypeError(
                "Loaded model does not support prediction."
            )

        if not hasattr(self.model, "predict_proba"):
            raise TypeError(
                "Loaded model does not provide calibrated probabilities."
            )

        X = feature_vector.reshape(1, -1)

        prediction = str(self.model.predict(X)[0])

        probabilities = np.asarray(
            self.model.predict_proba(X)[0],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(probabilities)):
            raise ValueError(
                "Model returned non-finite probabilities."
            )

        if not np.all(probabilities >= 0.0):
            raise ValueError(
                "Model returned negative probabilities."
            )

        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError(
                "Model probabilities must sum to 1."
            )

        classifier = self.model.named_steps.get("classifier")

        if classifier is None or not hasattr(classifier, "classes_"):
            raise TypeError(
                "Loaded model does not expose class names."
            )

        classes = [
            str(value)
            for value in classifier.classes_
        ]

        if len(classes) != len(probabilities):
            raise ValueError(
                "Probability output and class list have different lengths."
            )

        self._validate_prediction(prediction)

        predicted_index = int(
            np.argmax(probabilities)
        )

        # Ensure predict() agrees with the highest probability.
        probability_prediction = classes[predicted_index]

        if probability_prediction != prediction:
            raise ValueError(
                "Model prediction disagrees with its probability output."
            )

        confidence = float(
            probabilities[predicted_index]
        )

        return (
            prediction,
            confidence,
            probabilities,
            classes,
        )

    def predict(
        self,
        features: dict[str, float] | np.ndarray,
    ) -> dict[str, Any]:
        """Predict the noise class from an extracted feature vector."""

        feature_vector = self._validate_features(features)

        prediction, confidence, probabilities, classes = (
            self._get_confidence(feature_vector)
        )

        is_unknown = confidence < self.confidence_threshold

        status = (
            "Unknown"
            if is_unknown
            else prediction
        )

        return {
            "prediction": status,
            "raw_prediction": prediction,
            "confidence": confidence,
            "threshold": self.confidence_threshold,
            "status": status,
            "is_unknown": is_unknown,
            "probabilities": probabilities,
            "classes": classes,
            "feature_names": list(ML_FEATURE_NAMES),
            "feature_vector": feature_vector,
        }

    def predict_signal(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        snr_db: float | None = None,
    ) -> dict[str, Any]:
        """Extract the 14 required features and classify a signal."""

        signal = np.asarray(
            signal,
            dtype=np.float64,
        )

        if signal.ndim != 1:
            raise ValueError(
                "signal must be one-dimensional."
            )

        if signal.size == 0:
            raise ValueError(
                "signal cannot be empty."
            )

        if not np.all(np.isfinite(signal)):
            raise ValueError(
                "signal must contain only finite values."
            )

        if not np.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError(
                "sampling_rate must be finite and greater than zero."
            )

        features = extract_noise_features(
            signal,
            sampling_rate,
            snr_db=snr_db,
        )

        result = self.predict(features)

        result["sampling_rate"] = float(sampling_rate)
        result["num_samples"] = int(signal.size)

        return result