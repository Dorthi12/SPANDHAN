"""
Classifier module.
"""
from pathlib import Path

import numpy as np
import joblib

from core.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    ML_CLASSES,
    ML_FEATURE_NAMES,
)


class NoiseClassifier:
    """
    Inference interface for the trained Spandhan noise classifier.

    The classifier expects the project's fixed 14-feature vector and
    a persisted scikit-learn pipeline/model.
    """

    def __init__(
        self,
        model=None,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError(
                "Confidence threshold must be between 0 and 1."
            )

        self.model = model
        self.confidence_threshold = float(confidence_threshold)

    @property
    def is_loaded(self):
        return self.model is not None

    def load(self, model_path):
        """
        Load a persisted classifier from disk.
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        self.model = joblib.load(model_path)

        return self

    def _validate_features(self, features):
        """
        Convert and validate the fixed 14-feature vector.

        A dictionary must contain exactly the project's feature names.
        An array-like input must contain exactly 14 values.
        """

        if isinstance(features, dict):
            missing = [
                name
                for name in ML_FEATURE_NAMES
                if name not in features
            ]

            if missing:
                raise ValueError(
                    f"Missing features: {missing}"
                )

            extra = [
                name
                for name in features
                if name not in ML_FEATURE_NAMES
            ]

            if extra:
                raise ValueError(
                    f"Unexpected features: {extra}"
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

            if vector.ndim == 2:
                if vector.shape[0] != 1:
                    raise ValueError(
                        "Feature array must contain one sample."
                    )

                vector = vector[0]

            if vector.ndim != 1:
                raise ValueError(
                    "Features must be one-dimensional."
                )

            if len(vector) != len(ML_FEATURE_NAMES):
                raise ValueError(
                    f"Expected {len(ML_FEATURE_NAMES)} features, "
                    f"got {len(vector)}."
                )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                "Features must contain only finite values."
            )

        return vector

    def _validate_prediction(self, prediction):
        prediction = str(prediction)

        if prediction not in ML_CLASSES:
            raise ValueError(
                f"Model returned unsupported class: {prediction}"
            )

        return prediction

    def _get_confidence(self, feature_vector, prediction):
        """
        Obtain a model confidence when supported.

        Preference:
        1. predict_proba()
        2. calibrated/explicit confidence attribute
        3. None

        We deliberately do not convert SVM decision scores into
        probabilities using softmax.
        """

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(
                feature_vector.reshape(1, -1)
            )

            probabilities = np.asarray(
                probabilities,
                dtype=np.float64,
            )

            if probabilities.ndim != 2 or probabilities.shape[0] != 1:
                raise ValueError(
                    "Model returned invalid probability shape."
                )

            classes = getattr(
                self.model,
                "classes_",
                None,
            )

            if classes is None:
                raise ValueError(
                    "Model probabilities require classes_."
                )

            classes = np.asarray(classes)

            matching = np.where(
                classes.astype(str) == prediction
            )[0]

            if len(matching) == 0:
                raise ValueError(
                    f"Predicted class '{prediction}' "
                    "not found in model classes."
                )

            return float(probabilities[0, matching[0]])

        return None

    def predict(self, features):
        """
        Predict the noise class.

        Returns
        -------
        dict
            Prediction, confidence, known/unknown status.
        """

        if not self.is_loaded:
            raise RuntimeError(
                "No classifier model is loaded."
            )

        feature_vector = self._validate_features(features)

        raw_prediction = self.model.predict(
            feature_vector.reshape(1, -1)
        )[0]

        prediction = self._validate_prediction(
            raw_prediction
        )

        confidence = self._get_confidence(
            feature_vector,
            prediction,
        )

        if confidence is None:
            confidence_status = "unavailable"
            final_prediction = prediction
            is_unknown = False

        elif confidence < self.confidence_threshold:
            confidence_status = "low"
            final_prediction = "Unknown"
            is_unknown = True

        else:
            confidence_status = "high"
            final_prediction = prediction
            is_unknown = False

        return {
            "prediction": final_prediction,
            "raw_prediction": prediction,
            "confidence": confidence,
            "confidence_threshold": self.confidence_threshold,
            "confidence_status": confidence_status,
            "is_unknown": is_unknown,
            "feature_names": list(ML_FEATURE_NAMES),
            "feature_vector": feature_vector,
        }