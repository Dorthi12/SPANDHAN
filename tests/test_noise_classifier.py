import numpy as np
import pytest

from core.config import ML_CLASSES, ML_FEATURE_NAMES
from intelligence.noise.classifier import NoiseClassifier


class DummyProbabilityModel:
    classes_ = np.array(
        ["Clean", "Gaussian", "Impulse"]
    )

    def predict(self, X):
        return np.array(["Gaussian"])

    def predict_proba(self, X):
        return np.array([
            [0.10, 0.85, 0.05]
        ])


class DummyLowConfidenceModel:
    classes_ = np.array(
        ["Clean", "Gaussian", "Impulse"]
    )

    def predict(self, X):
        return np.array(["Gaussian"])

    def predict_proba(self, X):
        return np.array([
            [0.30, 0.40, 0.30]
        ])


class DummyNoProbabilityModel:
    def predict(self, X):
        return np.array(["Periodic"])


class DummyInvalidClassModel:
    def predict(self, X):
        return np.array(["SomethingElse"])


def make_features():
    return {
        name: float(index + 1)
        for index, name in enumerate(ML_FEATURE_NAMES)
    }


def test_classifier_accepts_feature_dictionary():
    classifier = NoiseClassifier(
        model=DummyProbabilityModel()
    )

    result = classifier.predict(
        make_features()
    )

    assert result["prediction"] == "Gaussian"
    assert result["raw_prediction"] == "Gaussian"
    assert np.isclose(result["confidence"], 0.85)
    assert result["confidence_status"] == "high"
    assert result["is_unknown"] is False


def test_low_confidence_returns_unknown():
    classifier = NoiseClassifier(
        model=DummyLowConfidenceModel()
    )

    result = classifier.predict(
        make_features()
    )

    assert result["prediction"] == "Unknown"
    assert result["raw_prediction"] == "Gaussian"
    assert np.isclose(result["confidence"], 0.40)
    assert result["confidence_status"] == "low"
    assert result["is_unknown"] is True


def test_classifier_accepts_array():
    classifier = NoiseClassifier(
        model=DummyProbabilityModel()
    )

    features = np.arange(
        len(ML_FEATURE_NAMES),
        dtype=float,
    )

    result = classifier.predict(features)

    assert result["prediction"] == "Gaussian"


def test_classifier_rejects_wrong_feature_count():
    classifier = NoiseClassifier(
        model=DummyProbabilityModel()
    )

    with pytest.raises(ValueError):
        classifier.predict(np.ones(10))


def test_classifier_rejects_missing_dictionary_feature():
    classifier = NoiseClassifier(
        model=DummyProbabilityModel()
    )

    features = make_features()
    del features["rms"]

    with pytest.raises(ValueError):
        classifier.predict(features)


def test_classifier_rejects_nonfinite_features():
    classifier = NoiseClassifier(
        model=DummyProbabilityModel()
    )

    features = make_features()
    features["rms"] = np.nan

    with pytest.raises(ValueError):
        classifier.predict(features)


def test_no_probability_model_does_not_fake_confidence():
    classifier = NoiseClassifier(
        model=DummyNoProbabilityModel()
    )

    result = classifier.predict(
        make_features()
    )

    assert result["prediction"] == "Periodic"
    assert result["raw_prediction"] == "Periodic"
    assert result["confidence"] is None
    assert result["confidence_status"] == "unavailable"
    assert result["is_unknown"] is False


def test_invalid_model_prediction_is_rejected():
    classifier = NoiseClassifier(
        model=DummyInvalidClassModel()
    )

    with pytest.raises(ValueError):
        classifier.predict(
            make_features()
        )


def test_classifier_requires_model():
    classifier = NoiseClassifier()

    with pytest.raises(RuntimeError):
        classifier.predict(
            make_features()
        )


def test_invalid_confidence_threshold():
    with pytest.raises(ValueError):
        NoiseClassifier(
            confidence_threshold=1.5
        )

    with pytest.raises(ValueError):
        NoiseClassifier(
            confidence_threshold=-0.1
        )