import numpy as np
import pytest

from intelligence.noise.classifier import NoiseClassifier
from intelligence.noise.dataset import split_noise_dataset
from intelligence.noise.dataset_builder import build_noise_dataset
from intelligence.noise.model_io import save_noise_model
from intelligence.noise.trainer import train_svm


@pytest.fixture
def trained_system():
    dataset = build_noise_dataset(
        samples_per_class=10,
        length=1000,
        sampling_rate=1000,
        seed=42,
    )

    split = split_noise_dataset(
        dataset,
        random_state=42,
    )

    training_result = train_svm(
        split,
        random_state=42,
    )

    return training_result, split


@pytest.fixture
def classifier(trained_system):
    training_result, _ = trained_system

    return NoiseClassifier(
        model=training_result.model,
        confidence_threshold=0.60,
    )


def test_classifier_requires_model():
    classifier = NoiseClassifier()

    with pytest.raises(RuntimeError):
        classifier.predict(np.zeros(14))


def test_classifier_accepts_feature_array(
    classifier,
):
    features = np.zeros(14)

    result = classifier.predict(features)

    assert isinstance(result, dict)
    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result


def test_classifier_accepts_feature_dictionary(
    classifier,
):
    features = {
        name: 0.0
        for name in classifier.predict(
            np.zeros(14)
        )["feature_names"]
    }

    result = classifier.predict(features)

    assert isinstance(result["prediction"], str)


def test_probability_output_is_valid(
    classifier,
):
    result = classifier.predict(
        np.zeros(14)
    )

    probabilities = result["probabilities"]

    assert len(probabilities) == 6
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.isclose(probabilities.sum(), 1.0)


def test_confidence_matches_highest_probability(
    classifier,
):
    result = classifier.predict(
        np.zeros(14)
    )

    assert np.isclose(
        result["confidence"],
        np.max(result["probabilities"]),
    )


def test_unknown_threshold_is_applied(
    classifier,
):
    result = classifier.predict(
        np.zeros(14)
    )

    if result["confidence"] < 0.60:
        assert result["prediction"] == "Unknown"
        assert result["is_unknown"] is True
        assert result["status"] == "Unknown"
    else:
        assert result["prediction"] == result["raw_prediction"]
        assert result["is_unknown"] is False


def test_feature_order_is_fixed(
    classifier,
):
    result = classifier.predict(
        np.zeros(14)
    )

    expected = [
        "rms",
        "variance",
        "std",
        "kurtosis",
        "skewness",
        "crest_factor",
        "zero_crossing_rate",
        "spectral_centroid",
        "spectral_flatness",
        "spectral_entropy",
        "spectral_rolloff",
        "mains_band_energy",
        "high_band_energy",
        "snr_db",
    ]

    assert result["feature_names"] == expected


def test_invalid_feature_length_is_rejected(
    classifier,
):
    with pytest.raises(ValueError):
        classifier.predict(np.zeros(13))

    with pytest.raises(ValueError):
        classifier.predict(np.zeros(15))


def test_non_finite_features_are_rejected(
    classifier,
):
    features = np.zeros(14)
    features[0] = np.nan

    with pytest.raises(ValueError):
        classifier.predict(features)


def test_predict_signal_extracts_features(
    classifier,
):
    sampling_rate = 1000.0

    t = np.arange(2000) / sampling_rate

    signal = (
        np.sin(2 * np.pi * 50 * t)
        + 0.5 * np.sin(2 * np.pi * 120 * t)
    )

    result = classifier.predict_signal(
        signal,
        sampling_rate,
        snr_db=20.0,
    )

    assert result["num_samples"] == len(signal)
    assert result["sampling_rate"] == sampling_rate
    assert len(result["feature_vector"]) == 14
    assert len(result["feature_names"]) == 14


def test_predict_signal_rejects_invalid_input(
    classifier,
):
    with pytest.raises(ValueError):
        classifier.predict_signal(
            np.zeros((2, 100)),
            1000,
        )

    with pytest.raises(ValueError):
        classifier.predict_signal(
            np.zeros(100),
            0,
        )