from pathlib import Path

import numpy as np
import pytest

from intelligence.noise.dataset import split_noise_dataset
from intelligence.noise.dataset_builder import build_noise_dataset
from intelligence.noise.model_io import (
    MODEL_FORMAT_VERSION,
    load_noise_model,
    save_noise_model,
)
from intelligence.noise.trainer import NoiseTrainingResult, train_svm


@pytest.fixture
def trained_result():
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

    return train_svm(
        split,
        random_state=42,
    ), split


def test_save_model_creates_file(tmp_path, trained_result):
    result, _ = trained_result

    model_path = tmp_path / "noise_classifier.joblib"

    saved_path = save_noise_model(
        result,
        model_path,
    )

    assert saved_path == model_path
    assert model_path.exists()
    assert model_path.is_file()


def test_load_model_returns_training_result(tmp_path, trained_result):
    result, _ = trained_result

    model_path = tmp_path / "noise_classifier.joblib"

    save_noise_model(result, model_path)

    loaded = load_noise_model(model_path)

    assert isinstance(loaded, NoiseTrainingResult)
    assert loaded.feature_names == result.feature_names
    assert loaded.training_accuracy == result.training_accuracy
    assert loaded.validation_accuracy == result.validation_accuracy


def test_round_trip_predictions_are_identical(
    tmp_path,
    trained_result,
):
    result, split = trained_result

    model_path = tmp_path / "noise_classifier.joblib"

    save_noise_model(result, model_path)

    loaded = load_noise_model(model_path)

    original_predictions = result.model.predict(
        split.X_test
    )

    loaded_predictions = loaded.model.predict(
        split.X_test
    )

    assert np.array_equal(
        original_predictions,
        loaded_predictions,
    )


def test_round_trip_probabilities_are_identical(
    tmp_path,
    trained_result,
):
    result, split = trained_result

    model_path = tmp_path / "noise_classifier.joblib"

    save_noise_model(result, model_path)

    loaded = load_noise_model(model_path)

    original_probabilities = result.model.predict_proba(
        split.X_test
    )

    loaded_probabilities = loaded.model.predict_proba(
        split.X_test
    )

    assert np.allclose(
        original_probabilities,
        loaded_probabilities,
    )


def test_nested_directory_is_created(tmp_path, trained_result):
    result, _ = trained_result

    model_path = (
        tmp_path
        / "models"
        / "noise"
        / "classifier.joblib"
    )

    save_noise_model(result, model_path)

    assert model_path.exists()


def test_invalid_extension_is_rejected(tmp_path, trained_result):
    result, _ = trained_result

    with pytest.raises(ValueError):
        save_noise_model(
            result,
            tmp_path / "model.pkl",
        )


def test_missing_model_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_noise_model(
            tmp_path / "missing.joblib"
        )


def test_corrupted_model_is_rejected(tmp_path):
    model_path = tmp_path / "corrupted.joblib"
    model_path.write_bytes(b"not a valid joblib file")

    with pytest.raises(ValueError):
        load_noise_model(model_path)


def test_invalid_training_result_is_rejected(tmp_path):
    with pytest.raises(TypeError):
        save_noise_model(
            "not a training result",
            tmp_path / "model.joblib",
        )


def test_model_format_version_is_defined():
    assert isinstance(MODEL_FORMAT_VERSION, str)
    assert MODEL_FORMAT_VERSION == "1.0"