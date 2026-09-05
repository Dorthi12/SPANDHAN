from pathlib import Path

import numpy as np

from intelligence.noise.dataset import split_noise_dataset
from intelligence.noise.dataset_builder import build_noise_dataset
from intelligence.noise.evaluator import evaluate_svm
from intelligence.noise.model_io import load_noise_model, save_noise_model
from intelligence.noise.trainer import train_svm


def test_end_to_end_noise_training(tmp_path):
    dataset = build_noise_dataset(
        samples_per_class=20,
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

    evaluation = evaluate_svm(
        training_result,
        split,
    )

    model_path = (
        Path(tmp_path)
        / "models"
        / "noise_classifier.joblib"
    )

    save_noise_model(
        training_result,
        model_path,
    )

    loaded = load_noise_model(
        model_path,
    )

    predictions = loaded.model.predict(
        split.X_test
    )

    assert len(predictions) == len(split.y_test)

    assert 0.0 <= evaluation.accuracy <= 1.0
    assert 0.0 <= evaluation.f1_macro <= 1.0


def test_training_pipeline_is_reproducible():
    dataset_a = build_noise_dataset(
        samples_per_class=20,
        length=1000,
        sampling_rate=1000,
        seed=123,
    )

    dataset_b = build_noise_dataset(
        samples_per_class=20,
        length=1000,
        sampling_rate=1000,
        seed=123,
    )

    split_a = split_noise_dataset(
        dataset_a,
        random_state=123,
    )

    split_b = split_noise_dataset(
        dataset_b,
        random_state=123,
    )

    result_a = train_svm(
        split_a,
        random_state=123,
    )

    result_b = train_svm(
        split_b,
        random_state=123,
    )

    prediction_a = result_a.model.predict(
        split_a.X_test
    )

    prediction_b = result_b.model.predict(
        split_b.X_test
    )

    assert np.array_equal(
        prediction_a,
        prediction_b,
    )

    assert (
        result_a.training_accuracy
        == result_b.training_accuracy
    )

    assert (
        result_a.validation_accuracy
        == result_b.validation_accuracy
    )