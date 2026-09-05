import numpy as np
import pytest

from intelligence.noise.dataset import split_noise_dataset
from intelligence.noise.dataset_builder import build_noise_dataset
from intelligence.noise.evaluator import (
    NoiseEvaluationResult,
    evaluate_svm,
)
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


def test_evaluation_returns_result(trained_system):
    training_result, split = trained_system

    result = evaluate_svm(
        training_result,
        split,
    )

    assert isinstance(result, NoiseEvaluationResult)


def test_metrics_are_valid(trained_system):
    training_result, split = trained_system

    result = evaluate_svm(
        training_result,
        split,
    )

    metrics = [
        result.accuracy,
        result.precision_macro,
        result.recall_macro,
        result.f1_macro,
        result.precision_weighted,
        result.recall_weighted,
        result.f1_weighted,
    ]

    assert all(0.0 <= value <= 1.0 for value in metrics)


def test_predictions_match_test_set(trained_system):
    training_result, split = trained_system

    result = evaluate_svm(
        training_result,
        split,
    )

    assert len(result.y_true) == len(split.y_test)
    assert len(result.y_pred) == len(split.y_test)


def test_confusion_matrix_shape(trained_system):
    training_result, split = trained_system

    result = evaluate_svm(
        training_result,
        split,
    )

    num_classes = len(result.class_names)

    assert result.confusion_matrix.shape == (
        num_classes,
        num_classes,
    )

    assert np.all(result.confusion_matrix >= 0)


def test_probability_output_is_valid(trained_system):
    training_result, split = trained_system

    result = evaluate_svm(
        training_result,
        split,
    )

    assert result.probabilities.shape == (
        len(split.X_test),
        len(result.class_names),
    )

    assert np.all(np.isfinite(result.probabilities))
    assert np.all(result.probabilities >= 0.0)
    assert np.all(result.probabilities <= 1.0)

    assert np.allclose(
        result.probabilities.sum(axis=1),
        1.0,
    )


def test_all_expected_classes_are_present(trained_system):
    training_result, split = trained_system

    result = evaluate_svm(
        training_result,
        split,
    )

    expected_classes = {
        "Clean",
        "Gaussian",
        "Impulse",
        "Periodic",
        "Colored",
        "Mixed",
    }

    assert set(result.class_names) == expected_classes


def test_evaluation_is_reproducible(trained_system):
    training_result, split = trained_system

    result_a = evaluate_svm(
        training_result,
        split,
    )

    result_b = evaluate_svm(
        training_result,
        split,
    )

    assert result_a.accuracy == result_b.accuracy
    assert np.array_equal(
        result_a.y_pred,
        result_b.y_pred,
    )

    assert np.allclose(
        result_a.probabilities,
        result_b.probabilities,
    )


def test_training_and_validation_metrics_are_preserved(trained_system):
    training_result, split = trained_system

    result = evaluate_svm(
        training_result,
        split,
    )

    assert (
        result.training_accuracy
        == training_result.training_accuracy
    )

    assert (
        result.validation_accuracy
        == training_result.validation_accuracy
    )


def test_feature_mismatch_is_rejected(trained_system):
    training_result, split = trained_system

    broken = type(split)(
        X_train=split.X_train,
        X_val=split.X_val,
        X_test=split.X_test,
        y_train=split.y_train,
        y_val=split.y_val,
        y_test=split.y_test,
        metadata_train=split.metadata_train,
        metadata_val=split.metadata_val,
        metadata_test=split.metadata_test,
        feature_names=list(reversed(split.feature_names)),
    )

    with pytest.raises(ValueError):
        evaluate_svm(training_result, broken)


def test_non_finite_test_data_is_rejected(trained_system):
    training_result, split = trained_system

    broken_X_test = split.X_test.copy()
    broken_X_test[0, 0] = np.nan

    broken = type(split)(
        X_train=split.X_train,
        X_val=split.X_val,
        X_test=broken_X_test,
        y_train=split.y_train,
        y_val=split.y_val,
        y_test=split.y_test,
        metadata_train=split.metadata_train,
        metadata_val=split.metadata_val,
        metadata_test=split.metadata_test,
        feature_names=split.feature_names,
    )

    with pytest.raises(ValueError):
        evaluate_svm(training_result, broken)