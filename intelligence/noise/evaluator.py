from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from intelligence.noise.dataset import NoiseDatasetSplit
from intelligence.noise.trainer import NoiseTrainingResult


@dataclass
class NoiseEvaluationResult:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float

    precision_weighted: float
    recall_weighted: float
    f1_weighted: float

    confusion_matrix: np.ndarray
    class_names: list[str]

    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray

    validation_accuracy: float
    training_accuracy: float


def _validate_evaluation_inputs(
    training_result: NoiseTrainingResult,
    dataset: NoiseDatasetSplit,
) -> None:
    if not isinstance(training_result, NoiseTrainingResult):
        raise TypeError(
            "training_result must be a NoiseTrainingResult."
        )

    if not isinstance(dataset, NoiseDatasetSplit):
        raise TypeError(
            "dataset must be a NoiseDatasetSplit."
        )

    if len(dataset.X_test) == 0:
        raise ValueError("Test dataset cannot be empty.")

    if len(dataset.X_test) != len(dataset.y_test):
        raise ValueError(
            "Test features and labels must have equal lengths."
        )

    X_test = np.asarray(dataset.X_test, dtype=np.float64)

    if X_test.ndim != 2:
        raise ValueError("X_test must be two-dimensional.")

    if not np.all(np.isfinite(X_test)):
        raise ValueError(
            "X_test must contain only finite values."
        )

    if len(dataset.feature_names) != len(training_result.feature_names):
        raise ValueError(
            "Dataset and model feature counts do not match."
        )

    if list(dataset.feature_names) != list(training_result.feature_names):
        raise ValueError(
            "Dataset and model feature ordering does not match."
        )


def evaluate_svm(
    training_result: NoiseTrainingResult,
    dataset: NoiseDatasetSplit,
) -> NoiseEvaluationResult:
    """
    Evaluate a trained noise-classification model on the untouched test set.
    """

    _validate_evaluation_inputs(training_result, dataset)

    model = training_result.model

    y_true = np.asarray(dataset.y_test)
    y_pred = np.asarray(model.predict(dataset.X_test))
    probabilities = np.asarray(
        model.predict_proba(dataset.X_test),
        dtype=np.float64,
    )

    # Use the model's learned class ordering so the confusion matrix
    # and probability columns have a consistent interpretation.
    classifier = model.named_steps["classifier"]
    class_names = list(classifier.classes_)

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(
        y_true,
        y_pred,
        labels=class_names,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        labels=class_names,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        labels=class_names,
        average="macro",
        zero_division=0,
    )

    precision_weighted = precision_score(
        y_true,
        y_pred,
        labels=class_names,
        average="weighted",
        zero_division=0,
    )

    recall_weighted = recall_score(
        y_true,
        y_pred,
        labels=class_names,
        average="weighted",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        labels=class_names,
        average="weighted",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=class_names,
    )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Model produced non-finite probability values."
        )

    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError(
            "Model probability rows must sum to 1."
        )

    return NoiseEvaluationResult(
        accuracy=float(accuracy),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        precision_weighted=float(precision_weighted),
        recall_weighted=float(recall_weighted),
        f1_weighted=float(f1_weighted),
        confusion_matrix=matrix,
        class_names=class_names,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        validation_accuracy=training_result.validation_accuracy,
        training_accuracy=training_result.training_accuracy,
    )