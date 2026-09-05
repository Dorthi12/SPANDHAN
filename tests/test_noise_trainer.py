import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from intelligence.noise.dataset import split_noise_dataset
from intelligence.noise.dataset_builder import build_noise_dataset
from intelligence.noise.trainer import (
    NoiseTrainingResult,
    create_svm_pipeline,
    train_svm,
)


@pytest.fixture
def dataset_split():
    dataset = build_noise_dataset(
        samples_per_class=10,
        length=1000,
        sampling_rate=1000,
        seed=42,
    )

    return split_noise_dataset(dataset, random_state=42)


def test_create_svm_pipeline():
    pipeline = create_svm_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["classifier"], CalibratedClassifierCV)

    classifier = pipeline.named_steps["classifier"]

    assert classifier.estimator.kernel == "rbf"
    # When SVC is wrapped in CalibratedClassifierCV, sklearn marks
    # SVC.probability as deprecated (returns 'deprecated' sentinel, not False).
    # The important invariant is that probability=True was NOT explicitly set
    # on the raw SVC; CalibratedClassifierCV owns the probability output.
    assert classifier.estimator.probability != True  # noqa: E712
    assert classifier.ensemble is False
    assert classifier.method == "sigmoid"


def test_scaler_is_inside_training_pipeline(dataset_split):
    result = train_svm(dataset_split)

    pipeline = result.model

    assert "scaler" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    scaler = pipeline.named_steps["scaler"]

    # StandardScaler must have been fitted by the training data.
    assert hasattr(scaler, "mean_")
    assert hasattr(scaler, "scale_")


def test_training_produces_valid_result(dataset_split):
    result = train_svm(dataset_split)

    assert isinstance(result, NoiseTrainingResult)
    assert isinstance(result.model, Pipeline)

    assert 0.0 <= result.training_accuracy <= 1.0
    assert 0.0 <= result.validation_accuracy <= 1.0

    assert len(result.feature_names) == 14


def test_model_can_predict(dataset_split):
    result = train_svm(dataset_split)

    predictions = result.model.predict(dataset_split.X_test)

    assert len(predictions) == len(dataset_split.y_test)
    assert set(predictions).issubset(set(dataset_split.y_train))


def test_model_provides_probabilities(dataset_split):
    result = train_svm(dataset_split)

    probabilities = result.model.predict_proba(dataset_split.X_test)

    assert probabilities.shape == (
        len(dataset_split.X_test),
        len(np.unique(dataset_split.y_train)),
    )

    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)

    row_sums = probabilities.sum(axis=1)

    assert np.allclose(row_sums, 1.0)


def test_training_is_reproducible(dataset_split):
    result_a = train_svm(dataset_split, random_state=123)
    result_b = train_svm(dataset_split, random_state=123)

    predictions_a = result_a.model.predict(dataset_split.X_test)
    predictions_b = result_b.model.predict(dataset_split.X_test)

    assert np.array_equal(predictions_a, predictions_b)
    assert result_a.training_accuracy == result_b.training_accuracy
    assert result_a.validation_accuracy == result_b.validation_accuracy


def test_invalid_C_is_rejected():
    with pytest.raises(ValueError):
        create_svm_pipeline(C=0)

    with pytest.raises(ValueError):
        create_svm_pipeline(C=-1)


def test_invalid_gamma_is_rejected():
    with pytest.raises(ValueError):
        create_svm_pipeline(gamma=0)

    with pytest.raises(ValueError):
        create_svm_pipeline(gamma=-1)

    with pytest.raises(ValueError):
        create_svm_pipeline(gamma="invalid")


def test_invalid_random_state_is_rejected():
    with pytest.raises(ValueError):
        create_svm_pipeline(random_state=1.5)


def test_invalid_training_split_is_rejected(dataset_split):
    broken = type(dataset_split)(
        X_train=np.empty((0, 14)),
        X_val=dataset_split.X_val,
        X_test=dataset_split.X_test,
        y_train=np.empty((0,), dtype=object),
        y_val=dataset_split.y_val,
        y_test=dataset_split.y_test,
        metadata_train=[],
        metadata_val=dataset_split.metadata_val,
        metadata_test=dataset_split.metadata_test,
        feature_names=dataset_split.feature_names,
    )

    with pytest.raises(ValueError):
        train_svm(broken)