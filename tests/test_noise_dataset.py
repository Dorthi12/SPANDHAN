import numpy as np
import pytest

from intelligence.noise.dataset import (
    prepare_noise_dataset,
    split_noise_dataset,
    validate_noise_dataset,
)
from intelligence.noise.dataset_builder import build_noise_dataset
from intelligence.noise.features import NOISE_FEATURE_NAMES


@pytest.fixture
def dataset():
    return build_noise_dataset(
        samples_per_class=10,
        length=1000,
        sampling_rate=1000,
        seed=42,
    )


def test_validate_dataset(dataset):
    prepared = prepare_noise_dataset(dataset)

    validate_noise_dataset(prepared)

    assert prepared.X.shape[1] == 20
    assert np.all(np.isfinite(prepared.X))


def test_clean_infinity_is_resolved(dataset):
    snr_index = dataset.feature_names.index("snr_db")

    # Identify which rows had +inf SNR before preparation (Clean samples).
    inf_mask = np.isposinf(dataset.X[:, snr_index])
    assert np.any(inf_mask), "Expected at least one Clean (+inf SNR) sample."

    prepared = prepare_noise_dataset(dataset)

    # All values must be finite after preparation.
    assert np.all(np.isfinite(prepared.X))

    # Only the formerly-infinite (Clean) rows should be clamped to 100.0.
    assert np.all(prepared.X[inf_mask, snr_index] == 100.0)


def test_feature_order_is_preserved(dataset):
    prepared = prepare_noise_dataset(dataset)

    assert prepared.feature_names == NOISE_FEATURE_NAMES


def test_stratified_split_preserves_classes(dataset):
    prepared = prepare_noise_dataset(dataset)

    split = split_noise_dataset(prepared)

    assert set(split.y_train) == set(prepared.y)
    assert set(split.y_val) == set(prepared.y)
    assert set(split.y_test) == set(prepared.y)


def test_default_split_is_70_15_15(dataset):
    prepared = prepare_noise_dataset(dataset)

    split = split_noise_dataset(prepared)

    total = len(prepared.X)

    assert len(split.X_train) == int(0.70 * total)
    assert len(split.X_val) == int(0.15 * total)
    assert len(split.X_test) == total - len(split.X_train) - len(split.X_val)


def test_split_is_reproducible(dataset):
    prepared = prepare_noise_dataset(dataset)

    split_a = split_noise_dataset(prepared, random_state=123)
    split_b = split_noise_dataset(prepared, random_state=123)

    assert np.array_equal(split_a.X_train, split_b.X_train)
    assert np.array_equal(split_a.y_train, split_b.y_train)
    assert np.array_equal(split_a.X_val, split_b.X_val)
    assert np.array_equal(split_a.X_test, split_b.X_test)


def test_different_seed_changes_split(dataset):
    prepared = prepare_noise_dataset(dataset)

    split_a = split_noise_dataset(prepared, random_state=1)
    split_b = split_noise_dataset(prepared, random_state=2)

    assert not np.array_equal(split_a.X_train, split_b.X_train)


def test_invalid_dataset_is_rejected(dataset):
    broken = type(dataset)(
        X=dataset.X[:, :-1],
        y=dataset.y,
        feature_names=dataset.feature_names[:-1],
        metadata=dataset.metadata,
    )

    with pytest.raises(ValueError):
        prepare_noise_dataset(broken)


def test_metadata_remains_separate(dataset):
    prepared = prepare_noise_dataset(dataset)

    split = split_noise_dataset(prepared)

    assert all(isinstance(item, dict) for item in split.metadata_train)
    assert all(isinstance(item, dict) for item in split.metadata_val)
    assert all(isinstance(item, dict) for item in split.metadata_test)


def test_invalid_split_parameters_are_rejected(dataset):
    prepared = prepare_noise_dataset(dataset)

    with pytest.raises(ValueError):
        split_noise_dataset(prepared, test_size=0.0)

    with pytest.raises(ValueError):
        split_noise_dataset(prepared, val_size=0.0)

    with pytest.raises(ValueError):
        split_noise_dataset(prepared, test_size=0.6, val_size=0.5)