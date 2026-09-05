import numpy as np
import pytest

from core.config import ML_FEATURE_NAMES
from intelligence.noise.dataset_builder import (
    NOISE_CLASSES,
    NoiseDataset,
    build_noise_dataset,
)


def test_dataset_shape():
    dataset = build_noise_dataset(
        samples_per_class=3,
        length=1000,
        sampling_rate=1000,
        seed=42,
    )

    assert isinstance(dataset, NoiseDataset)

    assert dataset.X.shape == (
        18,
        14,
    )

    assert dataset.y.shape == (18,)

    assert len(dataset.metadata) == 18


def test_all_noise_classes_are_present():
    dataset = build_noise_dataset(
        samples_per_class=2,
        length=1000,
        sampling_rate=1000,
        seed=42,
    )

    assert set(dataset.y) == set(NOISE_CLASSES)

    for noise_class in NOISE_CLASSES:
        assert np.sum(dataset.y == noise_class) == 2


def test_feature_names_match_project_definition():
    dataset = build_noise_dataset(
        samples_per_class=1,
        length=1000,
        sampling_rate=1000,
        seed=42,
    )

    assert dataset.feature_names == ML_FEATURE_NAMES


def test_dataset_is_reproducible():
    dataset_a = build_noise_dataset(
        samples_per_class=2,
        length=1000,
        sampling_rate=1000,
        seed=123,
    )

    dataset_b = build_noise_dataset(
        samples_per_class=2,
        length=1000,
        sampling_rate=1000,
        seed=123,
    )

    assert np.allclose(
        dataset_a.X,
        dataset_b.X,
        equal_nan=True,
    )

    assert np.array_equal(
        dataset_a.y,
        dataset_b.y,
    )


def test_different_seeds_produce_different_features():
    dataset_a = build_noise_dataset(
        samples_per_class=2,
        length=1000,
        sampling_rate=1000,
        seed=1,
    )

    dataset_b = build_noise_dataset(
        samples_per_class=2,
        length=1000,
        sampling_rate=1000,
        seed=2,
    )

    assert not np.allclose(
        dataset_a.X,
        dataset_b.X,
        equal_nan=True,
    )


def test_metadata_is_separate_from_features():
    dataset = build_noise_dataset(
        samples_per_class=2,
        length=1000,
        sampling_rate=1000,
        seed=42,
    )

    assert len(dataset.metadata) == len(dataset.X)

    for item in dataset.metadata:
        assert "generation" in item
        assert "class" in item
        assert "sampling_rate" in item
        assert "length" in item


def test_features_are_finite_except_clean_snr():
    dataset = build_noise_dataset(
        samples_per_class=2,
        length=1000,
        sampling_rate=1000,
        seed=42,
    )

    # The current dataset builder uses +inf for Clean SNR.
    # All other generated feature values must be finite.
    for row, label in zip(dataset.X, dataset.y):
        if label == "Clean":
            assert np.all(
                np.isfinite(row[:-1])
            )
        else:
            assert np.all(
                np.isfinite(row)
            )


def test_invalid_samples_per_class():
    with pytest.raises(ValueError):
        build_noise_dataset(
            samples_per_class=0,
        )


def test_invalid_length():
    with pytest.raises(ValueError):
        build_noise_dataset(
            length=0,
        )


def test_invalid_sampling_rate():
    with pytest.raises(ValueError):
        build_noise_dataset(
            sampling_rate=0,
        )


def test_invalid_seed():
    with pytest.raises(ValueError):
        build_noise_dataset(
            seed="42",
        )