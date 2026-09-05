from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from intelligence.noise.dataset_builder import NoiseDataset
from intelligence.noise.features import NOISE_FEATURE_NAMES


@dataclass
class NoiseDatasetSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    metadata_train: list[dict[str, Any]]
    metadata_val: list[dict[str, Any]]
    metadata_test: list[dict[str, Any]]

    feature_names: list[str]


def validate_noise_dataset(dataset: NoiseDataset) -> None:
    """Validate a NoiseDataset before it enters the ML pipeline."""

    if not isinstance(dataset, NoiseDataset):
        raise TypeError("dataset must be a NoiseDataset.")

    X = np.asarray(dataset.X)
    y = np.asarray(dataset.y)

    if X.ndim != 2:
        raise ValueError("Feature matrix X must be two-dimensional.")

    if X.shape[1] != len(NOISE_FEATURE_NAMES):
        raise ValueError(
            f"X must contain exactly {len(NOISE_FEATURE_NAMES)} features."
        )

    if len(y) != len(X):
        raise ValueError("X and y must contain the same number of samples.")

    if len(dataset.metadata) != len(X):
        raise ValueError("Metadata length must match the number of samples.")

    if list(dataset.feature_names) != list(NOISE_FEATURE_NAMES):
        raise ValueError("Feature names/order does not match the ML specification.")

    # Training algorithms cannot safely consume NaN or infinity.
    if not np.all(np.isfinite(X)):
        raise ValueError("Feature matrix X must contain only finite values.")

    if len(X) == 0:
        raise ValueError("Dataset cannot be empty.")

    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError("Dataset must contain at least two classes.")


def prepare_noise_dataset(dataset: NoiseDataset) -> NoiseDataset:
    """
    Prepare a validated dataset for ML.

    Clean samples currently contain +inf SNR because their noise power is
    zero. For ML, this is replaced with a finite sentinel value.
    """

    validate_noise_dataset_allow_clean_inf(dataset)

    X = np.asarray(dataset.X, dtype=np.float64).copy()

    snr_index = dataset.feature_names.index("snr_db")

    clean_mask = np.isinf(X[:, snr_index])
    if np.any(clean_mask):
        if not np.all(X[clean_mask, snr_index] > 0):
            raise ValueError("Only positive infinity is allowed for Clean SNR.")

        # A finite, explicit representation for ideal/no-noise samples.
        X[clean_mask, snr_index] = 100.0

    if not np.all(np.isfinite(X)):
        raise ValueError("Prepared feature matrix contains non-finite values.")

    return NoiseDataset(
        X=X,
        y=np.asarray(dataset.y).copy(),
        feature_names=list(dataset.feature_names),
        metadata=list(dataset.metadata),
    )


def validate_noise_dataset_allow_clean_inf(dataset: NoiseDataset) -> None:
    """Validate the raw dataset while allowing +inf only in the SNR feature."""

    if not isinstance(dataset, NoiseDataset):
        raise TypeError("dataset must be a NoiseDataset.")

    X = np.asarray(dataset.X)

    if X.ndim != 2:
        raise ValueError("Feature matrix X must be two-dimensional.")

    if X.shape[1] != len(NOISE_FEATURE_NAMES):
        raise ValueError(
            f"X must contain exactly {len(NOISE_FEATURE_NAMES)} features."
        )

    if len(dataset.y) != len(X):
        raise ValueError("X and y must contain the same number of samples.")

    if len(dataset.metadata) != len(X):
        raise ValueError("Metadata length must match the number of samples.")

    if list(dataset.feature_names) != list(NOISE_FEATURE_NAMES):
        raise ValueError("Feature names/order does not match the ML specification.")

    if len(X) == 0:
        raise ValueError("Dataset cannot be empty.")

    # Every non-SNR feature must already be finite.
    snr_index = dataset.feature_names.index("snr_db")
    other_columns = np.delete(X, snr_index, axis=1)

    if not np.all(np.isfinite(other_columns)):
        raise ValueError("Non-SNR features must contain only finite values.")

    snr = X[:, snr_index]
    if not np.all(np.isfinite(snr) | np.isposinf(snr)):
        raise ValueError("SNR may contain finite values or positive infinity only.")

    if len(np.unique(dataset.y)) < 2:
        raise ValueError("Dataset must contain at least two classes.")


def split_noise_dataset(
    dataset: NoiseDataset,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> NoiseDatasetSplit:
    """
    Create reproducible stratified train/validation/test splits.

    Default proportions:
        train = 70%
        validation = 15%
        test = 15%
    """

    if not isinstance(test_size, float) or not 0 < test_size < 1:
        raise ValueError("test_size must be a float between 0 and 1.")

    if not isinstance(val_size, float) or not 0 < val_size < 1:
        raise ValueError("val_size must be a float between 0 and 1.")

    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1.")

    if not isinstance(random_state, int):
        raise ValueError("random_state must be an integer.")

    prepared = prepare_noise_dataset(dataset)

    X = prepared.X
    y = prepared.y
    metadata = np.asarray(prepared.metadata, dtype=object)

    X_train, X_temp, y_train, y_temp, meta_train, meta_temp = train_test_split(
        X,
        y,
        metadata,
        test_size=test_size + val_size,
        stratify=y,
        random_state=random_state,
    )

    relative_val_size = val_size / (test_size + val_size)

    X_val, X_test, y_val, y_test, meta_val, meta_test = train_test_split(
        X_temp,
        y_temp,
        meta_temp,
        test_size=1.0 - relative_val_size,
        stratify=y_temp,
        random_state=random_state,
    )

    return NoiseDatasetSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        metadata_train=list(meta_train),
        metadata_val=list(meta_val),
        metadata_test=list(meta_test),
        feature_names=list(prepared.feature_names),
    )