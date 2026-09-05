from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from intelligence.noise.dataset import NoiseDatasetSplit


@dataclass
class NoiseTrainingResult:
    model: Pipeline
    training_accuracy: float
    validation_accuracy: float
    feature_names: list[str]
    best_params: dict[str, Any] = field(default_factory=dict)


def _validate_split(dataset: NoiseDatasetSplit) -> None:
    """Validate a prepared dataset split before training."""

    if not isinstance(dataset, NoiseDatasetSplit):
        raise TypeError("dataset must be a NoiseDatasetSplit.")

    arrays = [
        dataset.X_train,
        dataset.X_val,
        dataset.X_test,
        dataset.y_train,
        dataset.y_val,
        dataset.y_test,
    ]

    if any(np.asarray(array).size == 0 for array in arrays):
        raise ValueError("Dataset splits cannot be empty.")

    for name, X in [
        ("X_train", dataset.X_train),
        ("X_val", dataset.X_val),
        ("X_test", dataset.X_test),
    ]:
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional.")

        if X.shape[1] != len(dataset.feature_names):
            raise ValueError(
                f"{name} has an incorrect number of features."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(f"{name} must contain only finite values.")

    if len(dataset.X_train) != len(dataset.y_train):
        raise ValueError("Training features and labels must have equal lengths.")

    if len(dataset.X_val) != len(dataset.y_val):
        raise ValueError("Validation features and labels must have equal lengths.")

    if len(dataset.X_test) != len(dataset.y_test):
        raise ValueError("Test features and labels must have equal lengths.")

    if len(np.unique(dataset.y_train)) < 2:
        raise ValueError("Training data must contain at least two classes.")


def create_svm_pipeline(
    random_state: int = 42,
    C: float = 10.0,
    gamma: str | float = "scale",
) -> Pipeline:
    """
    Create the primary noise-classification pipeline.

    StandardScaler is deliberately inside the Pipeline so that it is fitted
    only on the training data when Pipeline.fit() is called.
    """

    if not isinstance(random_state, int):
        raise ValueError("random_state must be an integer.")

    if C <= 0:
        raise ValueError("C must be greater than zero.")

    if isinstance(gamma, str):
        if gamma not in {"scale", "auto"}:
            raise ValueError("gamma must be 'scale', 'auto', or a positive float.")
    elif gamma <= 0:
        raise ValueError("Numeric gamma must be greater than zero.")
    else:
        gamma = float(gamma)

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=SVC(
                        kernel="rbf",
                        C=C,
                        gamma=gamma,
                        random_state=random_state,
                    ),
                    cv=5,
                    method="isotonic",
                ),
            ),
        ]
    )


def train_svm(
    dataset: NoiseDatasetSplit,
    random_state: int = 42,
    C: float = 10.0,
    gamma: str | float = "scale",
) -> NoiseTrainingResult:
    """Train the primary RBF-SVM noise classifier with GridSearchCV tuning."""

    _validate_split(dataset)

    # --- Hyperparameter search on training data only ---
    # Build a lightweight pipeline (no calibration) for the search.
    search_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", random_state=random_state)),
    ])

    param_grid = {
        "svc__C":     [1.0, 10.0, 50.0, 100.0],
        "svc__gamma": ["scale", "auto", 0.01, 0.001],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        search_pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(dataset.X_train, dataset.y_train)

    best_C     = grid_search.best_params_["svc__C"]
    best_gamma = grid_search.best_params_["svc__gamma"]

    # --- Build calibrated final model with best hyperparameters ---
    model = create_svm_pipeline(
        random_state=random_state,
        C=best_C,
        gamma=best_gamma,
    )

    model.fit(dataset.X_train, dataset.y_train)

    training_accuracy = float(
        model.score(dataset.X_train, dataset.y_train)
    )

    validation_accuracy = float(
        model.score(dataset.X_val, dataset.y_val)
    )

    return NoiseTrainingResult(
        model=model,
        training_accuracy=training_accuracy,
        validation_accuracy=validation_accuracy,
        feature_names=list(dataset.feature_names),
        best_params={"C": best_C, "gamma": best_gamma},
    )