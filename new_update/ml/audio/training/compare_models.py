"""
SPANDHAN — Training: Compare Candidate Models
===============================================
Runs 5-fold cross-validation on all four candidate classifiers
(LR, SVM, RF, GB) using the train split only.
Prints a comparison table and saves a JSON report.

Usage:
    python -m ml.audio.training.compare_models
    (requires X.npy / y.npy to exist — run build_dataset.py first)
"""

from __future__ import annotations
import sys, json, time
from pathlib import Path

import numpy as np
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.common.config  import (
    X_PATH, Y_PATH, CV_FOLDS, RANDOM_STATE,
    COMPARISON_REPORT, REPORTS_DIR,
)
from ml.audio.training.split import make_splits


CANDIDATES: dict[str, object] = {
    "LogisticRegression": LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000,
        multi_class="multinomial", random_state=RANDOM_STATE,
    ),
    "SVM": SVC(
        C=10.0, kernel="rbf", gamma="scale",
        probability=True, random_state=RANDOM_STATE,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5,
        random_state=RANDOM_STATE,
    ),
}

METRICS = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]


def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True,
) -> dict[str, dict]:
    """Cross-validate every candidate on X_train / y_train. Return results dict."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, dict] = {}

    for name, clf in CANDIDATES.items():
        pipe = Pipeline([
            ("scaler",     StandardScaler()),
            ("classifier", clf),
        ])
        t0 = time.time()
        scores = cross_validate(
            pipe, X_train, y_train,
            cv=cv,
            scoring=METRICS,
            n_jobs=-1,
            return_train_score=False,
        )
        elapsed = time.time() - t0
        summary = {
            metric: {
                "mean": float(np.mean(scores[f"test_{metric}"])),
                "std":  float(np.std(scores[f"test_{metric}"])),
            }
            for metric in METRICS
        }
        summary["fit_time_s"] = round(elapsed, 2)
        results[name] = summary

        if verbose:
            acc  = summary["accuracy"]
            f1   = summary["f1_macro"]
            print(
                f"  {name:22s}  "
                f"acc={acc['mean']:.4f}±{acc['std']:.4f}  "
                f"f1={f1['mean']:.4f}±{f1['std']:.4f}  "
                f"({elapsed:.1f}s)"
            )

    return results


def pick_best(results: dict[str, dict]) -> str:
    """Return the model name with highest mean F1-macro."""
    return max(results, key=lambda k: results[k]["f1_macro"]["mean"])


def save_report(results: dict, best: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"best_model": best, "results": results}
    with open(COMPARISON_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nComparison report → {COMPARISON_REPORT}")


if __name__ == "__main__":
    print("=" * 60)
    print("SPANDHAN — Model Comparison  (5-fold CV on train split)")
    print("=" * 60)

    X = np.load(str(X_PATH))
    y = np.load(str(Y_PATH))
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    print("\nCross-validating candidates …\n")
    results = compare_models(X_train, y_train)
    best    = pick_best(results)
    save_report(results, best)
    print(f"\n✓ Best model: {best}")
