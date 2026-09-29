"""
SPANDHAN — Training: Hyperparameter Tuning
===========================================
Runs GridSearchCV on the best-performing model (from compare_models.py)
using the training split only.

Usage:
    python -m ml.audio.training.tune_model  [--model <name>]
"""

from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path

import numpy as np
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.common.config import (
    X_PATH, Y_PATH, CV_FOLDS, RANDOM_STATE,
    LR_PARAM_GRID, SVM_PARAM_GRID, RF_PARAM_GRID, GB_PARAM_GRID,
    REPORTS_DIR, COMPARISON_REPORT,
)
from ml.audio.training.split import make_splits


_BASE_CLASSIFIERS: dict[str, object] = {
    "LogisticRegression": LogisticRegression(
        multi_class="multinomial", random_state=RANDOM_STATE
    ),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),
}

_PARAM_GRIDS: dict[str, dict] = {
    "LogisticRegression": LR_PARAM_GRID,
    "SVM":                SVM_PARAM_GRID,
    "RandomForest":       RF_PARAM_GRID,
    "GradientBoosting":   GB_PARAM_GRID,
}


def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Pipeline:
    """GridSearchCV for *model_name*. Returns best Pipeline."""
    base_clf  = _BASE_CLASSIFIERS[model_name]
    param_grid = _PARAM_GRIDS[model_name]

    pipe = Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", base_clf),
    ])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipe, param_grid, cv=cv,
        scoring="f1_macro", n_jobs=-1, verbose=1,
        refit=True,
    )
    t0 = time.time()
    gs.fit(X_train, y_train)
    print(f"\n  Best params  : {gs.best_params_}")
    print(f"  Best CV F1   : {gs.best_score_:.4f}")
    print(f"  Tuning time  : {time.time()-t0:.1f}s")
    return gs.best_estimator_


def _read_best_from_report() -> str:
    if COMPARISON_REPORT.exists():
        with open(COMPARISON_REPORT) as f:
            return json.load(f)["best_model"]
    # Default fallback
    return "RandomForest"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Model name (overrides comparison report)")
    args = parser.parse_args()

    X = np.load(str(X_PATH))
    y = np.load(str(Y_PATH))
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    model_name = args.model or _read_best_from_report()
    print("=" * 60)
    print(f"SPANDHAN — Tuning: {model_name}")
    print("=" * 60)

    best_pipe = tune_model(model_name, X_train, y_train)

    # Save tuning result for train_final.py to pick up
    tune_out = REPORTS_DIR / "tuned_model_name.txt"
    tune_out.write_text(model_name)
    print(f"\nTuned model name saved → {tune_out}")
