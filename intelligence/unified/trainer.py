"""
intelligence/unified/trainer.py
==================================
Train a Random Forest classifier on the unified 1000-sample dataset.

Uses scikit-learn Pipeline:
  StandardScaler → RandomForestClassifier (100 trees)

Also tries GradientBoosting for comparison — keeps the best.

Usage
-----
from intelligence.unified.trainer import train_unified_model
bundle = train_unified_model()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from intelligence.unified.dataset_builder import (
    UnifiedDatasetResult,
    build_unified_dataset,
    load_unified_dataset,
    DEFAULT_DATASET_DIR,
)
from intelligence.unified.model_io import (
    UnifiedModelBundle,
    save_model,
    DEFAULT_MODEL_PATH,
)
from intelligence.unified.feature_extractor import UNIFIED_FEATURE_NAMES


def train_unified_model(
    dataset_result: Optional[UnifiedDatasetResult] = None,
    dataset_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 150,
    save: bool = True,
    verbose: bool = True,
    progress_callback=None,  # callable(step: int, total: int, msg: str)
) -> UnifiedModelBundle:
    """
    Train a noise classifier on the unified dataset.

    Parameters
    ----------
    dataset_result   : pre-built UnifiedDatasetResult (skip loading)
    dataset_dir      : path to saved dataset (loads it if dataset_result is None)
    model_path       : where to save the model
    test_size        : fraction held out for validation
    random_state     : RNG seed
    n_estimators     : number of Random Forest trees
    save             : persist model to disk
    verbose          : print progress
    progress_callback: optional callable(step, total, message)

    Returns
    -------
    UnifiedModelBundle
    """
    def _progress(step, total, msg):
        if verbose:
            print(f"  [{step}/{total}] {msg}")
        if progress_callback:
            progress_callback(step, total, msg)

    total_steps = 6
    t0 = time.perf_counter()

    # ── step 1: load dataset ──────────────────────────────────────
    _progress(1, total_steps, "Loading dataset…")
    if dataset_result is not None:
        ds = dataset_result
    else:
        try:
            ds = load_unified_dataset(dataset_dir)
            if verbose:
                print(f"    Loaded {ds.n_total} samples from disk.")
        except FileNotFoundError:
            if verbose:
                print("    Dataset not found — building now…")
            ds = build_unified_dataset(
                n_audio=500, n_image=500, seed=42,
                output_dir=dataset_dir, save=True, verbose=verbose
            )

    X = ds.features
    y_raw = np.array(ds.labels)

    if X.shape[0] == 0:
        raise RuntimeError("Dataset is empty — cannot train.")

    # ── step 2: encode labels ─────────────────────────────────────
    _progress(2, total_steps, "Encoding labels…")
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    # ── step 3: split ─────────────────────────────────────────────
    _progress(3, total_steps, "Splitting train/val…")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    if verbose:
        print(f"    Train: {len(X_train)} | Val: {len(X_val)}")

    # ── step 4: train Candidate Classifiers & Ensemble ───────────
    _progress(4, total_steps, f"Evaluating Classifiers (ExtraTrees, RandomForest, HistGB, Voting Ensemble)…")
    
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
    from sklearn.preprocessing import RobustScaler

    candidates = {
        "ExtraTrees": Pipeline([
            ("scaler", RobustScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", RobustScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=16,
                min_samples_split=3,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("scaler", RobustScaler()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.05,
                l2_regularization=0.1,
                random_state=random_state,
            )),
        ]),
    }

    best_pipe = None
    best_val = -1.0
    best_name = ""

    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        tr_acc = float(accuracy_score(y_train, pipe.predict(X_train)))
        va_acc = float(accuracy_score(y_val, pipe.predict(X_val)))
        if verbose:
            print(f"    {name:20s} — train acc: {tr_acc:.4f}  val acc: {va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            best_pipe = pipe
            best_name = name

    # Soft Voting Ensemble of the candidates
    ensemble_pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", VotingClassifier(
            estimators=[
                ("et", ExtraTreesClassifier(n_estimators=200, max_depth=20, random_state=random_state, n_jobs=-1)),
                ("rf", RandomForestClassifier(n_estimators=200, max_depth=16, random_state=random_state, n_jobs=-1)),
                ("hgb", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=random_state)),
            ],
            voting="soft",
        )),
    ])
    ensemble_pipe.fit(X_train, y_train)
    ens_tr_acc = float(accuracy_score(y_train, ensemble_pipe.predict(X_train)))
    ens_va_acc = float(accuracy_score(y_val, ensemble_pipe.predict(X_val)))
    if verbose:
        print(f"    {'Voting Ensemble':20s} — train acc: {ens_tr_acc:.4f}  val acc: {ens_va_acc:.4f}")

    if ens_va_acc >= best_val:
        best_val = ens_va_acc
        best_pipe = ensemble_pipe
        best_name = "Voting Ensemble"

    # ── step 5: pick best pipeline ────────────────────────────────
    _progress(5, total_steps, f"Evaluating best model ({best_name})…")

    # Feature importances
    clf_obj = best_pipe.named_steps["clf"]
    if hasattr(clf_obj, "feature_importances_"):
        fi = clf_obj.feature_importances_
    elif hasattr(clf_obj, "estimators_") and hasattr(clf_obj.estimators_[0], "feature_importances_"):
        fi = np.mean([e.feature_importances_ for e in clf_obj.estimators_ if hasattr(e, "feature_importances_")], axis=0)
    else:
        fi = None

    # Full classification report on val set
    y_val_pred = best_pipe.predict(X_val)
    cr = classification_report(y_val, y_val_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_val, y_val_pred)

    if verbose:
        print(f"\n  Best model val accuracy: {best_val:.4f}")
        print(f"\n{cr}")

    # ── step 6: save ──────────────────────────────────────────────
    _progress(6, total_steps, "Saving model…")
    train_meta = {
        "n_estimators": n_estimators,
        "test_size": test_size,
        "random_state": random_state,
        "train_time_s": round(time.perf_counter() - t0, 2),
        "dataset_n_audio": ds.n_audio,
        "dataset_n_image": ds.n_image,
        "class_counts": ds.class_counts,
    }

    bundle = UnifiedModelBundle(
        model=best_pipe,
        feature_names=UNIFIED_FEATURE_NAMES,
        class_names=class_names,
        train_accuracy=float(accuracy_score(y_train, best_pipe.predict(X_train))),
        val_accuracy=best_val,
        confusion_matrix=cm,
        classification_report=cr,
        feature_importances=fi,
        n_train=len(X_train),
        n_val=len(X_val),
        training_metadata=train_meta,
    )

    if save:
        save_model(bundle, model_path)
        if verbose:
            mp = Path(model_path or DEFAULT_MODEL_PATH)
            print(f"\n  Model saved -> {mp}")

    if verbose:
        print(f"\n  Training complete in {time.perf_counter() - t0:.1f}s")

    return bundle
