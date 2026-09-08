"""
intelligence/unified/trainer.py
==================================
Domain-Routed Noise Classifier Training.

Two separate, specialized classifiers are trained:
  1. Audio Classifier  — on 37 pure audio wavelet features
  2. Image Classifier  — on 30 pure image texture features

At inference time, the domain tag routes the input to the correct model.
This avoids the zero-padding confusion of a unified feature space.

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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
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
    DomainRoutedBundle,
    save_model,
    DEFAULT_MODEL_PATH,
)
from intelligence.unified.feature_extractor import (
    DOMAIN_AUDIO,
    DOMAIN_IMAGE,
    UNIFIED_FEATURE_DIM,
)

# Number of audio features (excluding domain tag)
N_AUDIO_FEATURES = 37
# Number of image features (excluding domain tag)
N_IMAGE_FEATURES = 30


def _build_ensemble(random_state: int) -> Pipeline:
    """Build the VotingClassifier ensemble Pipeline."""
    return Pipeline([
        ("scaler", RobustScaler()),
        ("clf", VotingClassifier(
            estimators=[
                ("et", ExtraTreesClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                )),
                ("rf", RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                )),
                ("hgb", HistGradientBoostingClassifier(
                    max_iter=400,
                    learning_rate=0.03,
                    max_depth=None,
                    l2_regularization=0.05,
                    random_state=random_state,
                )),
            ],
            voting="soft",
        )),
    ])


def _get_feature_importances(clf_obj) -> Optional[np.ndarray]:
    """Extract feature importances from estimator or ensemble."""
    if hasattr(clf_obj, "feature_importances_"):
        return clf_obj.feature_importances_
    if hasattr(clf_obj, "estimators_"):
        fis = [
            e.feature_importances_
            for e in clf_obj.estimators_
            if hasattr(e, "feature_importances_")
        ]
        if fis:
            return np.mean(fis, axis=0)
    return None


def _train_domain(
    X: np.ndarray,
    y_raw: np.ndarray,
    n_features: int,
    domain_name: str,
    random_state: int,
    test_size: float,
    verbose: bool,
    progress_callback,
    step: int,
    total: int,
) -> dict:
    """Train one domain-specific classifier and return result dict."""

    def _log(msg):
        if verbose:
            print(f"  [{step}/{total}] [{domain_name.upper()}] {msg}")
        if progress_callback:
            progress_callback(step, total, f"[{domain_name.upper()}] {msg}")

    # Slice only the relevant features (drop zero-padded region + domain tag)
    X_dom = X[:, :n_features]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    n_classes = len(np.unique(y))
    n_val = max(1, int(len(y) * test_size))
    # Stratify only when val set is large enough to contain all classes
    strat = y if (len(np.unique(y)) > 1 and n_val >= n_classes) else None
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_dom, y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    _log(f"Train: {len(X_tr)} | Val: {len(X_va)} | Classes: {class_names}")

    pipe = _build_ensemble(random_state)
    pipe.fit(X_tr, y_tr)

    tr_acc = float(accuracy_score(y_tr, pipe.predict(X_tr)))
    va_acc = float(accuracy_score(y_va, pipe.predict(X_va)))
    _log(f"Train acc: {tr_acc:.4f}  Val acc: {va_acc:.4f}")

    y_va_pred = pipe.predict(X_va)
    # Only report on labels actually present in val predictions
    val_labels = sorted(set(y_va) | set(y_va_pred))
    val_names  = [class_names[i] for i in val_labels if i < len(class_names)]
    cr  = classification_report(y_va, y_va_pred, labels=val_labels,
                                 target_names=val_names, zero_division=0)
    cm  = confusion_matrix(y_va, y_va_pred)
    clf = pipe.named_steps["clf"]
    fi  = _get_feature_importances(clf)

    if verbose:
        print(f"\n{cr}")

    return dict(
        model=pipe,
        class_names=class_names,
        train_accuracy=tr_acc,
        val_accuracy=va_acc,
        n_train=len(X_tr),
        n_val=len(X_va),
        classification_report=cr,
        confusion_matrix=cm,
        feature_importances=fi,
    )


def train_unified_model(
    dataset_result: Optional[UnifiedDatasetResult] = None,
    dataset_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 400,
    save: bool = True,
    verbose: bool = True,
    progress_callback=None,
) -> DomainRoutedBundle:
    """
    Train domain-routed noise classifiers (audio + image separately).

    Parameters
    ----------
    dataset_result   : pre-built UnifiedDatasetResult (skip loading)
    dataset_dir      : path to saved dataset
    model_path       : where to save the combined bundle
    test_size        : fraction held out for validation
    random_state     : RNG seed
    n_estimators     : ignored (kept for API compat; ensemble uses 400 trees)
    save             : persist to disk
    verbose          : print progress
    progress_callback: optional callable(step, total, message)

    Returns
    -------
    DomainRoutedBundle
    """
    def _progress(step, total, msg):
        if verbose:
            print(f"  [{step}/{total}] {msg}")
        if progress_callback:
            progress_callback(step, total, msg)

    total_steps = 6
    t0 = time.perf_counter()

    # ── step 1: load dataset ──────────────────────────────────────
    _progress(1, total_steps, "Loading dataset...")
    if dataset_result is not None:
        ds = dataset_result
    else:
        try:
            ds = load_unified_dataset(dataset_dir)
            if verbose:
                print(f"    Loaded {ds.n_total} samples from disk.")
        except FileNotFoundError:
            if verbose:
                print("    Dataset not found - building now...")
            ds = build_unified_dataset(
                n_audio=600, n_image=600, seed=42,
                output_dir=dataset_dir, save=True, verbose=verbose
            )

    X_all    = ds.features                   # shape (N, UNIFIED_FEATURE_DIM)
    y_all    = np.array(ds.labels)           # shape (N,)
    domains  = X_all[:, -1]                  # domain tag: 0=audio, 1=image

    if X_all.shape[0] == 0:
        raise RuntimeError("Dataset is empty - cannot train.")

    # ── step 2: split by domain ───────────────────────────────────
    _progress(2, total_steps, "Splitting data by domain...")
    audio_mask = domains == DOMAIN_AUDIO
    image_mask = domains == DOMAIN_IMAGE

    X_audio = X_all[audio_mask]
    y_audio = y_all[audio_mask]
    X_image = X_all[image_mask]
    y_image = y_all[image_mask]

    if verbose:
        print(f"    Audio samples: {len(X_audio)}  |  Image samples: {len(X_image)}")

    # ── step 3: train audio classifier ───────────────────────────
    _progress(3, total_steps, "Training AUDIO classifier (37 wavelet features)...")
    audio_res = _train_domain(
        X=X_audio, y_raw=y_audio,
        n_features=N_AUDIO_FEATURES,
        domain_name="audio",
        random_state=random_state,
        test_size=test_size,
        verbose=verbose,
        progress_callback=progress_callback,
        step=3, total=total_steps,
    )

    # ── step 4: train image classifier ───────────────────────────
    _progress(4, total_steps, "Training IMAGE classifier (30 texture features)...")
    image_res = _train_domain(
        X=X_image, y_raw=y_image,
        n_features=N_IMAGE_FEATURES,
        domain_name="image",
        random_state=random_state,
        test_size=test_size,
        verbose=verbose,
        progress_callback=progress_callback,
        step=4, total=total_steps,
    )

    # ── step 5: assemble bundle ───────────────────────────────────
    _progress(5, total_steps, "Assembling domain-routed bundle...")
    elapsed = round(time.perf_counter() - t0, 2)

    bundle = DomainRoutedBundle(
        audio_model=audio_res["model"],
        image_model=image_res["model"],
        audio_class_names=audio_res["class_names"],
        audio_train_accuracy=audio_res["train_accuracy"],
        audio_val_accuracy=audio_res["val_accuracy"],
        audio_n_train=audio_res["n_train"],
        audio_n_val=audio_res["n_val"],
        audio_classification_report=audio_res["classification_report"],
        audio_confusion_matrix=audio_res["confusion_matrix"],
        audio_feature_importances=audio_res["feature_importances"],
        image_class_names=image_res["class_names"],
        image_train_accuracy=image_res["train_accuracy"],
        image_val_accuracy=image_res["val_accuracy"],
        image_n_train=image_res["n_train"],
        image_n_val=image_res["n_val"],
        image_classification_report=image_res["classification_report"],
        image_confusion_matrix=image_res["confusion_matrix"],
        image_feature_importances=image_res["feature_importances"],
        training_metadata={
            "train_time_s": elapsed,
            "dataset_n_audio": ds.n_audio,
            "dataset_n_image": ds.n_image,
            "class_counts": ds.class_counts,
            "test_size": test_size,
            "random_state": random_state,
            "n_estimators": n_estimators,
        },
    )

    if verbose:
        print(f"\n  Audio val acc  : {audio_res['val_accuracy']:.4f}")
        print(f"  Image val acc  : {image_res['val_accuracy']:.4f}")
        print(f"  Combined acc   : {bundle.val_accuracy:.4f}")

    # ── step 6: save ──────────────────────────────────────────────
    _progress(6, total_steps, "Saving bundle...")
    if save:
        save_model(bundle, model_path)
        mp = Path(model_path or DEFAULT_MODEL_PATH)
        if verbose:
            print(f"\n  Bundle saved -> {mp}")

    if verbose:
        print(f"\n  Training complete in {time.perf_counter() - t0:.1f}s")

    return bundle
