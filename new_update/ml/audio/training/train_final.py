"""
SPANDHAN - Audio Signal Classifier: Full Training Pipeline
===========================================================
Audio-only ML model. Completely separate from image ML model.
Reads WAVs from:  datasets/audio/{impulse,sinusoidal,white_noise,step,chirp}/
Saves model to:   models/audio/audio_signal_classifier.pkl

Run once:
    python -m ml.audio.training.train_final

Pipeline steps
--------------
 1. Load all audio WAVs, extract 35 features each
 2. Stratified 70/15/15 train/val/test split
 3. 5-fold CV comparison: LR, SVM, RandomForest, GradientBoosting
 4. GridSearchCV tuning on best model
 5. Re-train on train+val combined
 6. Probability calibration on val set
 7. Final evaluation on held-out test set
 8. Save calibrated sklearn Pipeline -> models/audio/audio_signal_classifier.pkl

Output files
------------
 models/audio/audio_signal_classifier.pkl   <- inference bundle (scaler + classifier)
 data/output/predictions/X.npy              <- feature matrix (for reference)
 data/output/predictions/y.npy              <- labels
 data/output/predictions/model_comparison.json
 data/output/predictions/evaluation_report.json
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import joblib
from scipy.io import wavfile

from sklearn.calibration      import CalibratedClassifierCV
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (accuracy_score, classification_report,
                                      confusion_matrix, roc_auc_score)
from sklearn.model_selection  import (GridSearchCV, StratifiedKFold,
                                      cross_validate, train_test_split)
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler
from sklearn.svm              import SVC

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT        = Path(__file__).resolve().parents[3]
DATASET_ROOT = _ROOT / "datasets" / "audio"
MODELS_DIR   = _ROOT / "models" / "audio"
REPORTS_DIR  = _ROOT / "data" / "output" / "predictions"
PKL_PATH     = MODELS_DIR / "audio_signal_classifier.pkl"

CLASS_NAMES  = ["impulse", "sinusoidal", "white_noise", "step", "chirp"]
CLASS_DIRS   = {c: DATASET_ROOT / c for c in CLASS_NAMES}

RANDOM_STATE = 42
CV_FOLDS     = 5

# ---------------------------------------------------------------------------
# Feature extraction  (import from shared module)
# ---------------------------------------------------------------------------
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from ml.audio.features import extract_features, NUM_FEATURES


# ---------------------------------------------------------------------------
# Step 1 - Load dataset
# ---------------------------------------------------------------------------
def _load_wav(path: Path):
    sr, data = wavfile.read(str(path))
    if data.dtype.kind == "i":
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, int(sr)


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    x_cache = REPORTS_DIR / "X.npy"
    y_cache = REPORTS_DIR / "y.npy"
    if x_cache.exists() and y_cache.exists():
        print(f"\n[1/7] Loading existing features from {REPORTS_DIR} ...")
        X = np.load(str(x_cache))
        y = np.load(str(y_cache))
        print(f"  Loaded X={X.shape}, y={y.shape} ({len(y)} audio samples)")
        return X, y

    print("\n[1/7] Extracting features from audio WAVs ...")
    X_rows, y_rows = [], []
    total_ok = total_err = 0

    for class_id, cls in enumerate(CLASS_NAMES):
        cls_dir = CLASS_DIRS[cls]
        if not cls_dir.exists():
            print(f"  WARNING: missing directory {cls_dir}")
            continue
        wavs = sorted(cls_dir.glob("*.wav"))
        print(f"  [{cls:12s}] {len(wavs)} files", flush=True)
        ok = err = 0
        for wav in wavs:
            try:
                audio, sr = _load_wav(wav)
                feat      = extract_features(audio, sr)
                X_rows.append(feat)
                y_rows.append(class_id)
                ok += 1
            except Exception as e:
                err += 1
                print(f"    ERR {wav.name}: {e}")
        total_ok  += ok
        total_err += err
        print(f"    >> {ok} ok, {err} errors")

    X = np.vstack(X_rows).astype(np.float64)
    y = np.array(y_rows, dtype=np.int64)
    print(f"\n  Total: {len(y)} samples, {NUM_FEATURES} features each")
    print(f"  Errors: {total_err}")

    # Save for reference (not required for inference)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(REPORTS_DIR / "X.npy"), X)
    np.save(str(REPORTS_DIR / "y.npy"), y)
    print(f"  Saved X.npy / y.npy to {REPORTS_DIR}")
    return X, y


# ---------------------------------------------------------------------------
# Step 2 - Split
# ---------------------------------------------------------------------------
def make_splits(X, y):
    print("\n[2/7] Stratified 70/15/15 split ...")
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=(0.15 / 0.85), stratify=y_tv, random_state=RANDOM_STATE)
    print(f"  Train={len(y_train)}  Val={len(y_val)}  Test={len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Step 3 - Model comparison
# ---------------------------------------------------------------------------
CANDIDATES = {
    "LogisticRegression": LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000,
        random_state=RANDOM_STATE),
    "SVM": SVC(
        C=10.0, kernel="rbf", gamma="scale",
        probability=True, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5,
        random_state=RANDOM_STATE),
}

PARAM_GRIDS = {
    "LogisticRegression": {
        "classifier__C":        [0.01, 0.1, 1.0, 10.0],
        "classifier__solver":   ["lbfgs", "saga"],
        "classifier__max_iter": [1000],
    },
    "SVM": {
        "classifier__C":      [0.1, 1.0, 10.0],
        "classifier__kernel": ["rbf", "poly"],
        "classifier__gamma":  ["scale", "auto"],
    },
    "RandomForest": {
        "classifier__n_estimators":      [100, 200, 300],
        "classifier__max_depth":         [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
    },
    "GradientBoosting": {
        "classifier__n_estimators":  [100],
        "classifier__learning_rate": [0.1, 0.2],
        "classifier__max_depth":     [5, 7],
    },
}


def compare_models(X_train, y_train) -> tuple[dict, str]:
    report_file = REPORTS_DIR / "model_comparison.json"
    if report_file.exists():
        with open(report_file) as f:
            data = json.load(f)
        results = data.get("results", {})
        best = data.get("best_model", "GradientBoosting")
        print("\n[3/7] Using 5-fold CV comparison results ...")
        for name, r in results.items():
            print(f"  {name:22s}  acc={r['accuracy_mean']:.4f}+-{r['accuracy_std']:.4f}"
                  f"  f1={r['f1_macro_mean']:.4f}+-{r['f1_macro_std']:.4f}"
                  f"  ({r.get('fit_time_s', 0):.1f}s)")
        print(f"\n  Best candidate: {best}")
        return results, best

    print("\n[3/7] 5-fold CV comparison of 4 candidates ...")
    cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, clf in CANDIDATES.items():
        pipe   = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
        t0     = time.time()
        scores = cross_validate(pipe, X_train, y_train, cv=cv,
                                scoring=["accuracy", "f1_macro"],
                                n_jobs=-1)
        elapsed = time.time() - t0
        results[name] = {
            "accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "accuracy_std":  float(np.std(scores["test_accuracy"])),
            "f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
            "f1_macro_std":  float(np.std(scores["test_f1_macro"])),
            "fit_time_s":    round(elapsed, 2),
        }
        r = results[name]
        print(f"  {name:22s}  acc={r['accuracy_mean']:.4f}+-{r['accuracy_std']:.4f}"
              f"  f1={r['f1_macro_mean']:.4f}+-{r['f1_macro_std']:.4f}"
              f"  ({elapsed:.1f}s)")

    best = max(results, key=lambda k: results[k]["f1_macro_mean"])
    print(f"\n  Best candidate: {best}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump({"best_model": best, "results": results}, f, indent=2)

    return results, best


# ---------------------------------------------------------------------------
# Step 4 - Hyperparameter tuning
# ---------------------------------------------------------------------------
def tune_model(best_name: str, X_train, y_train) -> Pipeline:
    print(f"\n[4/7] GridSearchCV tuning: {best_name} ...")
    clf  = CANDIDATES[best_name]
    pipe = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs   = GridSearchCV(pipe, PARAM_GRIDS[best_name],
                        cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0, refit=True)
    gs.fit(X_train, y_train)
    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV F1  : {gs.best_score_:.4f}")
    return gs.best_estimator_


# ---------------------------------------------------------------------------
# Step 5+6 - Re-train on train+val, then calibrate on val
# ---------------------------------------------------------------------------
def retrain_and_calibrate(
    tuned_pipe: Pipeline,
    X_train, X_val, y_train, y_val
):
    print("\n[5/7] Re-training on train+val combined ...")
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    tuned_pipe.fit(X_tv, y_tv)

    print("[6/7] Probability calibration (sigmoid, 3-fold CV) ...")
    try:
        calibrated = CalibratedClassifierCV(tuned_pipe, method="sigmoid", cv=3)
        calibrated.fit(X_tv, y_tv)
        print("  Calibrated model ready.")
        return calibrated
    except Exception as e:
        print(f"  Note on calibration: {e}. Using native softmax probabilities.")
        return tuned_pipe


# ---------------------------------------------------------------------------
# Step 7 - Evaluate on test set
# ---------------------------------------------------------------------------
def evaluate(model, X_test, y_test) -> dict:
    print("\n[7/7] Final evaluation on held-out test set ...")
    t0      = time.perf_counter()
    y_pred  = model.predict(X_test)
    t_inf   = time.perf_counter() - t0
    y_proba = model.predict_proba(X_test)

    acc     = float(accuracy_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
    cm      = confusion_matrix(y_test, y_pred).tolist()
    report  = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    inf_ms  = (t_inf / len(y_test)) * 1000

    metrics = {
        "accuracy":                acc,
        "roc_auc_macro":           roc_auc,
        "inference_ms_per_sample": round(inf_ms, 4),
        "classification_report":   report,
        "confusion_matrix":        cm,
    }

    print(f"\n  Accuracy        : {acc:.4f}")
    print(f"  ROC-AUC (macro) : {roc_auc:.4f}")
    print(f"  Inference time  : {inf_ms:.4f} ms/sample")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    print("  Confusion Matrix (rows=true, cols=pred):")
    print(f"  Labels: {CLASS_NAMES}")
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:12s}: {row}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Evaluation report -> {REPORTS_DIR / 'evaluation_report.json'}")
    return metrics


# ---------------------------------------------------------------------------
# Step 8 - Save model
# ---------------------------------------------------------------------------
def save_model(model) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(PKL_PATH))
    size_mb = PKL_PATH.stat().st_size / 1e6
    print(f"\n  Model saved -> {PKL_PATH}  ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    print("=" * 60)
    print("SPANDHAN - Audio Signal Classifier Training")
    print("Dataset : datasets/audio/  (5 classes x 1000 WAVs)")
    print("Output  : models/audio/audio_signal_classifier.pkl")
    print("=" * 60)

    X, y = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
    _, best_name   = compare_models(X_train, y_train)
    tuned_pipe     = tune_model(best_name, X_train, y_train)
    calibrated     = retrain_and_calibrate(tuned_pipe, X_train, X_val, y_train, y_val)
    _              = evaluate(calibrated, X_test, y_test)
    save_model(calibrated)

    elapsed = time.time() - t_start
    print(f"\nTotal training time: {elapsed/60:.1f} min")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
