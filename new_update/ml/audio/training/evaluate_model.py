"""
SPANDHAN — Training: Final Model Evaluation
=============================================
Evaluates a trained Pipeline on the held-out test set.
Produces:
  • per-class precision / recall / F1
  • confusion matrix
  • macro-averaged ROC-AUC
  • inference time per sample

Call evaluate(pipeline, X_test, y_test) from train_final.py.
"""

from __future__ import annotations
import sys, json, time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)
from sklearn.pipeline import Pipeline

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.common.config import REPORTS_DIR
from ml.common.labels import CLASS_NAMES


def evaluate(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    verbose: bool = True,
) -> dict:
    """Evaluate *pipeline* on the test set. Returns metrics dict."""
    # ── Predictions ────────────────────────────────────────────────────────
    t0        = time.perf_counter()
    y_pred    = pipeline.predict(X_test)
    t_predict = time.perf_counter() - t0
    y_proba   = pipeline.predict_proba(X_test)

    # ── Metrics ────────────────────────────────────────────────────────────
    accuracy  = float(accuracy_score(y_test, y_pred))
    report    = classification_report(
        y_test, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    cm        = confusion_matrix(y_test, y_pred).tolist()
    roc_auc   = float(roc_auc_score(
        y_test, y_proba, multi_class="ovr", average="macro"
    ))
    inference_ms = (t_predict / len(y_test)) * 1000   # ms per sample

    metrics = {
        "accuracy":           accuracy,
        "roc_auc_macro":      roc_auc,
        "inference_ms_per_sample": round(inference_ms, 4),
        "classification_report":  report,
        "confusion_matrix":       cm,
    }

    if verbose:
        print("\n── Test-set Evaluation ─────────────────────────────────────")
        print(f"  Accuracy        : {accuracy:.4f}")
        print(f"  ROC-AUC (macro) : {roc_auc:.4f}")
        print(f"  Inference time  : {inference_ms:.4f} ms / sample")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
        print("Confusion Matrix (rows=true, cols=pred):")
        print(f"  Labels: {CLASS_NAMES}")
        for i, row in enumerate(cm):
            print(f"  {CLASS_NAMES[i]:12s}: {row}")

    return metrics


def save_metrics(metrics: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "evaluation_report.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nEvaluation report → {out}")
