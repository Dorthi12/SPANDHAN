"""
SPANDHAN — Image Training: Model Evaluation
=============================================
Evaluates the trained image signal classifier on held-out test data.

Produces:
    - Overall accuracy
    - Formatted Confusion Matrix
    - Precision, Recall, F1 per class
    - Macro / Weighted F1
    - Multi-class ROC-AUC (OVR)
    - Error Analysis (e.g., chirp <-> sinusoidal, impulse <-> white noise)
    - Optional Out-of-Distribution (OOD) testing

Usage:
    python -m ml.image.training.evaluate
    python -m ml.image.training.evaluate --model path/to/model.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.common.config import (
    IMAGE_EVALUATION_REPORT,
    IMAGE_PIPELINE_PATH,
    IMAGE_REPORTS_DIR,
    IMAGE_X_PATH,
    IMAGE_Y_PATH,
)
from ml.common.labels import CLASS_NAMES, NUM_CLASSES
from ml.image.cnn_model import SignalCNN
from ml.image.training.split_dataset import make_splits

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(model_path: Path | str | None = None) -> tuple[SignalCNN, dict]:
    """Load model weights and metadata from .pkl file."""
    path = Path(model_path) if model_path else IMAGE_PIPELINE_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}.\n"
            "Run 'python -m ml.image.training.train' first."
        )

    payload = joblib.load(str(path))
    model = SignalCNN(
        num_classes=payload.get("num_classes", NUM_CLASSES),
        dropout=payload.get("dropout", 0.4),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, payload


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    verbose: bool = True,
) -> dict:
    """Compute detailed evaluation metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    try:
        roc_auc = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        )
    except Exception:
        roc_auc = None

    accuracy = float(np.mean(y_true == y_pred))

    # Error analysis: check significant off-diagonal pairs
    confusions = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "true_class": CLASS_NAMES[i],
                    "predicted_class": CLASS_NAMES[j],
                    "count": int(cm[i, j]),
                    "rate": float(cm[i, j] / max(np.sum(cm[i, :]), 1)),
                })
    confusions.sort(key=lambda x: x["count"], reverse=True)

    metrics = {
        "accuracy": accuracy,
        "roc_auc_macro": roc_auc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "top_confusions": confusions[:5],
    }

    if verbose:
        print("\n" + "=" * 65)
        print("SPANDHAN - Image Classifier Evaluation")
        print("=" * 65)
        print(f"Overall Accuracy: {accuracy * 100:.2f}%")
        if roc_auc is not None:
            print(f"ROC-AUC (Macro):  {roc_auc:.4f}")
        print(f"Macro F1-Score:   {metrics['macro_f1']:.4f}")
        print(f"Weighted F1:      {metrics['weighted_f1']:.4f}")

        print("\n-- Confusion Matrix -------------------------------------------")
        abbrev = [c[:5].capitalize() for c in CLASS_NAMES]
        header = "True \\ Pred   " + "  ".join(f"{a:>7}" for a in abbrev)
        print(header)
        print("-" * len(header))
        for i, row in enumerate(cm):
            row_str = "  ".join(f"{v:>7d}" for v in row)
            print(f"{CLASS_NAMES[i]:12s}  {row_str}")

        print("\n-- Per-Class Metrics ------------------------------------------")
        print(f"{'Class':14s} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 58)
        for name in CLASS_NAMES:
            stats = report[name]
            print(
                f"{name:14s} {stats['precision']:10.4f} {stats['recall']:10.4f} "
                f"{stats['f1-score']:10.4f} {int(stats['support']):10d}"
            )

        if confusions:
            print("\n-- Top Misclassifications -------------------------------------")
            for c in confusions[:5]:
                print(
                    f"  {c['true_class']:12s} -> {c['predicted_class']:12s}: "
                    f"{c['count']:3d} samples ({c['rate'] * 100:.1f}%)"
                )
        else:
            print("\n  Clean classification: 0 misclassifications on this set.")

    return metrics


def evaluate_ood(model: SignalCNN, ood_dir: Path | str) -> dict:
    """Evaluate classifier on Out-Of-Distribution (OOD) test images."""
    from PIL import Image

    ood_dir = Path(ood_dir)
    if not ood_dir.exists():
        return {}

    print(f"\n── OOD Evaluation ({ood_dir.name}) ──────────────────────────")
    results = {}
    model.eval()

    subdirs = [d for d in ood_dir.iterdir() if d.is_dir()]
    if not subdirs:
        subdirs = [ood_dir]

    for sd in subdirs:
        pngs = list(sd.glob("*.png")) + list(sd.glob("*.jpg"))
        if not pngs:
            continue

        preds_count = {name: 0 for name in CLASS_NAMES}
        max_confs = []

        for p in pngs:
            try:
                img = Image.open(p).convert("L").resize((128, 128))
                arr = (np.array(img, dtype=np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
                inp = torch.from_numpy(arr).to(DEVICE)
                with torch.no_grad():
                    probs = model.predict_proba(inp).cpu().numpy()[0]
                pred_cls = CLASS_NAMES[int(probs.argmax())]
                preds_count[pred_cls] += 1
                max_confs.append(float(probs.max()))
            except Exception:
                continue

        avg_conf = float(np.mean(max_confs)) if max_confs else 0.0
        results[sd.name] = {
            "num_samples": len(pngs),
            "distribution": preds_count,
            "avg_confidence": avg_conf,
        }
        print(f"  Folder: {sd.name} ({len(pngs)} images, Avg Confidence: {avg_conf:.3f})")
        for cls_name, cnt in preds_count.items():
            if cnt > 0:
                print(f"    {cls_name:12s}: {cnt}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Image Classifier")
    parser.add_argument("--model", type=str, default=str(IMAGE_PIPELINE_PATH))
    parser.add_argument("--save-report", action="store_true", default=True)
    args = parser.parse_args()

    model, payload = load_trained_model(args.model)

    if not (IMAGE_X_PATH.exists() and IMAGE_Y_PATH.exists()):
        raise FileNotFoundError(
            f"Dataset arrays not found at {IMAGE_X_PATH}. "
            "Run 'python -m ml.image.training.build_dataset' first."
        )

    X = np.load(str(IMAGE_X_PATH))
    y = np.load(str(IMAGE_Y_PATH))

    # Get test split
    _, _, X_test, _, _, y_test, _, _, _ = make_splits(X, y, verbose=False)

    Xt = torch.from_numpy(X_test).float()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, torch.from_numpy(y_test).long()),
        batch_size=32,
        shuffle=False,
    )

    all_preds = []
    all_probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            probs = model.predict_proba(xb).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.append(probs)
            all_preds.append(preds)

    y_pred = np.concatenate(all_preds)
    y_proba = np.concatenate(all_probs, axis=0)

    metrics = evaluate_predictions(y_test, y_pred, y_proba, verbose=True)

    # Check for OOD datasets
    ood_path = _ROOT / "datasets" / "unknown_test" / "image"
    if ood_path.exists():
        ood_results = evaluate_ood(model, ood_path)
        metrics["ood_evaluation"] = ood_results

    if args.save_report:
        IMAGE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(IMAGE_EVALUATION_REPORT, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nEvaluation report saved to: {IMAGE_EVALUATION_REPORT}")


if __name__ == "__main__":
    main()
