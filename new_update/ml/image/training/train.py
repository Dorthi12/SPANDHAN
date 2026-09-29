"""
SPANDHAN — Image Training: CNN Trainer
========================================
Trains the SignalCNN on 128x128 grayscale images using PyTorch.

Pipeline:
    1. Load X.npy / y.npy (built by build_dataset.py)
    2. Stratified train / val / test split (70 / 15 / 15)
    3. Train CNN with early stopping on val accuracy
    4. Full evaluation on held-out test set -> confusion matrix + metrics
    5. Save model as models/image/image_signal_classifier.pkl (joblib)

Usage:
    python -m ml.image.training.train

Outputs:
    models/image/image_signal_classifier.pkl
    data/output/image_predictions/training_history.json
    data/output/image_predictions/evaluation_report.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.common.labels import CLASS_NAMES, NUM_CLASSES
from ml.image.cnn_model import SignalCNN
from ml.image.training.build_dataset import (
    build_image_dataset,
    X_PATH, Y_PATH, FILENAMES_PATH, IMAGE_REPORTS_DIR,
)
from ml.image.training.split_dataset import make_splits

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR:   Path = _ROOT / "models" / "image"
MODEL_PATH:   Path = MODELS_DIR / "image_signal_classifier.pkl"
ALT_MODEL_PATH: Path = _ROOT / "models" / "image_signal_classifier.pkl"
HISTORY_PATH: Path = IMAGE_REPORTS_DIR / "training_history.json"
EVAL_PATH:    Path = IMAGE_REPORTS_DIR / "evaluation_report.json"

# ---------------------------------------------------------------------------
# Training hyper-parameters (baseline, experiment A - no augmentation)
# ---------------------------------------------------------------------------
BATCH_SIZE:    int   = 64
MAX_EPOCHS:    int   = 15
LR:            float = 2e-3
WEIGHT_DECAY:  float = 1e-4
PATIENCE:      int   = 5     # early-stopping patience
RANDOM_STATE:  int   = 42
DROPOUT:       float = 0.4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int = RANDOM_STATE) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Wrap numpy arrays in a PyTorch DataLoader."""
    Xt = torch.from_numpy(X).float()    # (N, 1, 128, 128)
    yt = torch.from_numpy(y).long()     # (N,)
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(yb)
        preds      = logits.argmax(dim=1)
        correct    += (preds == yb).sum().item()
        total      += len(yb)

    return total_loss / total, correct / total


@torch.no_grad()
def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        logits = model(Xb)
        loss   = criterion(logits, yb)

        total_loss += loss.item() * len(yb)
        preds      = logits.argmax(dim=1)
        correct    += (preds == yb).sum().item()
        total      += len(yb)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Full confusion-matrix evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def full_evaluation(
    model: nn.Module,
    loader: DataLoader,
    verbose: bool = True,
) -> dict:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
    )

    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    for Xb, yb in loader:
        Xb = Xb.to(DEVICE)
        logits = model(Xb)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(yb.numpy())

    y_pred  = np.concatenate(all_preds)
    y_true  = np.concatenate(all_labels)
    y_proba = np.concatenate(all_probs, axis=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Classification report
    report_dict = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # ROC-AUC (one-vs-rest, macro)
    try:
        roc_auc = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        )
    except Exception:
        roc_auc = None

    accuracy = float(np.mean(y_pred == y_true))

    metrics = {
        "accuracy":    accuracy,
        "roc_auc_macro": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    if verbose:
        print("\n-- Confusion Matrix ---------------------------------------")
        abbrev = [c[:5].capitalize() for c in CLASS_NAMES]
        header = "         " + "  ".join(f"{a:>5}" for a in abbrev)
        print(header)
        print("         " + "-" * (7 * len(CLASS_NAMES)))
        for i, row in enumerate(cm):
            row_str = "  ".join(f"{v:>5}" for v in row)
            print(f"  {CLASS_NAMES[i]:8s} {row_str}")

        print(f"\n  Overall Accuracy : {accuracy*100:.2f}%")
        if roc_auc:
            print(f"  ROC-AUC (macro)  : {roc_auc:.4f}")

        print("\n-- Per-class Metrics -------------------------------------")
        print(f"  {'Class':12s}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'Support':>7}")
        print("  " + "-" * 45)
        for cname in CLASS_NAMES:
            r = report_dict.get(cname, {})
            print(
                f"  {cname:12s}  "
                f"{r.get('precision', 0):9.4f}  "
                f"{r.get('recall', 0):6.4f}  "
                f"{r.get('f1-score', 0):6.4f}  "
                f"{r.get('support', 0):>7}"
            )

    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train() -> None:
    _set_seed(RANDOM_STATE)
    t_start = time.time()
    _banner("SPANDHAN - Train Image Signal Classifier")

    # -- 1. Load dataset -----------------------------------------------------
    print("\n[1/6] Loading dataset ...")
    if X_PATH.exists() and Y_PATH.exists():
        print("  Loading cached arrays ...")
        X = np.load(str(X_PATH))
        y = np.load(str(Y_PATH))
    else:
        print("  Cache not found - loading from disk ...")
        X, y, filenames = build_image_dataset()
        from ml.image.training.build_dataset import save_dataset
        save_dataset(X, y, filenames)
    print(f"  X={X.shape}  y={y.shape}  device={DEVICE}")

    # -- 2. Split -------------------------------------------------------------
    print("\n[2/6] Stratified split ...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     _, _, _) = make_splits(X, y, random_state=RANDOM_STATE)

    train_loader = _make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader   = _make_loader(X_val,   y_val,   BATCH_SIZE, shuffle=False)
    test_loader  = _make_loader(X_test,  y_test,  BATCH_SIZE, shuffle=False)

    # -- 3. Build model -------------------------------------------------------
    print("\n[3/6] Building CNN ...")
    model     = SignalCNN(num_classes=NUM_CLASSES, dropout=DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # -- 4. Training loop with early stopping ---------------------------------
    print(f"\n[4/6] Training (max {MAX_EPOCHS} epochs, patience={PATIENCE}) ...")
    history:     list[dict] = []
    best_val_acc: float     = 0.0
    best_state:  dict | None = None
    epochs_no_improve: int  = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc   = _train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = _evaluate_loader(model, val_loader, criterion)
        scheduler.step(val_acc)

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 5),
            "train_acc":  round(tr_acc,  5),
            "val_loss":   round(val_loss, 5),
            "val_acc":    round(val_acc,  5),
        })

        print(
            f"  Epoch {epoch:3d}/{MAX_EPOCHS}"
            f"  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}"
            f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n  [INFO] Early stopping at epoch {epoch} (best val_acc={best_val_acc:.4f})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)

    # Save history
    IMAGE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Training history -> {HISTORY_PATH}")

    # -- 5. Final test evaluation ---------------------------------------------
    print("\n[5/6] Evaluating on held-out TEST set ...")
    metrics = full_evaluation(model, test_loader, verbose=True)
    with open(EVAL_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Evaluation report -> {EVAL_PATH}")

    # -- 6. Save model as .pkl ------------------------------------------------
    print("\n[6/6] Saving model ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.cpu()

    payload = {
        "model_state_dict": model.state_dict(),
        "class_names":      CLASS_NAMES,
        "num_classes":      NUM_CLASSES,
        "img_height":       128,
        "img_width":        128,
        "channels":         1,
        "dropout":          DROPOUT,
        "best_val_acc":     best_val_acc,
        "test_accuracy":    metrics["accuracy"],
    }
    joblib.dump(payload, str(MODEL_PATH))
    print(f"  [OK] Model saved -> {MODEL_PATH}")

    # Also save to models/image_signal_classifier.pkl for backward compatibility
    joblib.dump(payload, str(ALT_MODEL_PATH))
    print(f"  [OK] Model copy saved -> {ALT_MODEL_PATH}")

    elapsed = time.time() - t_start
    print(f"\nTotal training time: {elapsed / 60:.1f} min")
    _banner("TRAINING COMPLETE")


def _banner(msg: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n{msg}\n{line}")


if __name__ == "__main__":
    train()
