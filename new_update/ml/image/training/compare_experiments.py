"""
SPANDHAN — Image Training: Augmentation Ablation / Experiment Comparison
========================================================================
Runs controlled experiments on the 5-class image signal dataset:
    - Experiment A: No augmentation (Baseline)
    - Experiment B: Small translation (±4 pixels)
    - Experiment C: Small rotation (±5 degrees) + translation

Tests the hypothesis: Because signal patterns (especially chirps) have strict
spatial-frequency trajectories, aggressive augmentations might degrade or distort
the physics of the signal.

Usage:
    python -m ml.image.training.compare_experiments --epochs 20
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.common.config import (
    IMAGE_COMPARISON_REPORT,
    IMAGE_REPORTS_DIR,
    IMAGE_X_PATH,
    IMAGE_Y_PATH,
)
from ml.common.labels import CLASS_NAMES, NUM_CLASSES
from ml.image.cnn_model import SignalCNN
from ml.image.training.split_dataset import make_splits

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AugmentedSignalDataset(Dataset):
    """Dataset with optional on-the-fly augmentation (numpy/torch based)."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mode: str = "none",  # 'none', 'translation', 'rot_trans'
    ) -> None:
        self.X = torch.from_numpy(X).float()  # (N, 1, 128, 128)
        self.y = torch.from_numpy(y).long()
        self.mode = mode

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.X[idx].clone()  # (1, 128, 128)
        label = self.y[idx]

        if self.mode == "translation":
            # Small translation: +/- 4 pixels
            dx = int(torch.randint(-4, 5, (1,)).item())
            dy = int(torch.randint(-4, 5, (1,)).item())
            img = torch.roll(img, shifts=(dy, dx), dims=(1, 2))

        elif self.mode == "rot_trans":
            # Small translation
            dx = int(torch.randint(-3, 4, (1,)).item())
            dy = int(torch.randint(-3, 4, (1,)).item())
            img = torch.roll(img, shifts=(dy, dx), dims=(1, 2))

            # Small affine rotation: -5 to +5 degrees
            angle_deg = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            angle_rad = angle_deg * math.pi / 180.0
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            theta = torch.tensor([
                [cos_a, -sin_a, 0.0],
                [sin_a,  cos_a, 0.0]
            ], dtype=torch.float32).unsqueeze(0)
            grid = nn.functional.affine_grid(theta, img.unsqueeze(0).size(), align_corners=False)
            img = nn.functional.grid_sample(
                img.unsqueeze(0), grid, mode="bilinear", padding_mode="reflection", align_corners=False
            ).squeeze(0)

        return img, label


def train_single_experiment(
    exp_name: str,
    aug_mode: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 25,
    batch_size: int = 32,
) -> dict:
    """Train and evaluate one experiment configuration."""
    print(f"\n[{exp_name}] Augmentation Mode: {aug_mode}")

    train_ds = AugmentedSignalDataset(X_train, y_train, mode=aug_mode)
    val_ds = AugmentedSignalDataset(X_val, y_val, mode="none")
    test_ds = AugmentedSignalDataset(X_test, y_test, mode="none")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    torch.manual_seed(42)
    model = SignalCNN(num_classes=NUM_CLASSES, dropout=0.4).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(yb)

        tr_acc = correct / total

        # Validation
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                v_loss += criterion(logits, yb).item() * len(yb)
                v_corr += (logits.argmax(dim=1) == yb).sum().item()
                v_tot += len(yb)

        val_acc = v_corr / v_tot
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs} - Tr Acc: {tr_acc:.4f} | Val Acc: {val_acc:.4f}")

    if best_weights:
        model.load_state_dict(best_weights)

    # Final evaluation on test split
    model.eval()
    t_corr, t_tot = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
            t_corr += (preds == yb.numpy()).sum()
            t_tot += len(yb)

    test_acc = float(t_corr / t_tot)

    from sklearn.metrics import classification_report
    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    chirp_f1 = report["chirp"]["f1-score"]
    sin_f1 = report["sinusoidal"]["f1-score"]
    macro_f1 = report["macro avg"]["f1-score"]

    print(f"  -> Test Accuracy: {test_acc * 100:.2f}% | Macro F1: {macro_f1:.4f} | Chirp F1: {chirp_f1:.4f}")

    return {
        "experiment": exp_name,
        "augmentation": aug_mode,
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
        "chirp_f1": chirp_f1,
        "sinusoidal_f1": sin_f1,
        "best_val_accuracy": best_val_acc,
    }


def compare_all(epochs: int = 20) -> list[dict]:
    print("=" * 65)
    print("SPANDHAN — Data Augmentation Ablation Study")
    print("=" * 65)

    if not (IMAGE_X_PATH.exists() and IMAGE_Y_PATH.exists()):
        raise FileNotFoundError(
            f"Dataset arrays not found. Run 'python -m ml.image.training.build_dataset' first."
        )

    X = np.load(str(IMAGE_X_PATH))
    y = np.load(str(IMAGE_Y_PATH))

    X_tr, X_val, X_te, y_tr, y_val, y_te, _, _, _ = make_splits(X, y, verbose=False)

    experiments = [
        ("Experiment A", "none", "No augmentation (Baseline)"),
        ("Experiment B", "translation", "Small translation (+/- 4px)"),
        ("Experiment C", "rot_trans", "Small rotation (+/- 5 deg) + translation"),
    ]

    results = []
    for exp_id, aug_mode, desc in experiments:
        res = train_single_experiment(
            f"{exp_id}: {desc}",
            aug_mode,
            X_tr, y_tr,
            X_val, y_val,
            X_te, y_te,
            epochs=epochs,
        )
        results.append(res)

    print("\n" + "=" * 65)
    print("Summary of Augmentation Comparison")
    print("=" * 65)
    print(f"{'Experiment':<32} {'Test Acc':>10} {'Macro F1':>10} {'Chirp F1':>10}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['experiment'][:32]:<32} "
            f"{r['test_accuracy'] * 100:>9.2f}% "
            f"{r['macro_f1']:>10.4f} "
            f"{r['chirp_f1']:>10.4f}"
        )

    IMAGE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(IMAGE_COMPARISON_REPORT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison report saved to: {IMAGE_COMPARISON_REPORT}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    compare_all(epochs=args.epochs)
