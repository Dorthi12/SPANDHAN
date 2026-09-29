"""
SPANDHAN — Image CNN Model Definition (PyTorch)
================================================
Defines the CNN architecture for 128×128 grayscale image classification.

Architecture:
    Conv2D(32)  → ReLU → MaxPool
    Conv2D(64)  → ReLU → MaxPool
    Conv2D(128) → ReLU → MaxPool
    GlobalAveragePooling
    Dense(128)  → ReLU → Dropout(0.4)
    Dense(5)    → Softmax  [5 signal classes]

Input:  (N, 1, 128, 128)  float32, values in [0, 1]
Output: (N, 5)            log-softmax probabilities

Serialization: This module is pickle-safe so the whole
               SignalCNN instance can be saved via joblib.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SignalCNN(nn.Module):
    """
    Baseline CNN for 128×128 grayscale signal images.

    Follows the exact architecture prescribed in the SPANDHAN image-ML spec:
        Conv32 → Conv64 → Conv128 → GlobalAvgPool → Dense128 → Dense5
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.4) -> None:
        super().__init__()

        # ── Convolutional backbone ──────────────────────────────────────────
        self.features = nn.Sequential(
            # Block 1 – 128×128 → 64×64
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                      # 64×64

            # Block 2 – 64×64 → 32×32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                      # 32×32

            # Block 3 – 32×32 → 16×16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                      # 16×16
        )

        # ── Global Average Pooling ──────────────────────────────────────────
        # Collapses 128×16×16 → 128×1×1 → 128 (no spatial info leakage)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Classifier head ─────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x   # raw logits (CrossEntropyLoss expects logits)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (for inference)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
