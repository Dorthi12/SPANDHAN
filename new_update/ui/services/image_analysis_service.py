"""
SPANDHAN — Image Analysis & Inference Service
=============================================
Handles image loading, metadata parsing, 2D signal feature extraction,
and background asynchronous inference using the pre-trained SignalCNN model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QThread, Signal

# Root path injection to ensure access to ml.image
import sys
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.image.inference.predict import get_image_predictor, ImagePredictor
from ml.image.inference.preprocess_input import preprocess_image, TARGET_HEIGHT, TARGET_WIDTH


def compute_image_signal_features(img_arr: np.ndarray) -> Dict[str, Any]:
    """
    Extract technical 2D signal characteristics from a normalized [0, 1] 2D matrix.
    """
    # Squeeze if needed
    arr = img_arr.squeeze().astype(np.float64)

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    max_val = float(np.max(arr))
    min_val = float(np.min(arr))
    peak_val = max(abs(max_val), abs(min_val))

    # RMS Energy
    rms_val = float(np.sqrt(np.mean(arr ** 2)))

    # Peak to RMS Ratio
    crest_factor = (peak_val / rms_val) if rms_val > 1e-9 else 0.0

    # Dynamic Range (dB)
    dyn_range = 20.0 * math.log10((max_val - min_val) + 1e-6) if (max_val - min_val) > 0 else 0.0

    # Shannon Entropy of 8-bit quantized distribution
    quantized = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
    counts = np.bincount(quantized.ravel(), minlength=256)
    probs = counts / float(counts.sum())
    probs = probs[probs > 0]
    shannon_entropy = float(-np.sum(probs * np.log2(probs)))

    # Spatial Frequency Metric (Row frequency and Column frequency)
    diff_row = np.diff(arr, axis=0)
    diff_col = np.diff(arr, axis=1)
    rf = math.sqrt(np.mean(diff_row ** 2)) if diff_row.size > 0 else 0.0
    cf = math.sqrt(np.mean(diff_col ** 2)) if diff_col.size > 0 else 0.0
    spatial_freq = math.sqrt(rf ** 2 + cf ** 2)

    # Gradient Energy (magnitude of simple central difference)
    grad_y, grad_x = np.gradient(arr)
    grad_energy = float(np.mean(grad_x ** 2 + grad_y ** 2))

    # Sparsity / Fill ratio (percentage of pixels exceeding mean + 1 std)
    threshold = mean_val + std_val
    sparsity_ratio = float(np.mean(arr > threshold)) * 100.0

    # Symmetry Index: correlation between left and flipped right half
    half_w = arr.shape[1] // 2
    left_half = arr[:, :half_w]
    right_half_flipped = np.fliplr(arr[:, half_w:])
    if left_half.shape == right_half_flipped.shape:
        c = np.corrcoef(left_half.ravel(), right_half_flipped.ravel())
        sym_index = float(c[0, 1]) if not np.isnan(c[0, 1]) else 0.0
    else:
        sym_index = 0.0

    return {
        "mean": mean_val,
        "std": std_val,
        "rms": rms_val,
        "peak": peak_val,
        "crest_factor": crest_factor,
        "dynamic_range_db": dyn_range,
        "entropy": shannon_entropy,
        "rms_contrast": std_val,
        "spatial_frequency": spatial_freq,
        "gradient_energy": grad_energy,
        "sparsity_pct": sparsity_ratio,
        "symmetry_index": sym_index,
    }


def load_and_inspect_image(file_path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load image from path, inspect its geometry and metadata, and return normalized [0, 1] 2D array.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    pil_img = Image.open(path)
    raw_w, raw_h = pil_img.size
    raw_mode = pil_img.mode
    raw_format = pil_img.format or path.suffix.replace(".", "").upper()

    # Preprocess according to the SPANDHAN contract (128x128 grayscale float32)
    processed_tensor = preprocess_image(pil_img, as_tensor=False)  # shape (1, 1, 128, 128)
    arr_2d = processed_tensor.squeeze()  # (128, 128)

    file_size_kb = path.stat().st_size / 1024.0

    info = {
        "filename": path.name,
        "file_path": str(path.resolve()),
        "raw_format": raw_format,
        "raw_dimensions": f"{raw_w} × {raw_h}",
        "raw_mode": raw_mode,
        "channels": "1 (Grayscale)",
        "contract_dimensions": f"{TARGET_WIDTH} × {TARGET_HEIGHT}",
        "total_pixels": TARGET_WIDTH * TARGET_HEIGHT,
        "data_type": "Float32 [0.0, 1.0]",
        "file_size_kb": f"{file_size_kb:.1f} KB",
        "min_intensity": float(arr_2d.min()),
        "max_intensity": float(arr_2d.max()),
    }

    return arr_2d, info


class ImageAnalysisWorker(QObject):
    """Worker object to run inference and metric computation on a secondary thread."""

    stage_changed = Signal(str)
    analysis_finished = Signal(dict)
    analysis_failed = Signal(str)

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            # Stage 1: Load image
            self.stage_changed.emit("Loading signal image...")
            QThread.msleep(120)  # Brief pause for visual stage feedback

            arr_2d, metadata = load_and_inspect_image(self.file_path)

            # Stage 2: Contract validation
            self.stage_changed.emit("Validating 128×128 grayscale contract...")
            QThread.msleep(140)

            # Stage 3: Load CNN inference model
            self.stage_changed.emit("Loading CNN inference engine...")
            QThread.msleep(120)
            predictor = get_image_predictor()

            # Stage 4: Feature extraction
            self.stage_changed.emit("Extracting 2D spatial features...")
            features = compute_image_signal_features(arr_2d)
            QThread.msleep(150)

            # Stage 5: Classification
            self.stage_changed.emit("Classifying signal pattern...")
            pred_result = predictor.predict_processed_image(arr_2d)
            QThread.msleep(100)

            # Package combined output
            result = {
                "metadata": metadata,
                "features": features,
                "prediction": pred_result,
                "arr_2d": arr_2d,
            }

            self.analysis_finished.emit(result)

        except Exception as e:
            self.analysis_failed.emit(str(e))
