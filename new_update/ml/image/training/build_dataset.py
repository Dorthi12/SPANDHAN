"""
SPANDHAN — Image Training: Build Dataset
==========================================
Walks the 5 class image folders, loads each 128×128 grayscale PNG,
normalises pixel values to [0, 1] float32, and returns:

    X      : np.ndarray  shape (N, 1, 128, 128)  float32
    y      : np.ndarray  shape (N,)               int64
    filenames : list[str]

Dataset Contract (from MATLAB preprocessing):
    Height   = 128
    Width    = 128
    Channels = 1   (grayscale)
    Values   = [0, 1]  float32

Run once before training:
    python -m ml.image.training.build_dataset
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.common.labels import CLASS_NAMES, label_to_id

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IMAGE_DATASET_DIR: Path = _ROOT / "datasets" / "image"
IMAGE_REPORTS_DIR: Path = _ROOT / "data" / "output" / "image_predictions"

X_PATH:         Path = IMAGE_REPORTS_DIR / "X.npy"
Y_PATH:         Path = IMAGE_REPORTS_DIR / "y.npy"
FILENAMES_PATH: Path = IMAGE_REPORTS_DIR / "filenames.npy"

# Expected spatial dimensions (contract from MATLAB preprocessing)
IMG_HEIGHT: int = 128
IMG_WIDTH:  int = 128


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_single_image(path: Path) -> np.ndarray:
    """
    Load one PNG → (1, 128, 128) float32 in [0, 1].

    Handles:
    - Grayscale 'L' images (most common)
    - RGBA / RGB images (convert to grayscale)
    - Resize to 128×128 if dimensions differ
    """
    img = Image.open(path)

    # Convert to grayscale if needed
    if img.mode != "L":
        img = img.convert("L")

    # Resize if MATLAB produced a different size (safety net)
    if img.size != (IMG_WIDTH, IMG_HEIGHT):
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0   # → [0, 1]
    return arr[np.newaxis, :, :]                     # → (1, 128, 128)


def build_image_dataset(
    dataset_dir: Path | str | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Walk all 5 class folders and load every PNG.

    Returns
    -------
    X         : float32 array  (N, 1, 128, 128)
    y         : int64 array    (N,)
    filenames : list of str    length N
    """
    dataset_dir = Path(dataset_dir) if dataset_dir else IMAGE_DATASET_DIR

    X_list:    list[np.ndarray] = []
    y_list:    list[int]        = []
    filenames: list[str]        = []

    for cls in CLASS_NAMES:
        cls_dir = dataset_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {cls_dir}\n"
                f"Expected structure: {dataset_dir}/<class>/<images>.png"
            )

        pngs = sorted(cls_dir.glob("*.png"))
        if not pngs:
            print(f"  [WARN] No PNGs found in {cls_dir}")
            continue

        cid = label_to_id(cls)
        ok = err = 0

        if verbose:
            print(f"  [{cls:11s}] {len(pngs):4d} images …", flush=True)

        for png in pngs:
            try:
                arr = load_single_image(png)
                X_list.append(arr)
                y_list.append(cid)
                filenames.append(str(png))
                ok += 1
            except Exception as e:                   # noqa: BLE001
                err += 1
                if verbose:
                    print(f"    [ERR] {png.name}: {e}")

        if verbose:
            print(f"    >> {ok} ok, {err} errors")

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, 1, 128, 128)
    y = np.array(y_list, dtype=np.int64)              # (N,)

    if verbose:
        print(f"\nDataset loaded: X={X.shape}  y={y.shape}")
        print(f"Value range: [{X.min():.3f}, {X.max():.3f}]")

    return X, y, filenames


def save_dataset(X: np.ndarray, y: np.ndarray, filenames: list[str]) -> None:
    """Persist the dataset arrays for subsequent training runs."""
    IMAGE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(X_PATH), X)
    np.save(str(Y_PATH), y)
    np.save(str(FILENAMES_PATH), np.array(filenames, dtype=object))
    print(f"\nSaved  X={X.shape}  y={y.shape}  to {IMAGE_REPORTS_DIR}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("SPANDHAN — Build Image Dataset")
    print("=" * 60)

    X, y, fns = build_image_dataset()

    from ml.common.labels import ID_TO_CLASS
    print(f"\nTotal: {len(y)} samples")
    for cid, cname in ID_TO_CLASS.items():
        count = int(np.sum(y == cid))
        print(f"  {cname:12s}: {count}")

    save_dataset(X, y, fns)
    print(f"\nFinished in {time.time() - t0:.1f}s")
