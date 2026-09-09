"""
converters/image_to_mat/converter.py
====================================
Core conversion logic to transform image files (.png, .jpg, .jpeg, .bmp, .tiff, etc.)
into 1D digital signal representations and export them as MATLAB .mat files
fully compatible with Spandhan's signal loader and DSP workflows.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
from PIL import Image
from scipy.io import savemat

# Supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)

# Supported conversion strategies for mapping 2-D images to 1-D signals
CONVERSION_MODES = {
    "raster": "Full Raster Scan (Continuous pixel sequence, shape H*W)",
    "row_mean": "Row Mean Profile (Horizontal average per row, shape H)",
    "col_mean": "Column Mean Profile (Vertical average per col, shape W)",
    "center_row": "Center Horizontal Line Scan (Middle row cross-section, shape W)",
    "center_col": "Center Vertical Line Scan (Middle col cross-section, shape H)",
}


def load_image_as_array(
    image_path: Union[str, Path],
    grayscale: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Load an image file and convert it into a 2-D float64 NumPy array.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file.
    grayscale : bool
        If True, convert image to single-channel luminance (grayscale).
    normalize : bool
        If True, scale pixel intensities to [0.0, 1.0].

    Returns
    -------
    np.ndarray
        2-D array of shape (H, W), dtype float64.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS and ext != ".npy":
        raise ValueError(
            f"Unsupported image format '{ext}'. Supported: {SUPPORTED_IMAGE_EXTENSIONS}"
        )

    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:
            # Multi-channel image array -> grayscale average
            arr = np.mean(arr, axis=2)
        elif arr.ndim != 2:
            raise ValueError(f"Expected 2-D array in .npy file, got shape {arr.shape}")
        arr = arr.astype(np.float64)
        if normalize and arr.max() > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0) if normalize else arr

    # Load standard image with PIL
    try:
        with Image.open(path) as img:
            if grayscale:
                img = img.convert("L")
            arr = np.array(img, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Failed to decode image '{path.name}': {exc}") from exc

    if normalize:
        arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)

    return arr


def convert_image_to_signal(
    image_array: np.ndarray,
    mode: str = "raster",
) -> np.ndarray:
    """
    Transform a 2-D image array into a 1-D digital signal vector.

    Parameters
    ----------
    image_array : np.ndarray
        2-D array of shape (H, W).
    mode : str
        One of CONVERSION_MODES keys:
        - "raster": Flatten all rows into a continuous 1-D stream (H*W)
        - "row_mean": Average intensity across columns for each row (H)
        - "col_mean": Average intensity across rows for each column (W)
        - "center_row": Extract the center horizontal scan line (W)
        - "center_col": Extract the center vertical scan line (H)

    Returns
    -------
    np.ndarray
        1-D float64 signal array.
    """
    if image_array.ndim != 2:
        raise ValueError(f"image_array must be 2-dimensional (H, W), got {image_array.shape}")

    H, W = image_array.shape
    if H == 0 or W == 0:
        raise ValueError("Image array cannot be empty.")

    mode = mode.lower()
    if mode == "raster":
        signal = image_array.ravel()
    elif mode == "row_mean":
        signal = np.mean(image_array, axis=1)
    elif mode == "col_mean":
        signal = np.mean(image_array, axis=0)
    elif mode == "center_row":
        center_y = H // 2
        signal = image_array[center_y, :]
    elif mode == "center_col":
        center_x = W // 2
        signal = image_array[:, center_x]
    else:
        raise ValueError(
            f"Unknown conversion mode '{mode}'. Available modes: {list(CONVERSION_MODES.keys())}"
        )

    # Ensure 1-D float64 with finite values
    signal = np.asarray(signal, dtype=np.float64).squeeze()
    if not np.all(np.isfinite(signal)):
        signal = np.nan_to_num(signal, nan=0.0, posinf=1.0, neginf=0.0)

    return signal


def save_signal_to_mat(
    signal: np.ndarray,
    output_mat_path: Union[str, Path],
    sampling_rate: float = 1000.0,
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """
    Save a 1-D signal array to a MATLAB .mat file formatted for Spandhan.

    Parameters
    ----------
    signal : np.ndarray
        1-D numerical signal array.
    output_mat_path : str or Path
        Destination .mat file path.
    sampling_rate : float
        Sampling rate in Hz to store in metadata.
    metadata : dict, optional
        Additional metadata key-values to store inside the MAT dictionary.

    Returns
    -------
    Path
        Absolute path to the created .mat file.
    """
    out_path = Path(output_mat_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sig = np.asarray(signal, dtype=np.float64).squeeze()
    if sig.ndim != 1:
        raise ValueError(f"Signal must be 1-dimensional, got shape {sig.shape}")

    mat_dict: dict[str, Any] = {
        "signal": sig,
        "sampling_rate": float(sampling_rate),
    }

    if metadata:
        for k, v in metadata.items():
            if not k.startswith("__"):
                mat_dict[k] = v

    savemat(out_path, mat_dict)
    return out_path


def convert_image_to_mat(
    image_path: Union[str, Path],
    output_mat_path: Optional[Union[str, Path]] = None,
    mode: str = "raster",
    sampling_rate: float = 1000.0,
    normalize: bool = True,
) -> tuple[Path, np.ndarray, dict[str, Any]]:
    """
    High-level convenience function: Load image -> Convert to 1D signal -> Save to .mat.

    Parameters
    ----------
    image_path : str or Path
        Path to input image file.
    output_mat_path : str or Path, optional
        Destination .mat file path. If None, defaults to same filename with .mat in output directory.
    mode : str
        Conversion mode ("raster", "row_mean", "col_mean", "center_row", "center_col").
    sampling_rate : float
        Sampling rate in Hz (default: 1000.0).
    normalize : bool
        Whether to normalize pixel values in [0.0, 1.0].

    Returns
    -------
    (out_mat_path, signal_1d, info_dict)
    """
    img_path = Path(image_path)
    img_arr = load_image_as_array(img_path, grayscale=True, normalize=normalize)
    signal_1d = convert_image_to_signal(img_arr, mode=mode)

    if output_mat_path is None:
        out_dir = img_path.parent / "output_mat"
        out_path = out_dir / f"{img_path.stem}_{mode}.mat"
    else:
        out_path = Path(output_mat_path)

    meta = {
        "source_image": img_path.name,
        "conversion_mode": mode,
        "original_shape": np.array(img_arr.shape, dtype=np.int32),
        "num_samples": int(len(signal_1d)),
    }

    saved_path = save_signal_to_mat(
        signal=signal_1d,
        output_mat_path=out_path,
        sampling_rate=sampling_rate,
        metadata=meta,
    )

    info = {
        "input_image": str(img_path),
        "output_mat": str(saved_path),
        "image_shape": img_arr.shape,
        "signal_length": len(signal_1d),
        "mode": mode,
        "sampling_rate": sampling_rate,
    }

    return saved_path, signal_1d, info
