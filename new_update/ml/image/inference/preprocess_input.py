"""
SPANDHAN — Image Inference: Input Preprocessing Contract
==========================================================
Enforces the MATLAB <-> Python Image ML contract:
    - Height:    128
    - Width:     128
    - Channels:  1 (grayscale)
    - Intensity: [0.0, 1.0] float32
    - Output:    (1, 1, 128, 128) torch.Tensor or (1, 128, 128) np.ndarray

Used by inference pipelines before calling model.predict().
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

TARGET_HEIGHT: int = 128
TARGET_WIDTH: int = 128


def preprocess_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    as_tensor: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Preprocess any valid image input into the standard SPANDHAN ML contract.

    Parameters
    ----------
    image_input : str, Path, PIL.Image, or np.ndarray
        Raw image from MATLAB or filesystem.
    as_tensor : bool
        If True, returns torch.FloatTensor of shape (1, 1, 128, 128).
        If False, returns np.ndarray of shape (1, 1, 128, 128) float32.

    Returns
    -------
    Preprocessed image ready for CNN inference.
    """
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        img = Image.open(path)
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        arr = image_input.copy()
        # Squeeze unnecessary dimensions
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)  # (1, H, W) -> (H, W)
        elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            if arr.shape[2] == 1:
                arr = arr.squeeze(2)
            elif arr.shape[2] in (3, 4):
                # RGB / RGBA -> grayscale via standard luminance weights
                arr = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D image array, got shape {image_input.shape}")

        # Check normalization
        if arr.dtype == np.uint8 or (arr.max() > 1.0 and arr.max() <= 255.0):
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
            # Clip between [0, 1] for safety
            arr = np.clip(arr, 0.0, 1.0)

        # Resize if dimensions differ from 128x128
        if arr.shape != (TARGET_HEIGHT, TARGET_WIDTH):
            pil_img = Image.fromarray((arr * 255.0).astype(np.uint8))
            pil_img = pil_img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
            arr = np.array(pil_img, dtype=np.float32) / 255.0

        # Reshape to (1, 1, 128, 128)
        arr_final = arr[np.newaxis, np.newaxis, :, :]
        if as_tensor:
            return torch.from_numpy(arr_final).float()
        return arr_final

    else:
        raise TypeError(f"Unsupported input type for image_input: {type(image_input)}")

    # PIL Image path:
    if img.mode != "L":
        img = img.convert("L")

    if img.size != (TARGET_WIDTH, TARGET_HEIGHT):
        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr_final = arr[np.newaxis, np.newaxis, :, :]

    if as_tensor:
        return torch.from_numpy(arr_final).float()
    return arr_final
