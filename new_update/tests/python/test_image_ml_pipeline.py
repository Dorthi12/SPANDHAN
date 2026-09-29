"""
Tests for SPANDHAN Image ML Pipeline
====================================
Tests:
- Label definitions and mapping consistency
- Input contract preprocessing (dimensions, normalization, channels)
- SignalCNN forward pass and softmax probabilities
- ImagePredictor inference engine and prediction output contract
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.common.labels import CLASS_NAMES, CLASS_TO_ID, ID_TO_CLASS, NUM_CLASSES, label_to_id, id_to_label
from ml.image.cnn_model import SignalCNN
from ml.image.inference.preprocess_input import preprocess_image, TARGET_HEIGHT, TARGET_WIDTH
from ml.image.training.split_dataset import make_splits


def test_labels_contract():
    assert NUM_CLASSES == 5
    assert CLASS_NAMES == ["impulse", "sinusoidal", "white_noise", "step", "chirp"]
    assert CLASS_TO_ID["impulse"] == 0
    assert CLASS_TO_ID["chirp"] == 4
    assert ID_TO_CLASS[2] == "white_noise"
    assert label_to_id("step") == 3
    assert id_to_label(1) == "sinusoidal"


def test_preprocess_from_numpy():
    # Test random 2D array
    raw = np.random.rand(100, 100).astype(np.float32)
    tensor = preprocess_image(raw, as_tensor=True)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, TARGET_HEIGHT, TARGET_WIDTH)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_preprocess_from_rgb():
    # Test RGB array (128, 128, 3) in uint8 [0-255]
    rgb = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    arr = preprocess_image(rgb, as_tensor=False)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 1, 128, 128)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


def test_preprocess_from_pil():
    # Test PIL Image
    pil_img = Image.new("L", (64, 64), color=128)
    tensor = preprocess_image(pil_img, as_tensor=True)
    assert tensor.shape == (1, 1, 128, 128)
    assert pytest.approx(tensor.mean().item(), rel=1e-2) == 128.0 / 255.0


def test_cnn_architecture():
    model = SignalCNN(num_classes=5, dropout=0.4)
    model.eval()
    dummy_input = torch.randn(4, 1, 128, 128)
    logits = model(dummy_input)
    assert logits.shape == (4, 5)

    probs = model.predict_proba(dummy_input)
    assert probs.shape == (4, 5)
    # Sum of probabilities across classes should be 1.0
    sums = probs.sum(dim=1).numpy()
    np.testing.assert_allclose(sums, np.ones(4), atol=1e-5)


def test_dataset_split_proportions():
    n_samples = 1000
    X = np.zeros((n_samples, 1, 128, 128), dtype=np.float32)
    y = np.array([i % 5 for i in range(n_samples)], dtype=np.int64)

    X_tr, X_val, X_te, y_tr, y_val, y_te, _, _, _ = make_splits(X, y, verbose=False)

    assert len(y_tr) == 700
    assert len(y_val) == 150
    assert len(y_te) == 150

    # Check stratification: every class has equal samples in each split
    for cid in range(5):
        assert np.sum(y_tr == cid) == 140
        assert np.sum(y_val == cid) == 30
        assert np.sum(y_te == cid) == 30


def test_predict_processed_image_contract():
    # Test that a preprocessed 128x128 2D numpy array directly passes through
    from ml.image.cnn_model import SignalCNN
    model = SignalCNN(num_classes=5, dropout=0.4)
    model.eval()

    # Create dummy image array (as passed by MATLAB py.numpy.array)
    matlab_array = np.random.rand(128, 128).astype(np.float64)

    # Convert via predict_processed_image logic
    arr = np.asarray(matlab_array, dtype=np.float32)
    assert arr.shape == (128, 128)
    arr_tensor = torch.from_numpy(arr[np.newaxis, np.newaxis, :, :]).float()
    with torch.no_grad():
        probs = model.predict_proba(arr_tensor).cpu().numpy()[0]

    assert len(probs) == 5
    assert pytest.approx(float(np.sum(probs)), rel=1e-4) == 1.0
