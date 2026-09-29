import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.common.labels import CLASS_NAMES, label_to_id, id_to_label
from ml.audio.features import extract_features, NUM_FEATURES
from ml.audio.inference.predict import AudioPredictor

def test_labels():
    assert len(CLASS_NAMES) == 5
    assert label_to_id("chirp") == 4
    assert id_to_label(0) == "impulse"

def test_audio_feature_extraction():
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2)
    sig = np.sin(2 * np.pi * 440 * t)
    feats = extract_features(sig, sr)
    assert len(feats) == NUM_FEATURES
    assert np.all(np.isfinite(feats))

def test_audio_predictor_inference():
    predictor = AudioPredictor()
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2)
    sig = np.sin(2 * np.pi * 440 * t)
    result = predictor.predict_signal(sig, sr)

    assert "class" in result
    assert "class_id" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert result["class"] == "sinusoidal"
    assert result["confidence"] > 0.90
    assert len(result["probabilities"]) == 5
