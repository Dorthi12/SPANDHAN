import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))

from audio.audio_classifier import AudioClassifier
from audio.audio_features import extract_audio_features

def test_audio_feature_extraction():
    signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    features = extract_audio_features(signal, 44100)
    assert len(features) == 4

def test_audio_classifier_prediction():
    clf = AudioClassifier()
    dummy_feat = np.array([[0.1, 0.5, 100, 2500]])
    pred = clf.predict(dummy_feat)
    assert isinstance(pred, str)
