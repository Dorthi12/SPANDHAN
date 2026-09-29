import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))

from image.image_classifier import ImageClassifier
from image.image_features import extract_image_features

def test_image_feature_extraction():
    img = np.ones((128, 128)) * 0.5
    features = extract_image_features(img)
    assert len(features) == 4
    assert features[0] == 0.5

def test_image_classifier_prediction():
    clf = ImageClassifier()
    dummy_feat = np.array([[0.5, 0.1, 1.0, 0.0]])
    pred = clf.predict(dummy_feat)
    assert isinstance(pred, str)
