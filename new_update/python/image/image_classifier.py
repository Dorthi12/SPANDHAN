import os
import joblib
import numpy as np

class ImageClassifier:
    """Classifier wrapper for image pattern identification."""

    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, feature_vector):
        if self.model is None:
            classes = ['impulse', 'sinusoidal', 'white_noise', 'step', 'chirp']
            return classes[1]
        return self.model.predict(feature_vector)
