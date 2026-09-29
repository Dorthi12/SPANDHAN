import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from ml.audio.inference.predict import AudioPredictor
from ml.image.inference.predict import ImagePredictor

def test_sinusoidal():
    print("="*60)
    print("TESTING ML PREDICTIONS FOR SINUSOIDAL 0003")
    print("="*60)

    # 1. AUDIO TEST
    audio_model_path = os.path.join(project_root, "models", "audio", "audio_signal_classifier.pkl")
    audio_file = os.path.join(project_root, "datasets", "audio", "sinusoidal", "sinusoidal_0003.wav")
    
    print(f"\n1. Audio File: {audio_file}")
    audio_pred = AudioPredictor(audio_model_path)
    res_audio = audio_pred.predict_wav(audio_file)
    print(f"   Predicted Class : {res_audio['class'].upper()}")
    print(f"   Class ID        : {res_audio['class_id']}")
    print(f"   Confidence      : {res_audio['confidence']*100:.2f}%")
    print("   Probabilities   :")
    for k, v in res_audio['probabilities'].items():
        print(f"     - {k:12s}: {v:.4f}")

    # 2. IMAGE TEST
    image_model_path = os.path.join(project_root, "models", "image", "image_signal_classifier.pkl")
    image_file = os.path.join(project_root, "datasets", "image", "sinusoidal", "sinusoidal_0003.png")

    print(f"\n2. Image File: {image_file}")
    image_pred = ImagePredictor(image_model_path)
    res_image = image_pred.predict_image(image_file)
    print(f"   Predicted Class : {res_image['class'].upper()}")
    print(f"   Class ID        : {res_image['class_id']}")
    print(f"   Confidence      : {res_image['confidence']*100:.2f}%")
    print("   Probabilities   :")
    for k, v in res_image['probabilities'].items():
        print(f"     - {k:12s}: {v:.4f}")

    print("\n" + "="*60)

if __name__ == '__main__':
    test_sinusoidal()
