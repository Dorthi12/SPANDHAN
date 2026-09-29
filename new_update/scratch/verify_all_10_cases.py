import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ml.audio.inference.predict import AudioPredictor
from ml.image.inference.predict import ImagePredictor

classes = ["impulse", "sinusoidal", "white_noise", "step", "chirp"]

audio_expected_dsp = {
    "impulse": ["FFT", "Wavelet", "FIR"],
    "sinusoidal": ["FFT", "FIR", "IIR"],
    "white_noise": ["FFT", "FIR", "IIR"],
    "step": ["FFT", "Wavelet", "FIR", "IIR"],
    "chirp": ["FFT", "STFT", "Wavelet"]
}

image_expected_dsp = {
    "impulse": ["2-D FFT", "2-D Wavelet", "Image Filter"],
    "sinusoidal": ["2-D FFT", "Image Filter"],
    "white_noise": ["2-D FFT", "Image Filter"],
    "step": ["2-D FFT", "2-D Wavelet", "Image Filter"],
    "chirp": ["2-D FFT", "2-D Wavelet", "Image Filter"]
}

def verify():
    print("=" * 80)
    print("         SPANDHAN 10-CASE VALIDATION MATRIX (5 AUDIO + 5 IMAGE)")
    print("=" * 80)
    print(f"{'Modality':<8} | {'Input Class':<12} | {'Predicted':<12} | {'Confidence':<10} | {'Status':<6} | {'Target DSP Modules'}")
    print("-" * 80)

    # 1. Audio
    audio_model_path = project_root / "models" / "audio" / "audio_signal_classifier.pkl"
    audio_pred = AudioPredictor(audio_model_path)
    
    for c in classes:
        f = project_root / "datasets" / "audio" / c / f"{c}_0001.wav"
        if not f.exists():
            print(f"AUDIO    | {c:<12} | MISSING      | -          | FAIL   | File not found")
            continue
        res = audio_pred.predict_wav(str(f))
        pred_class = res['class']
        conf = res['confidence'] * 100
        status = "PASS" if pred_class.lower() == c.lower() else "WARN"
        dsp = ", ".join(audio_expected_dsp.get(pred_class.lower(), []))
        print(f"AUDIO    | {c:<12} | {pred_class:<12} | {conf:8.2f}% | {status:<6} | {dsp}")

    print("-" * 80)

    # 2. Image
    image_model_path = project_root / "models" / "image" / "image_signal_classifier.pkl"
    image_pred = ImagePredictor(image_model_path)

    for c in classes:
        f = project_root / "datasets" / "image" / c / f"{c}_0001.png"
        if not f.exists():
            print(f"IMAGE    | {c:<12} | MISSING      | -          | FAIL   | File not found")
            continue
        res = image_pred.predict_image(str(f))
        pred_class = res['class']
        conf = res['confidence'] * 100
        status = "PASS" if pred_class.lower() == c.lower() else "WARN"
        dsp = ", ".join(image_expected_dsp.get(pred_class.lower(), []))
        print(f"IMAGE    | {c:<12} | {pred_class:<12} | {conf:8.2f}% | {status:<6} | {dsp}")

    print("=" * 80)

if __name__ == '__main__':
    verify()
