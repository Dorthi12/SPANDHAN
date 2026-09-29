"""
SPANDHAN - Audio Signal Predictor (Inference)
=============================================
Loads the trained audio_signal_classifier.pkl ONCE and classifies any WAV.
No retraining. No scaling recalculation. Audio-only model.

Usage (programmatic)
--------------------
    from ml.audio.inference.predict import AudioPredictor
    p = AudioPredictor()
    r = p.predict_wav("path/to/signal.wav")
    # r = {
    #   "class": "chirp", "class_id": 4, "confidence": 0.964,
    #   "probabilities": {"impulse": 0.01, ..., "chirp": 0.96}
    # }

Usage (CLI)
-----------
    python -m ml.audio.inference.predict --wav signal.wav
"""

from __future__ import annotations
import sys, argparse
from pathlib import Path

import numpy as np
import joblib
from scipy.io import wavfile

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.audio.features import extract_features

CLASS_NAMES = ["impulse", "sinusoidal", "white_noise", "step", "chirp"]
PKL_PATH    = _ROOT / "models" / "audio" / "audio_signal_classifier.pkl"


class AudioPredictor:
    """Load-once, predict-many inference wrapper for the audio classifier."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        path = Path(model_path) if model_path else PKL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                "Run  python -m ml.audio.training.train_final  first."
            )
        self._pipeline = joblib.load(str(path))

    @staticmethod
    def _load_wav(wav_path: str | Path):
        sr, data = wavfile.read(str(wav_path))
        if data.dtype.kind == "i":
            data = data.astype(np.float64) / np.iinfo(data.dtype).max
        else:
            data = data.astype(np.float64)
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data, int(sr)

    def _infer(self, feat: np.ndarray) -> dict:
        x     = feat.reshape(1, -1)
        cid   = int(self._pipeline.predict(x)[0])
        proba = self._pipeline.predict_proba(x)[0]
        return {
            "class":         CLASS_NAMES[cid],
            "class_id":      cid,
            "confidence":    round(float(proba[cid]), 6),
            "probabilities": {n: round(float(p), 6) for n, p in zip(CLASS_NAMES, proba)},
        }

    def predict_wav(self, wav_path: str | Path) -> dict:
        """Classify a .wav file."""
        audio, sr = self._load_wav(wav_path)
        return self._infer(extract_features(audio, sr))

    def predict_signal(self, signal: np.ndarray, sr: int) -> dict:
        """Classify a raw numpy audio array."""
        return self._infer(extract_features(signal, sr))

    def predict_features(self, features: np.ndarray) -> dict:
        """Classify from a pre-computed 35-element feature vector."""
        return self._infer(features)


def _cli():
    parser = argparse.ArgumentParser(description="SPANDHAN Audio Classifier (inference)")
    parser.add_argument("--wav",   required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    p = AudioPredictor(model_path=args.model)
    r = p.predict_wav(args.wav)

    print("\nSPANDHAN Audio Classification")
    print("-" * 38)
    print(f"  File       : {args.wav}")
    print(f"  Class      : {r['class']}")
    print(f"  Class ID   : {r['class_id']}")
    print(f"  Confidence : {r['confidence']:.4f}")
    print("\n  Probabilities:")
    for cls, prob in r["probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"    {cls:12s}: {prob:.4f}  {bar}")


if __name__ == "__main__":
    _cli()
