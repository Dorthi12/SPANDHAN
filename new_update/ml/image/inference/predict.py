"""
SPANDHAN — Image Signal Predictor (Inference)
==============================================
Loads the trained image_signal_classifier.pkl ONCE and classifies any 128×128 image.
No retraining. Grayscale 128×128×1 signal contract.

Usage (programmatic):
    from ml.image.inference.predict import ImagePredictor
    p = ImagePredictor()
    result = p.predict_image("path/to/signal.png")
    # Returns:
    # {
    #     "class": "chirp",
    #     "class_id": 4,
    #     "confidence": 0.973,
    #     "probabilities": {
    #         "impulse": 0.004,
    #         "sinusoidal": 0.015,
    #         "white_noise": 0.001,
    #         "step": 0.007,
    #         "chirp": 0.973
    #     }
    # }

Usage (CLI):
    python -m ml.image.inference.predict --image datasets/image/chirp/chirp_0001.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.common.config import IMAGE_PIPELINE_PATH
from ml.common.labels import CLASS_NAMES, NUM_CLASSES
from ml.image.cnn_model import SignalCNN
from ml.image.inference.preprocess_input import preprocess_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_IMAGE_PREDICTOR_INSTANCE: ImagePredictor | None = None


def get_image_predictor(model_path: Union[str, Path, None] = None) -> ImagePredictor:
    """
    Return the singleton ImagePredictor instance.
    Loads the trained model once into memory and reuses it across MATLAB calls.
    """
    global _IMAGE_PREDICTOR_INSTANCE
    if _IMAGE_PREDICTOR_INSTANCE is None or model_path is not None:
        _IMAGE_PREDICTOR_INSTANCE = ImagePredictor(model_path=model_path)
    return _IMAGE_PREDICTOR_INSTANCE


class ImagePredictor:
    """Load-once, predict-many inference engine for the SPANDHAN Image Classifier."""

    def __init__(self, model_path: Union[str, Path, None] = None) -> None:
        path = Path(model_path) if model_path else IMAGE_PIPELINE_PATH
        if not path.exists():
            # Fallback check in parent models folder
            alt_path = _ROOT / "models" / "image_signal_classifier.pkl"
            if alt_path.exists():
                path = alt_path
            else:
                raise FileNotFoundError(
                    f"Model not found at {path}.\n"
                    "Train the model first with: python -m ml.image.training.train"
                )

        payload = joblib.load(str(path))
        self.model = SignalCNN(
            num_classes=payload.get("num_classes", NUM_CLASSES),
            dropout=payload.get("dropout", 0.4),
        )
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()

        self.class_names = payload.get("class_names", CLASS_NAMES)
        self.metadata = {
            "best_val_acc": payload.get("best_val_acc"),
            "test_accuracy": payload.get("test_accuracy"),
            "model_path": str(path),
        }

    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> dict:
        """
        Predict signal pattern for a single image input.

        Parameters
        ----------
        image_input : str, Path, PIL.Image, or np.ndarray
            The image to classify.

        Returns
        -------
        dict with keys: 'class', 'class_id', 'confidence', 'probabilities'
        """
        tensor = preprocess_image(image_input, as_tensor=True).to(DEVICE)
        with torch.no_grad():
            probs = self.model.predict_proba(tensor).cpu().numpy()[0]

        pred_id = int(probs.argmax())
        confidence = float(probs[pred_id])
        pred_class = self.class_names[pred_id]

        return {
            "class": pred_class,
            "class_id": pred_id,
            "confidence": round(confidence, 6),
            "probabilities": {
                name: round(float(p), 6)
                for name, p in zip(self.class_names, probs)
            },
        }

    def predict_processed_image(
        self,
        image_array: Union[np.ndarray, torch.Tensor],
    ) -> dict:
        """
        Classify a 128x128 grayscale signal image directly preprocessed by MATLAB.
        Bypasses disk I/O and PIL decoding for maximum throughput and zero re-scaling.

        Parameters
        ----------
        image_array : np.ndarray or torch.Tensor
            2D (128x128) or 3D/4D float32 array in [0, 1].

        Returns
        -------
        dict with standardized structure:
            class, class_id, confidence, probabilities
        """
        if isinstance(image_array, torch.Tensor):
            arr = image_array.cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(image_array, dtype=np.float32)

        # Handle any dimension quirks from MATLAB py.numpy.array
        if arr.ndim == 2:
            arr = arr[np.newaxis, np.newaxis, :, :]
        elif arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[np.newaxis, :, :, :]
            elif arr.shape[2] == 1:
                arr = arr.transpose(2, 0, 1)[np.newaxis, :, :, :]
            else:
                arr = arr[np.newaxis, np.newaxis, :, :]
        elif arr.ndim == 4:
            if arr.shape[1] != 1 and arr.shape[3] == 1:
                arr = arr.transpose(0, 3, 1, 2)

        tensor = torch.from_numpy(arr).float().to(DEVICE)
        with torch.no_grad():
            probs = self.model.predict_proba(tensor).cpu().numpy()[0]

        pred_id = int(probs.argmax())
        confidence = float(probs[pred_id])
        pred_class = self.class_names[pred_id]

        return {
            "class": pred_class,
            "class_id": pred_id,
            "confidence": round(confidence, 6),
            "probabilities": {
                name: round(float(p), 6)
                for name, p in zip(self.class_names, probs)
            },
        }

    def predict_batch(
        self,
        images: list[Union[str, Path, Image.Image, np.ndarray]],
    ) -> list[dict]:
        """Classify a list of images in a single batched pass."""
        tensors = [preprocess_image(img, as_tensor=True) for img in images]
        batch = torch.cat(tensors, dim=0).to(DEVICE)

        with torch.no_grad():
            probs_batch = self.model.predict_proba(batch).cpu().numpy()

        results = []
        for probs in probs_batch:
            pred_id = int(probs.argmax())
            results.append({
                "class": self.class_names[pred_id],
                "class_id": pred_id,
                "confidence": round(float(probs[pred_id]), 6),
                "probabilities": {
                    name: round(float(p), 6)
                    for name, p in zip(self.class_names, probs)
                },
            })
        return results


def _cli():
    parser = argparse.ArgumentParser(description="SPANDHAN Image Signal Classifier (inference)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default=None, help="Path to model .pkl file")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    predictor = ImagePredictor(model_path=args.model)
    res = predictor.predict_image(args.image)

    if args.json:
        print(json.dumps(res, indent=2))
        return

    print("\nSPANDHAN Image Signal Classification")
    print("=" * 45)
    print(f"  Input File  : {args.image}")
    print(f"  Class       : {res['class']}")
    print(f"  Class ID    : {res['class_id']}")
    print(f"  Confidence  : {res['confidence']:.4f}")
    print("\n  Class Probabilities:")
    for cls_name, prob in res["probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"    {cls_name:12s}: {prob:.4f}  {bar}")


if __name__ == "__main__":
    _cli()
