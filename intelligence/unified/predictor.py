"""
intelligence/unified/predictor.py
=====================================
Inference on new uploaded audio / image files using the trained model.

Usage
-----
from intelligence.unified.predictor import UnifiedPredictor

pred = UnifiedPredictor.load()
result = pred.predict_file("path/to/audio.wav")
print(result.noise_type, result.confidence)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from intelligence.unified.feature_extractor import (
    extract_features_from_file,
    extract_audio_features,
    extract_image_features,
    DOMAIN_AUDIO,
    DOMAIN_IMAGE,
)
from intelligence.unified.model_io import (
    UnifiedModelBundle,
    load_model,
    model_exists,
    DEFAULT_MODEL_PATH,
)

# Denoising recommendation map (noise type → best method)
_AUDIO_DENOISE_MAP: dict[str, str] = {
    "gaussian":    "wavelet",
    "impulse":     "bandpass",
    "colored":     "lowpass",
    "periodic":    "bandstop",
    "mixed":       "staged",
    "clean":       "none",
}
_IMAGE_DENOISE_MAP: dict[str, str] = {
    "gaussian":        "wavelet",
    "salt_and_pepper": "median",
    "speckle":         "bilateral",
    "periodic":        "gaussian_blur",
    "uniform":         "gaussian_blur",
    "clean":           "none",
}


@dataclass
class PredictionResult:
    """
    Full prediction output for one uploaded file.

    Attributes
    ----------
    noise_type          : predicted noise class
    confidence          : probability of predicted class [0, 1]
    all_probabilities   : {class_name: probability} for all classes
    domain              : "audio" | "image"
    denoising_suggestion: recommended denoising method name
    feature_vector      : extracted feature vector (for display)
    processing_time_s   : feature extraction + inference time
    warnings            : list of non-fatal warnings
    error               : set if prediction failed
    """
    noise_type: str
    confidence: float
    all_probabilities: dict[str, float]
    domain: str
    denoising_suggestion: str
    feature_vector: np.ndarray
    processing_time_s: float
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def sorted_probs(self) -> list[tuple[str, float]]:
        """Class probabilities sorted descending."""
        return sorted(self.all_probabilities.items(), key=lambda x: -x[1])

    def generate_transcript(self, file_path: str = "") -> str:
        """Generate a clean, official diagnostic transcript report for the file."""
        if not self.success:
            return f"❌ DIAGNOSTIC TRANSCRIPT ERROR\nFile: {file_path}\nError: {self.error}"

        fname = Path(file_path).name if file_path else "Uploaded File"
        dom_title = "AUDIO SIGNAL ANALYZER" if self.domain == "audio" else "IMAGE MEDIA ANALYZER"
        
        top3 = self.sorted_probs[:3]
        top3_str = "\n".join([f"    • {cls.replace('_', ' ').title():20s} : {prob:.1%}" for cls, prob in top3])
        
        filter_name = self.denoising_suggestion.replace("_", " ").title()
        
        transcript = f"""================================================================================
                    SPANDHAN SIGNAL INTELLIGENCE TRANSCRIPT
================================================================================
Media File        : {fname}
Domain Mode       : {dom_title}
Processing Time   : {self.processing_time_s * 1000:.1f} ms
================================================================================
CLASSIFICATION DIAGNOSIS
--------------------------------------------------------------------------------
Primary Noise Type : {self.noise_type.replace("_", " ").upper()}
Confidence Score   : {self.confidence:.1%}

TOP PROBABILITY BREAKDOWN:
{top3_str}

RECOMENDED DENOISING STRATEGY:
    Filter Algorithm : {filter_name}
    Action Plan      : Apply {filter_name} filter in the {self.domain} preprocessing module to mitigate {self.noise_type} degradation.
================================================================================
"""
        return transcript


class UnifiedPredictor:
    """
    Loads the trained noise classifier and provides prediction methods.

    Parameters
    ----------
    bundle : UnifiedModelBundle — loaded from save_model()
    """

    def __init__(self, bundle: UnifiedModelBundle) -> None:
        self._bundle = bundle

    @classmethod
    def load(cls, model_path: Optional[Path] = None) -> "UnifiedPredictor":
        """
        Load the trained model from disk.

        Raises FileNotFoundError if model has not been trained yet.
        """
        bundle = load_model(model_path)
        return cls(bundle)

    @classmethod
    def is_available(cls, model_path: Optional[Path] = None) -> bool:
        """Return True if a trained model is available."""
        return model_exists(model_path)

    @property
    def class_names(self) -> list[str]:
        return self._bundle.class_names

    @property
    def val_accuracy(self) -> float:
        return self._bundle.val_accuracy

    @property
    def bundle(self) -> UnifiedModelBundle:
        return self._bundle

    # ── prediction from file path ─────────────────────────────────

    def predict_file(
        self,
        path: str,
        sampling_rate: float = 8000.0,
    ) -> PredictionResult:
        """
        Predict noise type for an uploaded file.

        Parameters
        ----------
        path          : absolute or relative path to the file
        sampling_rate : used for audio files (Hz)

        Returns
        -------
        PredictionResult
        """
        t0 = time.perf_counter()
        warnings: list[str] = []

        try:
            feat, domain = extract_features_from_file(path, sampling_rate)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            return PredictionResult(
                noise_type="unknown",
                confidence=0.0,
                all_probabilities={},
                domain="unknown",
                denoising_suggestion="none",
                feature_vector=np.zeros(1),
                processing_time_s=elapsed,
                error=str(exc),
            )

        return self._predict_features(feat, domain, t0, warnings)

    # ── prediction from in-memory arrays ─────────────────────────

    def predict_audio(
        self,
        signal: np.ndarray,
        sampling_rate: float = 8000.0,
    ) -> PredictionResult:
        """Predict noise type from an in-memory audio signal."""
        t0 = time.perf_counter()
        try:
            feat = extract_audio_features(signal, sampling_rate)
        except Exception as exc:
            return PredictionResult(
                noise_type="unknown", confidence=0.0,
                all_probabilities={}, domain="audio",
                denoising_suggestion="none",
                feature_vector=np.zeros(1),
                processing_time_s=time.perf_counter() - t0,
                error=str(exc),
            )
        return self._predict_features(feat, "audio", t0, [])

    def predict_image(self, image: np.ndarray) -> PredictionResult:
        """Predict noise type from an in-memory float64 [0,1] image."""
        t0 = time.perf_counter()
        try:
            feat = extract_image_features(image)
        except Exception as exc:
            return PredictionResult(
                noise_type="unknown", confidence=0.0,
                all_probabilities={}, domain="image",
                denoising_suggestion="none",
                feature_vector=np.zeros(1),
                processing_time_s=time.perf_counter() - t0,
                error=str(exc),
            )
        return self._predict_features(feat, "image", t0, [])

    # ── internal ──────────────────────────────────────────────────

    def _predict_features(
        self,
        feat: np.ndarray,
        domain: str,
        t0: float,
        warnings: list[str],
    ) -> PredictionResult:
        try:
            X = feat.reshape(1, -1)
            proba = self._bundle.model.predict_proba(X)[0]
            pred_idx = int(np.argmax(proba))
            noise_type = self._bundle.class_names[pred_idx]
            confidence = float(proba[pred_idx])

            all_probs = {
                cls: float(p)
                for cls, p in zip(self._bundle.class_names, proba)
            }

            # Denoising suggestion
            denoise_map = _AUDIO_DENOISE_MAP if domain == "audio" else _IMAGE_DENOISE_MAP
            suggestion = denoise_map.get(noise_type, "gaussian_blur")

            return PredictionResult(
                noise_type=noise_type,
                confidence=confidence,
                all_probabilities=all_probs,
                domain=domain,
                denoising_suggestion=suggestion,
                feature_vector=feat,
                processing_time_s=time.perf_counter() - t0,
                warnings=warnings,
            )

        except Exception as exc:
            return PredictionResult(
                noise_type="unknown",
                confidence=0.0,
                all_probabilities={},
                domain=domain,
                denoising_suggestion="none",
                feature_vector=feat,
                processing_time_s=time.perf_counter() - t0,
                error=str(exc),
            )
