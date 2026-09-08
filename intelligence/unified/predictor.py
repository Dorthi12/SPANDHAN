"""
intelligence/unified/predictor.py
=====================================
Domain-routed inference using the trained DomainRoutedBundle.

Audio files  → audio_model (37 wavelet features)
Image files  → image_model (30 texture features)

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
    DomainRoutedBundle,
    load_model,
    model_exists,
    DEFAULT_MODEL_PATH,
)
from intelligence.unified.trainer import N_AUDIO_FEATURES, N_IMAGE_FEATURES

# Denoising recommendation map (noise type → best filter method)
_AUDIO_DENOISE_MAP: dict[str, str] = {
    "gaussian": "wavelet",
    "impulse":  "bandpass",
    "colored":  "lowpass",
    "periodic": "bandstop",
    "mixed":    "staged",
    "clean":    "none",
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
            return (
                "DIAGNOSTIC TRANSCRIPT ERROR\n"
                f"File  : {file_path}\n"
                f"Error : {self.error}"
            )

        fname     = Path(file_path).name if file_path else "Uploaded File"
        dom_title = "AUDIO SIGNAL ANALYZER" if self.domain == "audio" else "IMAGE MEDIA ANALYZER"
        top3      = self.sorted_probs[:3]
        top3_str  = "\n".join(
            [f"    * {cls.replace('_', ' ').title():22s}: {prob:.1%}" for cls, prob in top3]
        )
        filter_name = self.denoising_suggestion.replace("_", " ").title()

        transcript = (
            "=" * 80 + "\n"
            "                   SPANDHAN SIGNAL INTELLIGENCE TRANSCRIPT\n"
            "=" * 80 + "\n"
            f"Media File        : {fname}\n"
            f"Domain Mode       : {dom_title}\n"
            f"Processing Time   : {self.processing_time_s * 1000:.1f} ms\n"
            "=" * 80 + "\n"
            "CLASSIFICATION DIAGNOSIS\n"
            "-" * 80 + "\n"
            f"Primary Noise Type : {self.noise_type.replace('_', ' ').upper()}\n"
            f"Confidence Score   : {self.confidence:.1%}\n"
            "\n"
            "TOP PROBABILITY BREAKDOWN:\n"
            f"{top3_str}\n"
            "\n"
            "RECOMMENDED DENOISING STRATEGY:\n"
            f"    Filter Algorithm : {filter_name}\n"
            f"    Action Plan      : Apply {filter_name} filter in the "
            f"{self.domain} preprocessing module to mitigate "
            f"{self.noise_type.replace('_', ' ')} degradation.\n"
            "=" * 80 + "\n"
        )
        return transcript


class UnifiedPredictor:
    """
    Loads the domain-routed noise classifier and dispatches inference.

    Audio files  → bundle.audio_model  (37-dimensional wavelet features)
    Image files  → bundle.image_model  (30-dimensional texture features)
    """

    def __init__(self, bundle: DomainRoutedBundle) -> None:
        self._bundle = bundle

    @classmethod
    def load(cls, model_path: Optional[Path] = None) -> "UnifiedPredictor":
        """Load the trained model from disk."""
        bundle = load_model(model_path)
        return cls(bundle)

    @classmethod
    def is_available(cls, model_path: Optional[Path] = None) -> bool:
        """Return True if a trained model is available."""
        return model_exists(model_path)

    # ── convenience properties ────────────────────────────────────

    @property
    def class_names(self) -> list[str]:
        """All unique class names across both domains."""
        seen = set()
        names = []
        for n in self._bundle.audio_class_names + self._bundle.image_class_names:
            if n not in seen:
                seen.add(n)
                names.append(n)
        return names

    @property
    def val_accuracy(self) -> float:
        return self._bundle.val_accuracy

    @property
    def bundle(self) -> DomainRoutedBundle:
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
            return PredictionResult(
                noise_type="unknown",
                confidence=0.0,
                all_probabilities={},
                domain="unknown",
                denoising_suggestion="none",
                feature_vector=np.zeros(1),
                processing_time_s=time.perf_counter() - t0,
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

    # ── internal domain router ────────────────────────────────────

    def _predict_features(
        self,
        feat: np.ndarray,
        domain: str,
        t0: float,
        warnings: list[str],
    ) -> PredictionResult:
        try:
            if domain == "audio":
                model       = self._bundle.audio_model
                class_names = self._bundle.audio_class_names
                denoise_map = _AUDIO_DENOISE_MAP
                # Slice out only the 37 audio features
                X = feat[:N_AUDIO_FEATURES].reshape(1, -1)
            elif domain == "image":
                model       = self._bundle.image_model
                class_names = self._bundle.image_class_names
                denoise_map = _IMAGE_DENOISE_MAP
                # Slice out only the 30 image features
                X = feat[:N_IMAGE_FEATURES].reshape(1, -1)
            else:
                raise ValueError(f"Unknown domain: {domain!r}")

            proba      = model.predict_proba(X)[0]
            pred_idx   = int(np.argmax(proba))
            noise_type = class_names[pred_idx]
            confidence = float(proba[pred_idx])

            all_probs = {
                cls: float(p)
                for cls, p in zip(class_names, proba)
            }

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
