"""
tests/test_unified_model.py
============================
Unit tests for Spandhan unified noise classification intelligence module:
- Feature extraction (audio & image)
- Dataset generation
- Random Forest model training & evaluation
- Model saving / loading (Model IO)
- Prediction / inference pipeline
"""

from pathlib import Path
import numpy as np
import pytest

from intelligence.unified.feature_extractor import (
    extract_audio_features,
    extract_image_features,
    extract_features_from_file,
    UNIFIED_FEATURE_DIM,
    DOMAIN_AUDIO,
    DOMAIN_IMAGE,
)
from intelligence.unified.dataset_builder import (
    build_unified_dataset,
    load_unified_dataset,
    UnifiedDatasetResult,
)
from intelligence.unified.trainer import train_unified_model
from intelligence.unified.model_io import (
    DomainRoutedBundle,
    save_model,
    load_model,
    model_exists,
)
from intelligence.unified.predictor import UnifiedPredictor, PredictionResult


class TestFeatureExtractor:
    def test_audio_feature_shape_and_domain(self):
        sr = 8000
        t = np.linspace(0, 1, sr)
        audio_sig = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.default_rng(42).normal(size=len(t))
        feat = extract_audio_features(audio_sig, sampling_rate=sr)
        
        assert feat.shape == (UNIFIED_FEATURE_DIM,)
        assert feat[-1] == DOMAIN_AUDIO
        assert not np.isnan(feat).any()

    def test_image_feature_shape_and_domain(self):
        img = np.random.default_rng(42).uniform(0, 255, (64, 64)).astype(np.float64)
        feat = extract_image_features(img)
        
        assert feat.shape == (UNIFIED_FEATURE_DIM,)
        assert feat[-1] == DOMAIN_IMAGE
        assert not np.isnan(feat).any()

    def test_extract_from_file_audio(self, tmp_path):
        import scipy.io.wavfile as wav
        sr = 8000
        t = np.linspace(0, 0.5, int(sr * 0.5))
        sig = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        wav_path = tmp_path / "test_audio.wav"
        wav.write(str(wav_path), sr, sig)

        feat, domain = extract_features_from_file(wav_path)
        assert feat.shape == (UNIFIED_FEATURE_DIM,)
        assert feat[-1] == DOMAIN_AUDIO
        assert domain == "audio"


class TestDatasetBuilder:
    def test_small_dataset_generation(self, tmp_path):
        ds_dir = tmp_path / "unified_ds"
        res = build_unified_dataset(
            n_audio=12,
            n_image=12,
            seed=42,
            output_dir=ds_dir,
            verbose=False,
        )
        
        assert isinstance(res, UnifiedDatasetResult)
        assert res.n_total == 24
        assert res.features.shape == (24, UNIFIED_FEATURE_DIM)
        assert len(res.labels) == 24
        assert len(res.domains) == 24

        # Verify disk save & load
        assert (ds_dir / "features.npy").exists()
        loaded_res = load_unified_dataset(ds_dir)
        assert isinstance(loaded_res, UnifiedDatasetResult)
        assert loaded_res.features.shape == (24, UNIFIED_FEATURE_DIM)
        assert len(loaded_res.labels) == 24


class TestTrainerAndModelIO:
    def test_train_save_load_predict(self, tmp_path):
        # 1. Create mini dataset
        ds_dir = tmp_path / "unified_ds"
        ds_res = build_unified_dataset(
            n_audio=24,
            n_image=24,
            seed=42,
            output_dir=ds_dir,
            verbose=False,
        )

        # 2. Train model
        model_path = tmp_path / "unified_model.pkl"
        bundle = train_unified_model(
            dataset_result=ds_res,
            model_path=model_path,
            verbose=False,
        )

        assert isinstance(bundle, DomainRoutedBundle)
        assert model_exists(model_path)

        # 3. Model IO
        loaded_bundle = load_model(model_path)
        assert loaded_bundle.audio_class_names == bundle.audio_class_names

        # 4. Inference / Predictor
        predictor = UnifiedPredictor(loaded_bundle)
        audio_test = np.random.default_rng(123).normal(size=8000)
        res = predictor.predict_audio(audio_test)

        assert isinstance(res, PredictionResult)
        assert res.success
        assert res.confidence > 0.0
        assert res.domain == "audio"
        assert res.noise_type in bundle.audio_class_names
