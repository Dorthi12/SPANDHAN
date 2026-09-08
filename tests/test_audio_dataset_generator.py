"""
tests/test_audio_dataset_generator.py
=======================================
Tests for generators/audio_dataset_generator.py — Milestone A.

Coverage
--------
1. Deterministic generation (same seed → same signals)
2. Different seeds → different signals
3. Correct sample dimensions (num_samples = round(sr * duration))
4. Correct sampling rate in metadata
5. All supported signal types are generated
6. All generated values are finite (no NaN / Inf)
7. Metadata is separate from signal data
8. Metadata contains required fields
9. Dataset can be saved and loaded (round-trip)
10. Loaded signals are numerically identical to generated ones
11. All six signal types produce valid signals
12. Invalid config raises controlled errors
13. Empty signal_types list raises error
14. Unsupported signal type raises error
15. Negative sampling rate raises error
16. Zero duration raises error
17. Non-integer seed raises error
18. Zero samples_per_type raises error
19. Individual low-level generators (chirp, AM, FM, harmonic)
20. Existing waveform_generator delegation (sine, multi_tone)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from generators.audio_dataset_generator import (
    AUDIO_SIGNAL_TYPES,
    AudioDatasetConfig,
    AudioSample,
    AudioSampleMetadata,
    SyntheticAudioDataset,
    _generate_amplitude_modulated,
    _generate_chirp,
    _generate_frequency_modulated,
    _generate_harmonic,
    build_audio_dataset,
)


# ==================================================================
# Fixtures
# ==================================================================


@pytest.fixture
def default_config() -> AudioDatasetConfig:
    return AudioDatasetConfig(
        samples_per_type=2,
        sampling_rate=8000.0,   # low SR for speed in tests
        duration=0.5,
        seed=42,
    )


@pytest.fixture
def default_dataset(default_config) -> SyntheticAudioDataset:
    ds = SyntheticAudioDataset(config=default_config)
    ds.generate()
    return ds


# ==================================================================
# 1. Generation
# ==================================================================


class TestGeneration:

    def test_generates_correct_total_count(self, default_config):
        """samples_per_type * len(signal_types) total samples."""
        ds = SyntheticAudioDataset(config=default_config)
        samples = ds.generate()
        expected = default_config.samples_per_type * len(default_config.signal_types)
        assert len(samples) == expected

    def test_generates_all_six_supported_types(self, default_dataset):
        types_found = {s.meta.signal_type for s in default_dataset}
        assert types_found == set(AUDIO_SIGNAL_TYPES)

    def test_each_type_has_correct_sample_count(self, default_config, default_dataset):
        from collections import Counter
        counts = Counter(s.meta.signal_type for s in default_dataset)
        for sig_type in AUDIO_SIGNAL_TYPES:
            assert counts[sig_type] == default_config.samples_per_type

    def test_samples_are_audio_sample_instances(self, default_dataset):
        for sample in default_dataset:
            assert isinstance(sample, AudioSample)

    def test_metadata_is_audio_sample_metadata(self, default_dataset):
        for sample in default_dataset:
            assert isinstance(sample.meta, AudioSampleMetadata)


# ==================================================================
# 2. Dimensions and sampling rate
# ==================================================================


class TestDimensions:

    def test_signal_length_matches_metadata(self, default_dataset):
        for sample in default_dataset:
            expected_samples = int(round(
                sample.meta.sampling_rate * sample.meta.duration
            ))
            assert len(sample.signal) == expected_samples, (
                f"Type={sample.meta.signal_type}: "
                f"expected {expected_samples} samples, got {len(sample.signal)}"
            )

    def test_time_vector_length_matches_signal(self, default_dataset):
        for sample in default_dataset:
            assert len(sample.t) == len(sample.signal)

    def test_signal_is_1d(self, default_dataset):
        for sample in default_dataset:
            assert sample.signal.ndim == 1, (
                f"Type={sample.meta.signal_type}: signal must be 1-D"
            )

    def test_signal_dtype_is_float64(self, default_dataset):
        for sample in default_dataset:
            assert sample.signal.dtype == np.float64

    def test_num_samples_in_metadata_is_correct(self, default_dataset):
        for sample in default_dataset:
            assert sample.meta.num_samples == len(sample.signal)

    def test_sampling_rate_in_metadata(self, default_config, default_dataset):
        for sample in default_dataset:
            assert sample.meta.sampling_rate == default_config.sampling_rate


# ==================================================================
# 3. Signal quality — all values must be finite
# ==================================================================


class TestSignalQuality:

    def test_all_signals_are_finite(self, default_dataset):
        for sample in default_dataset:
            assert np.all(np.isfinite(sample.signal)), (
                f"Type={sample.meta.signal_type}: non-finite values detected."
            )

    def test_time_vectors_are_finite(self, default_dataset):
        for sample in default_dataset:
            assert np.all(np.isfinite(sample.t))

    def test_time_starts_at_zero(self, default_dataset):
        for sample in default_dataset:
            assert sample.t[0] == pytest.approx(0.0)

    def test_time_step_equals_inverse_sampling_rate(self, default_dataset):
        for sample in default_dataset:
            dt = sample.t[1] - sample.t[0]
            assert dt == pytest.approx(1.0 / sample.meta.sampling_rate, rel=1e-6)


# ==================================================================
# 4. Reproducibility
# ==================================================================


class TestReproducibility:

    def test_same_seed_produces_identical_signals(self, default_config):
        ds_a = SyntheticAudioDataset(config=default_config)
        ds_b = SyntheticAudioDataset(config=default_config)
        samples_a = ds_a.generate()
        samples_b = ds_b.generate()

        for sa, sb in zip(samples_a, samples_b):
            np.testing.assert_array_equal(
                sa.signal, sb.signal,
                err_msg=f"Type={sa.meta.signal_type}: signals differ with same seed."
            )

    def test_different_seeds_produce_different_signals(self, default_config):
        config_b = AudioDatasetConfig(
            **{**default_config.__dict__, "seed": 99}
        )
        ds_a = SyntheticAudioDataset(config=default_config)
        ds_b = SyntheticAudioDataset(config=config_b)
        ds_a.generate()
        ds_b.generate()

        # At least one signal pair must differ
        any_different = any(
            not np.array_equal(sa.signal, sb.signal)
            for sa, sb in zip(ds_a._samples, ds_b._samples)
        )
        assert any_different, "Expected at least one signal to differ across seeds."

    def test_generate_twice_gives_same_result(self, default_config):
        ds = SyntheticAudioDataset(config=default_config)
        first = ds.generate()
        second = ds.generate()

        for sa, sb in zip(first, second):
            np.testing.assert_array_equal(sa.signal, sb.signal)


# ==================================================================
# 5. Metadata completeness
# ==================================================================


class TestMetadata:

    def test_domain_is_always_audio(self, default_dataset):
        for sample in default_dataset:
            assert sample.meta.domain == "audio"

    def test_sample_id_is_unique(self, default_dataset):
        ids = [s.meta.sample_id for s in default_dataset]
        assert len(ids) == len(set(ids)), "All sample_ids must be unique."

    def test_metadata_dict_contains_required_keys(self, default_dataset):
        required = {
            "sample_id",
            "domain",
            "signal_type",
            "sampling_rate",
            "duration",
            "num_samples",
            "seed",
        }
        for m in default_dataset.metadata():
            assert required.issubset(m.keys()), (
                f"Missing keys: {required - m.keys()}"
            )

    def test_metadata_is_json_serialisable(self, default_dataset):
        for m in default_dataset.metadata():
            # Should not raise
            json.dumps(m)

    def test_metadata_list_length_matches_samples(self, default_dataset):
        assert len(default_dataset.metadata()) == len(default_dataset)


# ==================================================================
# 6. Save / Load round-trip
# ==================================================================


class TestSaveLoad:

    def test_save_creates_npy_and_json_files(self, default_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = default_dataset.save(tmp)
            files = list(out_dir.iterdir())
            npy_files = [f for f in files if f.suffix == ".npy"]
            json_files = [f for f in files if f.suffix == ".json"]
            assert len(npy_files) >= len(default_dataset)
            assert len(json_files) >= len(default_dataset)

    def test_load_restores_same_number_of_samples(self, default_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            default_dataset.save(tmp)
            loaded = SyntheticAudioDataset.load(tmp)
        assert len(loaded) == len(default_dataset)

    def test_load_restores_signals_exactly(self, default_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            default_dataset.save(tmp)
            loaded = SyntheticAudioDataset.load(tmp)

        for orig, rest in zip(default_dataset._samples, loaded._samples):
            np.testing.assert_array_equal(
                orig.signal,
                rest.signal,
                err_msg=f"Signal mismatch for sample {orig.meta.sample_id}",
            )

    def test_load_restores_metadata_fields(self, default_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            default_dataset.save(tmp)
            loaded = SyntheticAudioDataset.load(tmp)

        for orig, rest in zip(default_dataset._samples, loaded._samples):
            assert orig.meta.sample_id == rest.meta.sample_id
            assert orig.meta.signal_type == rest.meta.signal_type
            assert orig.meta.sampling_rate == rest.meta.sampling_rate

    def test_save_requires_prior_generate(self, default_config):
        ds = SyntheticAudioDataset(config=default_config)
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(RuntimeError):
                ds.save(tmp)

    def test_load_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            SyntheticAudioDataset.load("/this/does/not/exist_spandhan_test")

    def test_config_file_written_and_readable(self, default_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = default_dataset.save(tmp)
            config_file = out_dir / "dataset_config.json"
            assert config_file.exists()
            data = json.loads(config_file.read_text())
            assert "sampling_rate" in data
            assert "seed" in data


# ==================================================================
# 7. Invalid configuration
# ==================================================================


class TestInvalidConfig:

    def test_zero_samples_per_type_raises(self):
        with pytest.raises(ValueError):
            build_audio_dataset(samples_per_type=0)

    def test_negative_sampling_rate_raises(self):
        with pytest.raises(ValueError):
            build_audio_dataset(sampling_rate=-1.0)

    def test_zero_sampling_rate_raises(self):
        with pytest.raises(ValueError):
            build_audio_dataset(sampling_rate=0.0)

    def test_zero_duration_raises(self):
        with pytest.raises(ValueError):
            build_audio_dataset(duration=0.0)

    def test_non_integer_seed_raises(self):
        with pytest.raises(ValueError):
            build_audio_dataset(seed="42")  # type: ignore[arg-type]

    def test_empty_signal_types_raises(self):
        config = AudioDatasetConfig(signal_types=[], samples_per_type=1, seed=42)
        ds = SyntheticAudioDataset(config=config)
        with pytest.raises(ValueError):
            ds.generate()

    def test_unsupported_signal_type_raises(self):
        config = AudioDatasetConfig(
            signal_types=["unsupported_xyz"],
            samples_per_type=1,
            seed=42,
        )
        ds = SyntheticAudioDataset(config=config)
        with pytest.raises(ValueError):
            ds.generate()


# ==================================================================
# 8. Low-level generator unit tests
# ==================================================================


class TestChirpGenerator:

    def test_output_shape(self):
        sr, dur = 8000.0, 0.5
        expected = int(round(sr * dur))
        t, sig = _generate_chirp(f0=100.0, f1=1000.0, sampling_rate=sr, duration=dur)
        assert len(sig) == expected
        assert len(t) == expected

    def test_finite_output(self):
        _, sig = _generate_chirp(f0=100.0, f1=2000.0, sampling_rate=8000.0, duration=1.0)
        assert np.all(np.isfinite(sig))

    def test_amplitude_scaling(self):
        _, sig_1 = _generate_chirp(100, 1000, 8000, 0.5, amplitude=1.0)
        _, sig_2 = _generate_chirp(100, 1000, 8000, 0.5, amplitude=2.0)
        np.testing.assert_allclose(sig_2, sig_1 * 2.0)

    def test_negative_frequency_raises(self):
        with pytest.raises(ValueError):
            _generate_chirp(f0=-1.0, f1=1000.0, sampling_rate=8000.0, duration=0.5)

    def test_negative_amplitude_raises(self):
        with pytest.raises(ValueError):
            _generate_chirp(100, 1000, 8000, 0.5, amplitude=-1.0)


class TestAmplitudeModulatedGenerator:

    def test_output_shape(self):
        sr, dur = 8000.0, 0.5
        expected = int(round(sr * dur))
        t, sig = _generate_amplitude_modulated(440.0, 5.0, 0.5, sr, dur)
        assert len(sig) == expected

    def test_finite_output(self):
        _, sig = _generate_amplitude_modulated(440.0, 5.0, 0.5, 8000.0, 1.0)
        assert np.all(np.isfinite(sig))

    def test_zero_modulation_depth_is_pure_carrier(self):
        sr, dur = 8000.0, 0.25
        amp = 1.0
        _, sig_am = _generate_amplitude_modulated(440.0, 5.0, 0.0, sr, dur, amplitude=amp)
        t = np.arange(int(round(sr * dur))) / sr
        expected = amp * np.cos(2 * np.pi * 440.0 * t)
        np.testing.assert_allclose(sig_am, expected, atol=1e-10)

    def test_invalid_modulation_depth_raises(self):
        with pytest.raises(ValueError):
            _generate_amplitude_modulated(440.0, 5.0, 1.5, 8000.0, 0.5)

    def test_nonpositive_carrier_raises(self):
        with pytest.raises(ValueError):
            _generate_amplitude_modulated(0.0, 5.0, 0.5, 8000.0, 0.5)


class TestFrequencyModulatedGenerator:

    def test_output_shape(self):
        sr, dur = 8000.0, 0.5
        expected = int(round(sr * dur))
        t, sig = _generate_frequency_modulated(440.0, 5.0, 2.0, sr, dur)
        assert len(sig) == expected

    def test_finite_output(self):
        _, sig = _generate_frequency_modulated(440.0, 5.0, 2.0, 8000.0, 1.0)
        assert np.all(np.isfinite(sig))

    def test_zero_modulation_index_is_pure_carrier(self):
        sr, dur = 8000.0, 0.25
        amp = 1.0
        _, sig_fm = _generate_frequency_modulated(440.0, 5.0, 0.0, sr, dur, amplitude=amp)
        t = np.arange(int(round(sr * dur))) / sr
        expected = amp * np.cos(2 * np.pi * 440.0 * t)
        np.testing.assert_allclose(sig_fm, expected, atol=1e-10)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError):
            _generate_frequency_modulated(440.0, 5.0, -1.0, 8000.0, 0.5)


class TestHarmonicGenerator:

    def test_output_shape(self):
        sr, dur = 8000.0, 0.5
        expected = int(round(sr * dur))
        t, sig = _generate_harmonic(220.0, sr, dur, num_harmonics=4)
        assert len(sig) == expected

    def test_finite_output(self):
        _, sig = _generate_harmonic(220.0, 8000.0, 1.0, num_harmonics=5)
        assert np.all(np.isfinite(sig))

    def test_single_harmonic_equals_sine(self):
        """1 harmonic with decay=1.0 should equal a pure sine."""
        sr, dur, f = 8000.0, 0.5, 220.0
        _, sig_h = _generate_harmonic(f, sr, dur, num_harmonics=1, amplitude=1.0, harmonic_decay=1.0)
        from generators.waveform_generator import generate_sine
        _, sig_s = generate_sine(f, sr, dur, amplitude=1.0)
        np.testing.assert_allclose(sig_h, sig_s, atol=1e-10)

    def test_invalid_fundamental_raises(self):
        with pytest.raises(ValueError):
            _generate_harmonic(0.0, 8000.0, 0.5, num_harmonics=4)

    def test_invalid_num_harmonics_raises(self):
        with pytest.raises(ValueError):
            _generate_harmonic(220.0, 8000.0, 0.5, num_harmonics=0)

    def test_invalid_decay_raises(self):
        with pytest.raises(ValueError):
            _generate_harmonic(220.0, 8000.0, 0.5, num_harmonics=3, harmonic_decay=1.5)


# ==================================================================
# 9. build_audio_dataset() factory convenience tests
# ==================================================================


class TestBuildAudioDataset:

    def test_returns_synthetic_audio_dataset(self):
        ds = build_audio_dataset(samples_per_type=1, sampling_rate=8000.0, duration=0.25, seed=0)
        assert isinstance(ds, SyntheticAudioDataset)

    def test_samples_are_already_generated(self):
        ds = build_audio_dataset(samples_per_type=1, sampling_rate=8000.0, duration=0.25, seed=0)
        assert len(ds) > 0

    def test_specific_types_subset(self):
        types = ["clean_sine", "chirp"]
        ds = build_audio_dataset(
            signal_types=types,
            samples_per_type=2,
            sampling_rate=8000.0,
            duration=0.25,
            seed=1,
        )
        found = {s.meta.signal_type for s in ds}
        assert found == set(types)

    def test_dataset_is_reproducible_via_factory(self):
        ds_a = build_audio_dataset(samples_per_type=2, sampling_rate=8000.0, duration=0.25, seed=7)
        ds_b = build_audio_dataset(samples_per_type=2, sampling_rate=8000.0, duration=0.25, seed=7)
        for sa, sb in zip(ds_a._samples, ds_b._samples):
            np.testing.assert_array_equal(sa.signal, sb.signal)


# ==================================================================
# 10. dunder interface
# ==================================================================


class TestDunderInterface:

    def test_len(self, default_dataset, default_config):
        expected = default_config.samples_per_type * len(AUDIO_SIGNAL_TYPES)
        assert len(default_dataset) == expected

    def test_iter(self, default_dataset):
        count = sum(1 for _ in default_dataset)
        assert count == len(default_dataset)

    def test_getitem(self, default_dataset):
        first = default_dataset[0]
        assert isinstance(first, AudioSample)
