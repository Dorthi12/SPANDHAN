"""
domains/audio/experiment.py
============================
Audio synthetic experiment orchestrator — Milestone D.

Wires together the full A → B → C pipeline:
    A. Generate synthetic audio dataset  (generators.audio_dataset_generator)
    B. Inject noise                      (domains.audio.noise_pipeline)
    C. Analyze clean + noisy             (domains.audio.analyzer)
    D. Denoise                           (preprocessing.audio_denoising)
    E. Analyze cleaned                   (domains.audio.analyzer)
    F. Before/after quality comparison   (intelligence.noise.comparison)

Returns a single typed AudioExperimentResult that the frontend
renders directly — no computation happens in the UI layer.

Design rules
------------
* ALL displayed results come from actual computation. Nothing is faked.
* Every experiment is identified by: seed, signal_type, noise_type,
  noise_config, denoise_method, sampling_rate, duration, timestamp.
* ground_truth_noise is the generator label — never the classifier output.
* measured_snr_db is independently computed — never the target_snr_db.
* The frontend calls run_audio_experiment() and renders the result.
  It never calls individual pipeline stages directly.
* Replace SyntheticAudioDataset with RealAudioDataset here without
  changing the AudioExperimentResult contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from generators.audio_dataset_generator import AudioDatasetConfig, SyntheticAudioDataset
from domains.audio.noise_pipeline import AudioNoiseInjectionResult, inject_audio_noise
from domains.audio.analyzer import AudioSignalAnalysis, analyze_audio_signal
from preprocessing.audio_denoising import AudioDenoisingResult, denoise_audio, recommend_method
from intelligence.noise.comparison import compare_signal_quality


# ==================================================================
# Pipeline version — bump when the pipeline logic changes
# ==================================================================

PIPELINE_VERSION = "1.0.0"


# ==================================================================
# Experiment configuration (fully serialisable)
# ==================================================================


@dataclass
class AudioExperimentConfig:
    """
    All parameters that define one audio experiment run.

    Every field is a primitive so the config can be serialised to JSON
    without custom encoding.
    """

    # Dataset generation
    seed: int = 42
    signal_type: str = "clean_sine"    # must be in AUDIO_SIGNAL_TYPES
    sampling_rate: float = 8000.0
    duration: float = 1.0
    amplitude: float = 1.0

    # Signal-specific parameters (forwarded to generator as needed)
    frequency: float = 440.0        # for sine / harmonic / AM / FM
    num_tones: int = 3              # for multi_tone
    chirp_f0: float = 200.0
    chirp_f1: float = 2000.0

    # Noise injection
    noise_type: str = "gaussian"    # must be in AUDIO_NOISE_TYPES
    target_snr_db: float = 10.0     # configuration value — NOT measured
    impulse_probability: float = 0.01
    impulse_amplitude_factor: float = 5.0
    periodic_frequency: float = 50.0
    periodic_amplitude: float = 0.1
    color: str = "pink"
    colored_strength: float = 0.1
    mixed_periodic_frequency: float = 50.0
    mixed_periodic_amplitude: float = 0.05
    mixed_impulse_probability: float = 0.005

    # Denoising
    denoise_method: str = "auto"    # "auto" → recommend_method(noise_type)
    filter_order: int = 4
    cutoff_hz: float = 3000.0
    notch_freq_hz: float = 50.0
    notch_bandwidth_hz: float = 10.0
    bandpass_low_hz: float = 300.0
    bandpass_high_hz: float = 3000.0


# ==================================================================
# Experiment result (fully typed)
# ==================================================================


@dataclass
class AudioExperimentResult:
    """
    Complete result of one audio experiment run.

    Contains every signal, every analysis, and all quality metrics.
    The frontend renders this directly — no further computation.

    Fields
    ------
    config : AudioExperimentConfig
        The exact parameters that produced this result (for reproducibility).
    timestamp : str
        ISO 8601 UTC timestamp of when the experiment was run.
    pipeline_version : str
        Version of the pipeline that produced this result.

    --- Signals ---
    clean_signal : np.ndarray
    noisy_signal : np.ndarray
    cleaned_signal : np.ndarray
    time_axis : np.ndarray
    sampling_rate : float

    --- Injection ---
    injection : AudioNoiseInjectionResult
        ground_truth_noise = config.noise_type  (generator label)
        measured_snr_db    = independently measured SNR (not target)

    --- Analysis (three stages) ---
    clean_analysis : AudioSignalAnalysis   (stage="clean")
    noisy_analysis : AudioSignalAnalysis   (stage="noisy")
    cleaned_analysis : AudioSignalAnalysis (stage="cleaned")

    --- Denoising ---
    denoising : AudioDenoisingResult
        snr_improvement_db and rmse_reduction are independently computed.

    --- Before/After Quality Comparison ---
    quality_comparison : dict
        Output of compare_signal_quality(clean, noisy, cleaned).
        Keys: "before", "after", "improvement", "reference_energy".

    --- Timing ---
    total_time_s : float
    stage_times_s : dict[str, float]

    --- Diagnostics ---
    warnings : list[str]
    errors : list[str]
    """

    config: AudioExperimentConfig
    timestamp: str
    pipeline_version: str

    # Signals
    clean_signal: np.ndarray = field(repr=False)
    noisy_signal: np.ndarray = field(repr=False)
    cleaned_signal: np.ndarray = field(repr=False)
    time_axis: np.ndarray = field(repr=False)
    sampling_rate: float = 0.0

    # Sub-results
    injection: Optional[AudioNoiseInjectionResult] = None
    clean_analysis: Optional[AudioSignalAnalysis] = None
    noisy_analysis: Optional[AudioSignalAnalysis] = None
    cleaned_analysis: Optional[AudioSignalAnalysis] = None
    denoising: Optional[AudioDenoisingResult] = None

    # Quality comparison
    quality_comparison: dict[str, Any] = field(default_factory=dict)

    # Timing
    total_time_s: float = 0.0
    stage_times_s: dict[str, float] = field(default_factory=dict)

    # Diagnostics
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ==================================================================
# Pipeline runner
# ==================================================================


def run_audio_experiment(config: AudioExperimentConfig) -> AudioExperimentResult:
    """
    Execute the full audio experiment pipeline.

    Parameters
    ----------
    config : AudioExperimentConfig
        All parameters for this experiment run.

    Returns
    -------
    AudioExperimentResult
        Complete typed result — ready for the frontend to render.
    """
    t_total = time.perf_counter()
    stage_times: dict[str, float] = {}
    warnings: list[str] = []
    errors: list[str] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    sr = float(config.sampling_rate)
    n_samples = int(sr * config.duration)

    # Placeholders (filled progressively)
    clean = np.zeros(n_samples, dtype=np.float64)
    noisy = np.zeros(n_samples, dtype=np.float64)
    cleaned = np.zeros(n_samples, dtype=np.float64)
    t_axis = np.arange(n_samples) / sr
    injection_result: Optional[AudioNoiseInjectionResult] = None
    clean_analysis: Optional[AudioSignalAnalysis] = None
    noisy_analysis: Optional[AudioSignalAnalysis] = None
    cleaned_analysis: Optional[AudioSignalAnalysis] = None
    denoising_result: Optional[AudioDenoisingResult] = None
    quality_cmp: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Stage A: Generate clean synthetic signal
    # ------------------------------------------------------------------
    # Map user-friendly aliases to the generator's internal type names
    _TYPE_ALIAS: dict[str, str] = {
        "sine":                "clean_sine",
        "clean_sine":          "clean_sine",
        "multi_tone":          "multi_tone",
        "harmonic":            "harmonic",
        "chirp":               "chirp",
        "amplitude_modulated": "amplitude_modulated",
        "frequency_modulated": "frequency_modulated",
    }
    t0 = time.perf_counter()
    try:
        internal_type = _TYPE_ALIAS.get(config.signal_type, config.signal_type)
        dataset_cfg = AudioDatasetConfig(
            seed=config.seed,
            sampling_rate=sr,
            duration=config.duration,
            signal_types=[internal_type],
            samples_per_type=1,
            # Shared scalar overrides (generator randomises around these)
            frequency=config.frequency,
            amplitude=config.amplitude,
            fundamental=config.frequency,          # for harmonic
            chirp_f0=config.chirp_f0,
            chirp_f1=config.chirp_f1,
            am_carrier_freq=config.frequency,
            fm_carrier_freq=config.frequency,
        )
        dataset = SyntheticAudioDataset(dataset_cfg)
        samples = dataset.generate()
        if samples:
            clean = samples[0].signal.copy()
            t_axis = samples[0].t.copy()
        else:
            warnings.append("Dataset generator returned no samples; using zero signal.")
    except Exception as exc:
        errors.append(f"Stage A (generate): {exc}")
        warnings.append("Clean signal defaulted to zeros due to generation error.")
    stage_times["A_generate"] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Stage B: Inject noise
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        injection_kwargs: dict[str, Any] = {
            "noise_type": config.noise_type,
            "target_snr_db": config.target_snr_db,
            "seed": config.seed,
            "impulse_probability": config.impulse_probability,
            "impulse_amplitude_factor": config.impulse_amplitude_factor,
            "periodic_frequency": config.periodic_frequency,
            "periodic_amplitude": config.periodic_amplitude,
            "color": config.color,
            "colored_strength": config.colored_strength,
            "mixed_periodic_frequency": config.mixed_periodic_frequency,
            "mixed_periodic_amplitude": config.mixed_periodic_amplitude,
            "mixed_impulse_probability": config.mixed_impulse_probability,
        }
        injection_result = inject_audio_noise(clean, sr, **injection_kwargs)
        noisy = injection_result.noisy_signal.copy()
    except Exception as exc:
        errors.append(f"Stage B (inject): {exc}")
        noisy = clean.copy()
        warnings.append("Noise injection failed; noisy signal equals clean signal.")
    stage_times["B_inject"] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Stage C: Analyze clean and noisy signals
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        clean_analysis = analyze_audio_signal(
            clean, sr,
            stage="clean",
            ground_truth_noise=None,   # clean has no noise label
            measured_snr_db=None,
        )
    except Exception as exc:
        errors.append(f"Stage C (analyze clean): {exc}")
    try:
        measured_snr = (
            injection_result.measured_snr_db
            if injection_result is not None
            else None
        )
        noisy_analysis = analyze_audio_signal(
            noisy, sr,
            stage="noisy",
            ground_truth_noise=config.noise_type,   # generator label
            measured_snr_db=measured_snr,
        )
    except Exception as exc:
        errors.append(f"Stage C (analyze noisy): {exc}")
    stage_times["C_analyze_input"] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Stage D: Denoise
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        method = (
            recommend_method(config.noise_type)
            if config.denoise_method == "auto"
            else config.denoise_method
        )
        # Build denoiser kwargs — only include params relevant to the chosen method
        denoise_kwargs: dict[str, Any] = {
            "filter_order": config.filter_order,
            "cutoff_hz": config.cutoff_hz,
            "notch_freq_hz": config.notch_freq_hz,
            "notch_bandwidth_hz": config.notch_bandwidth_hz,
            "bandpass_low_hz": config.bandpass_low_hz,
            "bandpass_high_hz": config.bandpass_high_hz,
            "staged_notch_freq_hz": config.notch_freq_hz,
            "staged_notch_bandwidth_hz": config.notch_bandwidth_hz,
        }
        denoising_result = denoise_audio(
            noisy, sr,
            method=method,
            clean_reference=clean,
            **denoise_kwargs,
        )
        cleaned = denoising_result.cleaned_signal.copy()
        if denoising_result.warnings:
            warnings.extend(denoising_result.warnings)
    except Exception as exc:
        errors.append(f"Stage D (denoise): {exc}")
        cleaned = noisy.copy()
        warnings.append("Denoising failed; cleaned signal equals noisy signal.")
    stage_times["D_denoise"] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Stage E: Analyze cleaned signal
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        cleaned_analysis = analyze_audio_signal(
            cleaned, sr,
            stage="cleaned",
            ground_truth_noise=config.noise_type,
            measured_snr_db=None,
        )
    except Exception as exc:
        errors.append(f"Stage E (analyze cleaned): {exc}")
    stage_times["E_analyze_output"] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Stage F: Before/after quality comparison
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        quality_cmp = compare_signal_quality(clean, noisy, cleaned, sr)
    except Exception as exc:
        errors.append(f"Stage F (compare): {exc}")
        quality_cmp = {}
    stage_times["F_compare"] = time.perf_counter() - t0

    total = time.perf_counter() - t_total

    return AudioExperimentResult(
        config=config,
        timestamp=timestamp,
        pipeline_version=PIPELINE_VERSION,
        clean_signal=clean,
        noisy_signal=noisy,
        cleaned_signal=cleaned,
        time_axis=t_axis,
        sampling_rate=sr,
        injection=injection_result,
        clean_analysis=clean_analysis,
        noisy_analysis=noisy_analysis,
        cleaned_analysis=cleaned_analysis,
        denoising=denoising_result,
        quality_comparison=quality_cmp,
        total_time_s=float(total),
        stage_times_s=stage_times,
        warnings=warnings,
        errors=errors,
    )
