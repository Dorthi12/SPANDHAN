"""
Config module for Spandhan DSP Application.
"""

APP_NAME = "Spandhan"
APP_SUBTITLE = "Multi-Domain Digital Signal Analysis & Diagnostics Platform"
VERSION = "1.0.0"
DEFAULT_MODEL_PATH = "models/noise_model.pkl"

SUPPORTED_DOMAINS = [
    "general",
    "audio",
    "ecg",
    "mcsa",
]

SUPPORTED_AUDIO_EXTENSIONS = [
    ".wav",
    ".flac",
    ".mp3",
]

SUPPORTED_SIGNAL_EXTENSIONS = [
    ".csv",
    ".mat",
    ".txt",
]

ML_CLASSES = [
    "Clean",
    "Gaussian",
    "Impulse",
    "Periodic",
    "Colored",
    "Mixed",
]

ML_FEATURE_NAMES = [
    # Time-domain statistics
    "rms",
    "variance",
    "std",
    "kurtosis",
    "skewness",
    "crest_factor",
    "zero_crossing_rate",
    # Spectral features
    "spectral_centroid",
    "spectral_flatness",
    "spectral_entropy",
    "spectral_rolloff",
    "mains_band_energy",
    "high_band_energy",
    # SNR
    "snr_db",
    # New discriminative features
    "impulse_count",
    "peak_count_rate",
    "energy_ratio_first_half",
    "spectral_variance",
    "low_band_energy",
    "mid_band_energy",
]

DEFAULT_CONFIDENCE_THRESHOLD = 0.60