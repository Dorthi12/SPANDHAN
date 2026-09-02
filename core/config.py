"""
Config module.
"""
APP_NAME = "Spandhan"
VERSION = "1.0.0"

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
    "rms",
    "variance",
    "std",
    "kurtosis",
    "skewness",
    "crest_factor",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_flatness",
    "spectral_entropy",
    "spectral_rolloff",
    "mains_band_energy",
    "high_band_energy",
    "snr_db",
]

DEFAULT_CONFIDENCE_THRESHOLD = 0.60