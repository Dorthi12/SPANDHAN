"""
SPANDHAN — ML Pipeline Configuration  (Audio and Image)
=========================================================
All path and hyper-parameter constants for audio and image ML pipelines.
Change them here; other modules pick them up automatically.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root  (this file lives at <root>/ml/common/config.py)
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
REPO_ROOT: Path = _THIS_FILE.parents[2]          # …/new_update/

# ---------------------------------------------------------------------------
# Dataset paths  (MATLAB-processed WAV → Python input)
# ---------------------------------------------------------------------------
DATASET_ROOT: Path = REPO_ROOT / "datasets" / "audio"

CLASS_DIRS: dict[str, Path] = {
    "impulse":    DATASET_ROOT / "impulse",
    "sinusoidal": DATASET_ROOT / "sinusoidal",
    "white_noise": DATASET_ROOT / "white_noise",
    "step":       DATASET_ROOT / "step",
    "chirp":      DATASET_ROOT / "chirp",
}

# ---------------------------------------------------------------------------
# Model artefacts  (saved once after training; loaded at inference)
# ---------------------------------------------------------------------------
MODELS_DIR: Path      = REPO_ROOT / "models" / "audio"
PIPELINE_PATH: Path   = MODELS_DIR / "audio_signal_classifier.pkl"   # full sklearn Pipeline

# Saved dataset arrays (intermediate, produced by build_dataset.py)
REPORTS_DIR: Path     = REPO_ROOT / "data" / "output" / "predictions"
X_PATH: Path          = REPORTS_DIR / "X.npy"
Y_PATH: Path          = REPORTS_DIR / "y.npy"
FILENAMES_PATH: Path  = REPORTS_DIR / "filenames.npy"

# Model comparison report
COMPARISON_REPORT: Path = REPORTS_DIR / "model_comparison.json"

# ---------------------------------------------------------------------------
# Audio feature settings
# ---------------------------------------------------------------------------
N_FFT: int       = 2048
HOP_LENGTH: int  = 512
N_MFCC: int      = 13

# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------
TRAIN_RATIO:  float = 0.70
VAL_RATIO:    float = 0.15
TEST_RATIO:   float = 0.15
RANDOM_STATE: int   = 42

# ---------------------------------------------------------------------------
# Cross-validation folds used during model comparison
# ---------------------------------------------------------------------------
CV_FOLDS: int = 5

# ---------------------------------------------------------------------------
# Candidate model hyper-parameter grids  (used by compare_models.py)
# GridSearch is run on the TRAIN split only; test set stays untouched.
# ---------------------------------------------------------------------------
LR_PARAM_GRID: dict = {
    "classifier__C":        [0.01, 0.1, 1.0, 10.0],
    "classifier__solver":   ["lbfgs", "saga"],
    "classifier__max_iter": [1000],
}

SVM_PARAM_GRID: dict = {
    "classifier__C":      [0.1, 1.0, 10.0],
    "classifier__kernel": ["rbf", "poly"],
    "classifier__gamma":  ["scale", "auto"],
}

RF_PARAM_GRID: dict = {
    "classifier__n_estimators":      [100, 200, 300],
    "classifier__max_depth":         [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
}

GB_PARAM_GRID: dict = {
    "classifier__n_estimators":  [100, 200],
    "classifier__learning_rate": [0.05, 0.1, 0.2],
    "classifier__max_depth":     [3, 5, 7],
}

# ---------------------------------------------------------------------------
# Image ML Pipeline Settings
# ---------------------------------------------------------------------------
IMAGE_DATASET_ROOT: Path = REPO_ROOT / "datasets" / "image"
IMAGE_MODELS_DIR: Path   = REPO_ROOT / "models" / "image"
IMAGE_PIPELINE_PATH: Path = IMAGE_MODELS_DIR / "image_signal_classifier.pkl"
IMAGE_KERAS_PATH: Path   = IMAGE_MODELS_DIR / "image_signal_classifier.keras"

IMAGE_REPORTS_DIR: Path  = REPO_ROOT / "data" / "output" / "image_predictions"
IMAGE_X_PATH: Path       = IMAGE_REPORTS_DIR / "X.npy"
IMAGE_Y_PATH: Path       = IMAGE_REPORTS_DIR / "y.npy"
IMAGE_FILENAMES_PATH: Path = IMAGE_REPORTS_DIR / "filenames.npy"
IMAGE_COMPARISON_REPORT: Path = IMAGE_REPORTS_DIR / "experiment_comparison.json"
IMAGE_EVALUATION_REPORT: Path = IMAGE_REPORTS_DIR / "evaluation_report.json"
IMAGE_AUDIT_REPORT: Path = IMAGE_REPORTS_DIR / "audit_report.json"

IMAGE_CLASS_DIRS: dict[str, Path] = {
    "impulse":     IMAGE_DATASET_ROOT / "impulse",
    "sinusoidal":  IMAGE_DATASET_ROOT / "sinusoidal",
    "white_noise": IMAGE_DATASET_ROOT / "white_noise",
    "step":        IMAGE_DATASET_ROOT / "step",
    "chirp":       IMAGE_DATASET_ROOT / "chirp",
}

# Image contract
IMAGE_HEIGHT: int = 128
IMAGE_WIDTH: int  = 128
IMAGE_CHANNELS: int = 1

# Image training hyper-parameters
IMAGE_BATCH_SIZE: int    = 32
IMAGE_MAX_EPOCHS: int    = 60
IMAGE_LR: float          = 1e-3
IMAGE_WEIGHT_DECAY: float = 1e-4
IMAGE_DROPOUT: float     = 0.4
IMAGE_PATIENCE: int      = 10

