import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from intelligence.noise.features import extract_noise_features
from intelligence.noise.model_io import load_noise_model
from intelligence.noise.estimator import estimate_noise


MODEL_PATH = PROJECT_ROOT / "models" / "noise_classifier.joblib"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

SAMPLING_RATE = 16000
TARGET_SNR_DB = 15.0
IMPULSE_PROBABILITY = 0.005
IMPULSE_AMPLITUDE_FACTOR = 5.0


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def add_gaussian_noise(signal, snr_db, random_state=42):
    """Add Gaussian noise at approximately the requested SNR."""
    rng = np.random.default_rng(random_state)

    signal = np.asarray(signal, dtype=float)

    signal_power = np.mean(signal ** 2)

    if signal_power <= 0:
        raise ValueError("Signal power must be positive.")

    noise_power = signal_power / (10.0 ** (snr_db / 10.0))

    noise = rng.normal(
        0.0,
        np.sqrt(noise_power),
        size=signal.shape,
    )

    return signal + noise


def add_impulse_noise(
    signal,
    probability=0.005,
    amplitude_factor=5.0,
    random_state=42,
):
    """Add sparse impulsive disturbances."""
    rng = np.random.default_rng(random_state)

    signal = np.asarray(signal, dtype=float).copy()

    std = np.std(signal)

    if std <= 0:
        raise ValueError("Signal standard deviation must be positive.")

    mask = rng.random(signal.shape) < probability

    impulses = rng.choice(
        [-1.0, 1.0],
        size=signal.shape,
    )

    signal[mask] += (
        impulses[mask]
        * amplitude_factor
        * std
    )

    return signal


def analyze_case(name, clean, noisy, sampling_rate, model):
    """Extract features, estimate SNR, and classify one signal."""

    features = extract_noise_features(
        noisy,
        sampling_rate,
    )

    # Independent reference-free SNR estimate.
    estimated = estimate_noise(
        clean,
        noisy,
    )

    feature_vector = np.array(
        [
            features[name]
            for name in model.feature_names_in_
        ],
        dtype=float,
    )

    prediction = model.predict(
        feature_vector.reshape(1, -1)
    )[0]

    probabilities = model.predict_proba(
        feature_vector.reshape(1, -1)
    )[0]

    confidence = float(np.max(probabilities))

    print("\n" + "-" * 60)
    print(name.upper())
    print("-" * 60)

    print(f"Actual SNR:      {estimate_noise(clean, noisy)['snr_db']:.2f} dB")
    print(f"Estimated SNR:   {estimated['snr_db']:.2f} dB")
    print(f"Predicted class: {prediction}")
    print(f"Confidence:      {confidence:.4f}")

    print("\nClass probabilities:")

    for class_name, probability in zip(
        model.classes_,
        probabilities,
    ):
        print(
            f"  {class_name:<10}: "
            f"{probability:.4f}"
        )

    return {
        "name": name,
        "features": features,
        "prediction": prediction,
        "confidence": confidence,
        "actual_snr": estimated["snr_db"],
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

print("=" * 60)
print("Spandhan - Controlled Real Audio Noise Validation")
print("=" * 60)


# ------------------------------------------------------------
# 1. Find one UrbanSound8K WAV file
# ------------------------------------------------------------

print("\n[1/5] Locating UrbanSound8K recording...")

candidates = list(
    Path("datasets").rglob("*.wav")
)

if not candidates:
    raise FileNotFoundError(
        "No WAV file found under the project datasets/ directory."
    )

audio_path = candidates[0]

print(f"      File: {audio_path}")


# ------------------------------------------------------------
# 2. Load audio
# ------------------------------------------------------------

print("\n[2/5] Loading audio...")

from data_io.audio_loader import load_real_audio

clean, original_fs = load_real_audio(audio_path)

clean = np.asarray(clean, dtype=float)

print(f"      Original sampling rate: {original_fs}")
print(f"      Samples: {len(clean)}")


# ------------------------------------------------------------
# 3. Preprocess to 16 kHz
# ------------------------------------------------------------

print("\n[3/5] Preprocessing audio...")

from data_io.audio_loader import preprocess_real_audio

processed = preprocess_real_audio(
    clean,
    original_fs,
    target_sampling_rate=SAMPLING_RATE,
)

clean = np.asarray(processed, dtype=float)

print(f"      Sampling rate: {SAMPLING_RATE}")
print(f"      Samples: {len(clean)}")


# ------------------------------------------------------------
# 4. Add controlled noise
# ------------------------------------------------------------

print("\n[4/5] Creating controlled noisy signals...")

gaussian = add_gaussian_noise(
    clean,
    snr_db=TARGET_SNR_DB,
    random_state=42,
)

impulse = add_impulse_noise(
    clean,
    probability=IMPULSE_PROBABILITY,
    amplitude_factor=IMPULSE_AMPLITUDE_FACTOR,
    random_state=42,
)

print(f"      Gaussian target SNR: {TARGET_SNR_DB:.1f} dB")
print(f"      Impulse probability: {IMPULSE_PROBABILITY}")
print(f"      Impulse amplitude factor: {IMPULSE_AMPLITUDE_FACTOR}")


# ------------------------------------------------------------
# 5. Load model and evaluate
# ------------------------------------------------------------

print("\n[5/5] Running classifier...")

training_result = load_noise_model(MODEL_PATH)

model = training_result.model

print("      Model loaded successfully.")
print(f"      Features: {len(training_result.feature_names)}")


results = []

results.append(
    analyze_case(
        "Gaussian",
        clean,
        gaussian,
        SAMPLING_RATE,
        model,
    )
)

results.append(
    analyze_case(
        "Impulse",
        clean,
        impulse,
        SAMPLING_RATE,
        model,
    )
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("REAL-AUDIO VALIDATION SUMMARY")
print("=" * 60)

print(
    f"{'Case':<12}"
    f"{'Actual SNR':<15}"
    f"{'Prediction':<15}"
    f"{'Confidence':<12}"
)

print("-" * 60)

for result in results:
    print(
        f"{result['name']:<12}"
        f"{result['actual_snr']:<15.2f}"
        f"{result['prediction']:<15}"
        f"{result['confidence']:<12.4f}"
    )

print("\nValidation completed.")