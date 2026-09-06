import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from intelligence.noise.dataset import prepare_noise_dataset
from intelligence.noise.dataset_builder import build_noise_dataset
from intelligence.noise.model_io import load_noise_model


MODEL_PATH = PROJECT_ROOT / "models" / "noise_classifier.joblib"


print("=" * 60)
print("Spandhan - Fresh Noise Model Validation")
print("=" * 60)


# ------------------------------------------------------------
# 1. Build fresh synthetic dataset
# ------------------------------------------------------------

print("\n[1/4] Building fresh synthetic dataset...")

dataset = build_noise_dataset(
    samples_per_class=30,
    length=2000,
    sampling_rate=1000,
)

# Replace +inf SNR for Clean samples with the finite sentinel (100.0)
# so the StandardScaler inside the model pipeline can handle the data.
dataset = prepare_noise_dataset(dataset)

X = dataset.X
y = dataset.y

print(f"      Samples:  {len(X)}")
print(f"      Features: {X.shape[1]}")
print(f"      Classes:  {list(np.unique(y))}")


# ------------------------------------------------------------
# 2. Validate feature contract
# ------------------------------------------------------------

print("\n[2/4] Validating feature contract...")

training_result = load_noise_model(MODEL_PATH)

print(f"      Dataset features: {len(dataset.feature_names)}")
print(f"      Model features:   {len(training_result.feature_names)}")

if dataset.feature_names != training_result.feature_names:
    raise ValueError(
        "Feature mismatch between dataset and trained model.\n"
        f"Dataset: {dataset.feature_names}\n"
        f"Model:   {training_result.feature_names}"
    )

print("      Feature order: OK")


# ------------------------------------------------------------
# 3. Evaluate fresh dataset
# ------------------------------------------------------------

print("\n[3/4] Evaluating fresh unseen synthetic data...")

model = training_result.model

y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)

precision = precision_score(
    y,
    y_pred,
    average="macro",
    zero_division=0,
)

recall = recall_score(
    y,
    y_pred,
    average="macro",
    zero_division=0,
)

f1 = f1_score(
    y,
    y_pred,
    average="macro",
    zero_division=0,
)


# ------------------------------------------------------------
# 4. Report
# ------------------------------------------------------------

print("\n[4/4] Results")

print("\n" + "-" * 60)
print("FRESH SYNTHETIC VALIDATION RESULTS")
print("-" * 60)

print(f"Accuracy:   {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision:  {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall:     {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score:   {f1:.4f} ({f1 * 100:.2f}%)")


print("\nClassification Report:")

class_names = sorted(np.unique(y).tolist())

print(
    classification_report(
        y,
        y_pred,
        labels=class_names,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
)


cm = confusion_matrix(
    y,
    y_pred,
    labels=class_names,
)

print("Confusion Matrix:")
print("Classes:", ", ".join(class_names))

for class_name, row in zip(class_names, cm):
    print(
        f"{class_name:<10}: "
        + " ".join(f"{value:4d}" for value in row)
    )


print("\n" + "=" * 60)
print("Fresh validation completed successfully.")
print("=" * 60)