import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from intelligence.noise.dataset import split_noise_dataset
from intelligence.noise.dataset_builder import build_noise_dataset
from intelligence.noise.evaluator import evaluate_svm
from intelligence.noise.model_io import save_noise_model
from intelligence.noise.trainer import train_svm


MODEL_PATH = PROJECT_ROOT / "models" / "noise_classifier.joblib"

def main() -> None:
    print("=" * 60)
    print("Spandhan - Noise Classification Model Training")
    print("=" * 60)

    print("\n[1/5] Building synthetic noise dataset...")

    dataset = build_noise_dataset(
        samples_per_class=100,
        length=2000,
        sampling_rate=1000,
        seed=42,
    )

    print(f"      Samples:  {len(dataset.X)}")
    print(f"      Features: {dataset.X.shape[1]}")
    print(f"      Classes:  {sorted(set(dataset.y))}")

    print("\n[2/5] Preparing and splitting dataset...")

    split = split_noise_dataset(
        dataset,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
    )

    print(f"      Training:   {len(split.X_train)}")
    print(f"      Validation: {len(split.X_val)}")
    print(f"      Test:       {len(split.X_test)}")

    print("\n[3/5] Training calibrated RBF-SVM (with GridSearchCV tuning)...")

    training_result = train_svm(
        split,
        random_state=42,
    )

    print(
        f"      Best C:              "
        f"{training_result.best_params.get('C', 'n/a')}"
    )
    print(
        f"      Best gamma:          "
        f"{training_result.best_params.get('gamma', 'n/a')}"
    )

    print(
        f"      Training accuracy:   "
        f"{training_result.training_accuracy:.4f}"
    )

    print(
        f"      Validation accuracy: "
        f"{training_result.validation_accuracy:.4f}"
    )

    print("\n[4/5] Evaluating on untouched test set...")

    evaluation = evaluate_svm(
        training_result,
        split,
    )

    print(
        f"      Test accuracy:       "
        f"{evaluation.accuracy:.4f}"
    )

    print(
        f"      Test precision:      "
        f"{evaluation.precision_macro:.4f}"
    )

    print(
        f"      Test recall:         "
        f"{evaluation.recall_macro:.4f}"
    )

    print(
        f"      Test F1:             "
        f"{evaluation.f1_macro:.4f}"
    )

    print("\n      Confusion matrix:")

    print(
        "      Classes: "
        + ", ".join(evaluation.class_names)
    )

    for class_name, row in zip(
        evaluation.class_names,
        evaluation.confusion_matrix,
    ):
        print(
            f"      {class_name:10s}: "
            + " ".join(f"{value:4d}" for value in row)
        )

    print("\n[5/5] Saving trained model...")

    saved_path = save_noise_model(
        training_result,
        MODEL_PATH,
    )

    print(f"      Saved: {saved_path}")

    print("\n" + "=" * 60)
    print("Training completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()