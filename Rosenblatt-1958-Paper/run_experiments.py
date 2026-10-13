
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import Perceptron as SklearnPerceptron

from perceptron import Perceptron


RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)

DATASETS = {
    "AND": np.array([0, 0, 0, 1]),
    "OR": np.array([0, 1, 1, 1]),
    "XOR": np.array([0, 1, 1, 0]),
}


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def save_csv(filename, rows):
    path = RESULTS / filename
    if not rows:
        return

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {path}")


def experiment_a():
    """Measure learning-rate sensitivity on the AND dataset."""
    rows = []

    for lr in [0.01, 0.1, 1.0]:
        model = Perceptron(learning_rate=lr, epochs=100)
        model.fit(X, DATASETS["AND"])

        predictions = model.predict(X)
        errors = model.errors_per_epoch

        row = {
            "learning_rate": lr,
            "accuracy": accuracy(DATASETS["AND"], predictions),
            "epochs_run": len(errors),
            "final_epoch_errors": errors[-1],
            "total_updates": sum(errors),
            "converged": errors[-1] == 0,
            "weights": json.dumps(model.weights.tolist()),
            "bias": model.bias,
        }
        rows.append(row)
        print("A:", row)

    save_csv("learning_rate.csv", rows)
    return rows


def experiment_b():
    """Compare linearly separable and non-separable datasets."""
    rows = []

    for name, y in DATASETS.items():
        model = Perceptron(learning_rate=0.1, epochs=100)
        model.fit(X, y)

        predictions = model.predict(X)
        errors = model.errors_per_epoch

        row = {
            "dataset": name,
            "accuracy": accuracy(y, predictions),
            "epochs_run": len(errors),
            "final_epoch_errors": errors[-1],
            "total_updates": sum(errors),
            "converged": errors[-1] == 0,
            "predictions": json.dumps(predictions.tolist()),
            "actual": json.dumps(y.tolist()),
        }
        rows.append(row)
        print("B:", row)

    save_csv("separability.csv", rows)
    return rows


def experiment_c():
    """Compare predictions with scikit-learn."""
    rows = []

    for name, y in DATASETS.items():
        custom = Perceptron(
            learning_rate=0.1,
            epochs=100,
        ).fit(X, y)

        # Fixed example order: avoid random shuffling.
        reference = SklearnPerceptron(
            eta0=0.1,
            max_iter=100,
            tol=None,
            shuffle=False,
            random_state=42,
        )
        reference.fit(X, y)

        custom_predictions = custom.predict(X)
        sklearn_predictions = reference.predict(X)

        row = {
            "dataset": name,
            "custom_accuracy": accuracy(y, custom_predictions),
            "sklearn_accuracy": accuracy(y, sklearn_predictions),
            "custom_predictions": json.dumps(
                custom_predictions.tolist()
            ),
            "sklearn_predictions": json.dumps(
                sklearn_predictions.tolist()
            ),
            "predictions_match": bool(
                np.array_equal(
                    custom_predictions, sklearn_predictions
                )
            ),
            "custom_weights": json.dumps(
                custom.weights.tolist()
            ),
            "sklearn_weights": json.dumps(
                reference.coef_[0].tolist()
            ),
        }
        rows.append(row)
        print("C:", row)

    save_csv("library_comparison.csv", rows)
    return rows


def main():
    log = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "dataset": "Four Boolean input combinations",
        "settings": {
            "learning_rates": [0.01, 0.1, 1.0],
            "epochs": 100,
            "initial_weights": "zeros",
            "custom_model": "perceptron.py",
        },
        "experiment_a": experiment_a(),
        "experiment_b": experiment_b(),
        "experiment_c": experiment_c(),
    }

    with (RESULTS / "experiment_log.json").open("w") as file:
        json.dump(log, file, indent=2)

    print("\nAll experiments completed.")
    print(f"Results saved to: {RESULTS.resolve()}")


if __name__ == "__main__":
    main()
