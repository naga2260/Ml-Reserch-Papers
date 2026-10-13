
import numpy as np
from perceptron import Perceptron


def run_experiment(name, X, y):
    model = Perceptron(learning_rate=0.1, epochs=100)
    model.fit(X, y)

    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)

    print(f"\n{name}")
    print("Predictions:", predictions.tolist())
    print("Actual:     ", y.tolist())
    print(f"Accuracy: {accuracy:.2%}")
    print("Weights:", model.weights)
    print("Bias:", model.bias)
    print("Errors per epoch:", model.errors_per_epoch)

    return model


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)

y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])
y_xor = np.array([0, 1, 1, 0])

run_experiment("AND", X, y_and)
run_experiment("OR", X, y_or)
run_experiment("XOR", X, y_xor)
