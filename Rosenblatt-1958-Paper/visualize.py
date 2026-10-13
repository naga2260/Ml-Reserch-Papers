
import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=float)

y = np.array([0, 0, 0, 1])

model = Perceptron(learning_rate=0.1, epochs=100)
model.fit(X, y)

for label, marker in [(0, "o"), (1, "x")]:
    points = X[y == label]
    plt.scatter(
        points[:, 0],
        points[:, 1],
        marker=marker,
        s=120,
        label=f"Class {label}",
    )

w1, w2 = model.weights
b = model.bias

if abs(w2) > 1e-12:
    x_line = np.linspace(-0.2, 1.2, 100)
    y_line = -(w1 * x_line + b) / w2
    plt.plot(x_line, y_line, label="Decision boundary")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron: AND classification")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/decision_boundary.png", dpi=150)
plt.show()

plt.figure()
plt.plot(
    range(1, len(model.errors_per_epoch) + 1),
    model.errors_per_epoch,
    marker="o",
)
plt.xlabel("Epoch")
plt.ylabel("Number of mistakes")
plt.title("Perceptron learning curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/learning_curve.png", dpi=150)
plt.show()
