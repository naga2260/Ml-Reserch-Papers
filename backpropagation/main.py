
import numpy as np

from network import (
    forward_propagation,
    backward_propagation,
    update_parameters,
)
from losses import mse_loss
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# 1. XOR DATASET
# ==========================================

X_train = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

Y_train = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
])


# ==========================================
# 2. INITIALIZE PARAMETERS
# ==========================================

W1 = np.array([
    [0.2, 0.4],
    [0.3, 0.1],
    [0.5, 0.2],
])

b1 = np.array([
    [0.1],
    [0.1],
    [0.1],
])

W2 = np.array([
    [0.7, 0.2, 0.5],
])

b2 = np.array([
    [0.1],
])


# ==========================================
# 3. HYPERPARAMETERS
# ==========================================

epochs = 10000
learning_rate = 3.0

loss_history = []


# ==========================================
# 4. TRAINING LOOP
# ==========================================

for epoch in range(epochs):

    # Train on each example individually
    for x, y in zip(X_train, Y_train):

        X = x.reshape(-1, 1)
        Y = y.reshape(-1, 1)

        # Forward propagation
        Z1, A1, Z2, A2 = forward_propagation(
            X, W1, b1, W2, b2
        )

        # Backpropagation
        dW1, db1, dW2, db2 = backward_propagation(
            X, Y, A1, A2, W2
        )

        # Update parameters
        W1, b1, W2, b2 = update_parameters(
            W1, b1, W2, b2,
            dW1, db1, dW2, db2,
            learning_rate
        )

    # ======================================
    # 5. CALCULATE EPOCH LOSS
    # ======================================

    total_loss = 0.0

    for x, y in zip(X_train, Y_train):

        X = x.reshape(-1, 1)
        Y = y.reshape(-1, 1)

        _, _, _, A2 = forward_propagation(
            X, W1, b1, W2, b2
        )

        total_loss += mse_loss(Y, A2)

    average_loss = total_loss / len(X_train)
    loss_history.append(average_loss)

    if (epoch + 1) % 1000 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Loss: {average_loss:.6f}"
        )


# ==========================================
# 6. FINAL XOR PREDICTIONS
# ==========================================

print("\n========== FINAL XOR PREDICTIONS ==========")

for x, y in zip(X_train, Y_train):

    X = x.reshape(-1, 1)

    _, _, _, A2 = forward_propagation(
        X, W1, b1, W2, b2
    )

    print(
        f"Input: {x} "
        f"| Target: {y[0]:.0f} "
        f"| Prediction: {A2.item():.6f}"
    )


# ==========================================
# 7. FINAL PARAMETERS
# ==========================================

print("\n========== FINAL PARAMETERS ==========")

print("\nW1:\n", W1)
print("\nb1:\n", b1)

print("\nW2:\n", W2)
print("\nb2:\n", b2)

print("\nFinal Loss:", loss_history[-1])



# ==========================================
# 8. PLOT LOSS CURVE
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, epochs + 1),
    loss_history,
    label="Training Loss"
)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("XOR Neural Network - Training Loss")

# Log scale makes small loss changes easier to see
plt.yscale("log")

plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot
Path("results").mkdir(exist_ok=True)

plt.savefig(
    "results/loss_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nLoss curve saved to results/loss_curve.png")
