
import numpy as np

from losses import mse_loss
from network import forward_propagation, backward_propagation, update_parameters


def train(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    epochs: int = 10000,
    learning_rate: float = 0.5,
):
    history = []

    for epoch in range(epochs):

        # Train on each example
        for x, y in zip(X_train, Y_train):
            X = x.reshape(-1, 1)
            Y = y.reshape(-1, 1)

            # Forward propagation
            _, A1, _, A2 = forward_propagation(
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

        # Evaluate loss across the entire dataset
        total_loss = 0.0

        for x, y in zip(X_train, Y_train):
            X = x.reshape(-1, 1)
            Y = y.reshape(-1, 1)

            _, _, _, A2 = forward_propagation(
                X, W1, b1, W2, b2
            )

            total_loss += mse_loss(Y, A2)

        average_loss = total_loss / len(X_train)
        history.append(average_loss)

        if (epoch + 1) % 1000 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"| Loss: {average_loss:.6f}"
            )

    return W1, b1, W2, b2, history
