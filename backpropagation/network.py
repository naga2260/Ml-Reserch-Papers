import numpy as np

from activations import sigmoid


def forward_propagation(
    X: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform forward propagation through a 2-3-1 neural network.
    """

    # Hidden layer
    Z1 = W1 @ X + b1
    A1 = sigmoid(Z1)

    # Output layer
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    return Z1, A1, Z2, A2



def backward_propagation(
    X: np.ndarray,
    Y: np.ndarray,
    A1: np.ndarray,
    A2: np.ndarray,
    W2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate gradients using backpropagation."""

    # Output layer error
    delta2 = (A2 - Y) * A2 * (1 - A2)

    # Output layer gradients
    dW2 = delta2 @ A1.T
    db2 = delta2

    # Hidden layer error
    delta1 = (W2.T @ delta2) * A1 * (1 - A1)

    # Hidden layer gradients
    dW1 = delta1 @ X.T
    db1 = delta1

    return dW1, db1, dW2, db2



def update_parameters(
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    dW1: np.ndarray,
    db1: np.ndarray,
    dW2: np.ndarray,
    db2: np.ndarray,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update network parameters using gradient descent."""

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return W1, b1, W2, b2
