import numpy as np


def mse_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Calculate mean squared error with a 1/2 factor."""
    return float(0.5 * np.sum((y_true - y_pred) ** 2))