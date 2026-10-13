
import numpy as np

from perceptron import Perceptron


def test_and_gate():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)

    y = np.array([0, 0, 0, 1])

    model = Perceptron(epochs=100).fit(X, y)

    assert np.array_equal(model.predict(X), y)


def test_or_gate():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)

    y = np.array([0, 1, 1, 1])

    model = Perceptron(epochs=100).fit(X, y)

    assert np.array_equal(model.predict(X), y)
