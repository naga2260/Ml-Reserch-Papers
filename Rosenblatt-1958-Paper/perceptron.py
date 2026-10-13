
import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0
        self.errors_per_epoch = []

    def predict_one(self, x):
        z = np.dot(x, self.weights) + self.bias
        return int(z >= 0)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        self.errors_per_epoch = []

        for _ in range(self.epochs):
            errors = 0

            for x, target in zip(X, y):
                prediction = self.predict_one(x)
                update = self.learning_rate * (
                    target - prediction
                )

                if update != 0:
                    self.weights += update * x
                    self.bias += update
                    errors += 1

            self.errors_per_epoch.append(errors)

            if errors == 0:
                break

        return self
