"""core/logistic_regression.py : hérite de Model."""
import numpy as np
from core.model import Model
from core.optimizer import GradientDescent


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression(Model):
    def __init__(self, learning_rate=0.1, epochs=500):
        self._weights = None      # _ = interdit d'accès direct (encapsulation, p.138)
        self._bias = np.zeros(1)
        self._optimizer = GradientDescent(learning_rate)
        self._epochs = epochs

    def train(self, X, y):
        self._weights = np.zeros(X.shape[1])
        for _ in range(self._epochs):
            error = _sigmoid(X @ self._weights + self._bias[0]) - y
            grad_w = X.T @ error / len(y)
            grad_b = np.array([error.mean()])
            self._optimizer.step([self._weights, self._bias], [grad_w, grad_b])

    def predict(self, X):
        """Probabilité d'être infecté (float si un seul patient)."""
        proba = _sigmoid(np.asarray(X) @ self._weights + self._bias[0])
        return float(proba) if np.ndim(proba) == 0 else proba
