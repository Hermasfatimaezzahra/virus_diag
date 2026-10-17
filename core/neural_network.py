"""core/neural_network.py : réseau à 1 couche cachée, hérite de Model."""
import numpy as np
from core.model import Model
from core.optimizer import GradientDescent
from core.logistic_regression import _sigmoid


class NeuralNetwork(Model):
    def __init__(self, hidden=8, learning_rate=0.1, epochs=800, seed=0):
        self._hidden = hidden
        self._optimizer = GradientDescent(learning_rate)
        self._epochs = epochs
        self._rng = np.random.default_rng(seed)
        self._W1 = self._b1 = self._W2 = self._b2 = None

    def train(self, X, y):
        n = X.shape[1]
        self._W1 = self._rng.normal(0, 0.5, (n, self._hidden))
        self._b1 = np.zeros(self._hidden)
        self._W2 = self._rng.normal(0, 0.5, self._hidden)
        self._b2 = np.zeros(1)
        for _ in range(self._epochs):
            h = np.tanh(X @ self._W1 + self._b1)
            out = _sigmoid(h @ self._W2 + self._b2[0])
            d_out = (out - y) / len(y)
            g_W2 = h.T @ d_out
            g_b2 = np.array([d_out.sum()])
            d_h = np.outer(d_out, self._W2) * (1 - h ** 2)
            g_W1 = X.T @ d_h
            g_b1 = d_h.sum(axis=0)
            self._optimizer.step([self._W1, self._b1, self._W2, self._b2],
                                 [g_W1, g_b1, g_W2, g_b2])

    def predict(self, X):
        h = np.tanh(np.asarray(X) @ self._W1 + self._b1)
        proba = _sigmoid(h @ self._W2 + self._b2[0])
        return float(proba) if np.ndim(proba) == 0 else proba
