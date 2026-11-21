"""core/knn.py : k plus proches voisins, hérite de Model."""
import numpy as np
from core.model import Model


class KNN(Model):
    """Un patient ressemble à ses k voisins les plus proches dans les données d'entraînement."""

    def __init__(self, k=5):
        self._k = k
        self._X = None   # KNN n'apprend pas de poids : il mémorise les patients connus
        self._y = None

    def train(self, X, y):
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y, dtype=float)

    def predict(self, X):
        """Probabilité d'être infecté = part de patients infectés parmi les k voisins."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        distances = np.linalg.norm(X[:, None, :] - self._X[None, :, :], axis=2)
        voisins = np.argsort(distances, axis=1)[:, :self._k]
        proba = self._y[voisins].mean(axis=1)
        return float(proba[0]) if len(proba) == 1 else proba
