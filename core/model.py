"""core/model.py : abstraction (p.141) + classe mère Model (héritage, p.139)."""
import pickle
from abc import ABC, abstractmethod


class DiagnosticModel(ABC):
    """Interface obligatoire : aucun modèle ne peut l'oublier."""

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class Model(DiagnosticModel):
    """Classe mère : save() et load() sont écrites une seule fois."""

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
