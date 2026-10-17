"""utils/preprocessing.py : nettoyage et préparation des données (supporte l'IA, ne contient pas l'IA)."""
import numpy as np

FEATURES = ["temperature", "fievre", "toux", "fatigue", "pcr"]


class PatientData:
    """Un patient (même classe que p.137)."""

    def __init__(self, symptoms, temperature, test_results):
        self.symptoms = symptoms
        self.temperature = temperature
        self.test_results = test_results

    def to_vector(self):
        return np.array([
            self.temperature,
            1.0 if "fièvre" in self.symptoms else 0.0,
            1.0 if "toux" in self.symptoms else 0.0,
            1.0 if "fatigue" in self.symptoms else 0.0,
            float(self.test_results.get("PCR", 0)),
        ])


class Standardizer:
    """Normalise les variables (moyenne 0, écart-type 1)."""

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean) / self.std


def clean(rows):
    """Supprime les lignes incomplètes."""
    return [r for r in rows if all(v != "" for v in r)]
