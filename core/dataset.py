"""core/dataset.py : classe Dataset interne."""
import csv
import numpy as np
from utils.preprocessing import FEATURES, Standardizer, clean


class Dataset:
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)

    @classmethod
    def from_csv(cls, path):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # en-tête
            rows = clean(list(reader))
        data = np.array(rows, dtype=float)
        return cls(data[:, :len(FEATURES)], data[:, len(FEATURES)])

    def split(self, test_ratio=0.2, seed=0):
        idx = np.random.default_rng(seed).permutation(len(self.X))
        n_test = int(len(idx) * test_ratio)
        test, train = idx[:n_test], idx[n_test:]
        return Dataset(self.X[train], self.y[train]), Dataset(self.X[test], self.y[test])

    def standardize(self, scaler=None):
        scaler = scaler or Standardizer().fit(self.X)
        return Dataset(scaler.transform(self.X), self.y), scaler

    def __len__(self):
        return len(self.X)
