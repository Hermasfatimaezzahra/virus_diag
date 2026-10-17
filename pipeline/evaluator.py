"""pipeline/evaluator.py : mesure les performances (accuracy, F1...)"""
from utils import metrics


class Evaluator:
    def __init__(self, model, threshold=0.5):
        self._model = model
        self._threshold = threshold

    def evaluate(self, dataset):
        y_pred = (self._model.predict(dataset.X) >= self._threshold).astype(float)
        return {
            "accuracy": metrics.accuracy(dataset.y, y_pred),
            "precision": metrics.precision(dataset.y, y_pred),
            "recall": metrics.recall(dataset.y, y_pred),
            "f1": metrics.f1_score(dataset.y, y_pred),
        }
