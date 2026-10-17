"""pipeline/trainer.py : entraîne le modèle (boucle d'apprentissage)."""


class Trainer:
    def __init__(self, model):
        self._model = model

    def train(self, dataset):
        self._model.train(dataset.X, dataset.y)
        return self._model
