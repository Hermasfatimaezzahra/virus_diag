"""core/optimizer.py : algorithme d'optimisation."""


class GradientDescent:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def step(self, params, grads):
        """Met à jour chaque paramètre (tableaux numpy) en place."""
        for p, g in zip(params, grads):
            p -= self.learning_rate * g
