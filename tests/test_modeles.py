"""Tests des modèles : chacun apprend et renvoie une probabilité entre 0 et 1."""
import numpy as np
import pytest

from core.knn import KNN
from core.logistic_regression import LogisticRegression
from core.model import Model
from core.neural_network import NeuralNetwork

X = np.array([[0.0, 0.0], [0.1, 0.2], [0.2, 0.1], [3.0, 3.0], [3.1, 2.9], [2.9, 3.2]])
y = np.array([0, 0, 0, 1, 1, 1], dtype=float)


@pytest.mark.parametrize("modele", [LogisticRegression(), NeuralNetwork(), KNN(k=3)])
def test_modele_separe_les_deux_groupes(modele):
    modele.train(X, y)
    assert modele.predict([0.0, 0.1]) < 0.5
    assert modele.predict([3.0, 3.1]) >= 0.5


@pytest.mark.parametrize("modele", [LogisticRegression(), NeuralNetwork(), KNN(k=3)])
def test_probabilites_entre_0_et_1(modele):
    modele.train(X, y)
    proba = modele.predict(X)
    assert np.all((proba >= 0) & (proba <= 1))


def test_knn_refuse_k_negatif_ou_nul():
    with pytest.raises(ValueError):
        KNN(k=0)


def test_sauvegarde_puis_chargement(tmp_path):
    modele = LogisticRegression()
    modele.train(X, y)
    chemin = tmp_path / "modele.pkl"
    modele.save(chemin)
    assert Model.load(chemin).predict([3.0, 3.0]) == pytest.approx(modele.predict([3.0, 3.0]))
