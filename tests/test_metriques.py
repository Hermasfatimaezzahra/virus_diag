"""Tests des métriques sur un exemple calculé à la main."""
import numpy as np
import pytest

from utils import metrics

y_vrai = np.array([1, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0])   # 1 vrai positif, 1 faux négatif, 1 faux positif, 1 vrai négatif


def test_accuracy():
    assert metrics.accuracy(y_vrai, y_pred) == 0.5


def test_precision_rappel_f1():
    assert metrics.precision(y_vrai, y_pred) == 0.5
    assert metrics.recall(y_vrai, y_pred) == 0.5
    assert metrics.f1_score(y_vrai, y_pred) == pytest.approx(0.5)


def test_aucun_positif_predit_ne_divise_pas_par_zero():
    assert metrics.precision(y_vrai, np.zeros(4)) == 0.0
    assert metrics.f1_score(y_vrai, np.zeros(4)) == 0.0
