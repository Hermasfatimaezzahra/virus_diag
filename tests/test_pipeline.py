"""Tests de bout en bout : données réelles -> entraînement -> évaluation -> diagnostic -> API."""
import pytest

from app.api import ClinicalAPI
from core.dataset import Dataset
from core.logistic_regression import LogisticRegression
from pipeline.evaluator import Evaluator
from pipeline.predictor import ClinicalPredictor
from pipeline.trainer import Trainer
from utils.preprocessing import PatientData, clean


@pytest.fixture(scope="module")
def predicteur():
    donnees = Dataset.from_csv("data/patient_data.csv")
    entrainement, test = donnees.split()
    entrainement, scaler = entrainement.standardize()
    test, _ = test.standardize(scaler)
    modele = Trainer(LogisticRegression()).train(entrainement)
    return ClinicalPredictor(modele, scaler), Evaluator(modele).evaluate(test)


def test_donnees_chargees_et_decoupees():
    donnees = Dataset.from_csv("data/patient_data.csv")
    entrainement, test = donnees.split(test_ratio=0.2)
    assert len(donnees) == 300
    assert len(test) == 60 and len(entrainement) == 240


def test_lignes_incompletes_retirees():
    assert clean([["1", "2"], ["", "3"]]) == [["1", "2"]]


def test_modele_assez_precis(predicteur):
    _, scores = predicteur
    assert scores["accuracy"] >= 0.9


def test_diagnostic_patient_malade_et_patient_sain(predicteur):
    predictor, _ = predicteur
    assert predictor.diagnose(PatientData(["fièvre", "toux"], 39.2, {"PCR": 1})) == "Infecté"
    assert predictor.diagnose(PatientData([], 36.8, {"PCR": 0})) == "Sain"


def test_api_repond_et_refuse_une_requete_invalide(predicteur):
    api = ClinicalAPI(predicteur[0])
    ok = api.handle_request({"symptoms": ["fièvre"], "temperature": 39.5, "test_results": {"PCR": 1}})
    assert ok == {"status": "ok", "diagnosis": "Infecté"}
    assert api.handle_request({"symptoms": []})["status"] == "error"
