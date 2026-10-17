"""main.py : assemble tout le pipeline (données -> entraînement -> évaluation -> diagnostic)."""
import sys
from core.dataset import Dataset
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from pipeline.predictor import ClinicalPredictor
from utils.preprocessing import PatientData
from app.api import ClinicalAPI
from app.interface_clinique import InterfaceClinique


def main():
    data = Dataset.from_csv("data/patient_data.csv")
    train_set, test_set = data.split()
    train_set, scaler = train_set.standardize()
    test_set, _ = test_set.standardize(scaler)

    # Polymorphisme : même code pour tous les modèles
    models = [LogisticRegression(), NeuralNetwork()]
    for m in models:
        Trainer(m).train(train_set)
        scores = Evaluator(m).evaluate(test_set)
        print(f"{type(m).__name__:20s}", {k: round(v, 3) for k, v in scores.items()})

    best = models[0]
    best.save("model.pkl")
    predictor = ClinicalPredictor(type(best).load("model.pkl"), scaler)

    malade = PatientData(["fièvre", "toux", "fatigue"], 39.2, {"PCR": 1})
    sain = PatientData([], 36.8, {"PCR": 0})
    print("Patient 1 ->", predictor.diagnose(malade))
    print("Patient 2 ->", predictor.diagnose(sain))
    print("API       ->", ClinicalAPI(predictor).handle_request(
        {"symptoms": ["toux"], "temperature": 38.5, "test_results": {"PCR": 1}}))

    if "--interactive" in sys.argv:
        InterfaceClinique(predictor).run()


if __name__ == "__main__":
    main()
