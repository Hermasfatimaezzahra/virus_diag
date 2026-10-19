"""main.py (à la racine) : teste tout le pipeline.
données -> trainer -> evaluator (metrics) -> predictor (seuil >= 0.5)
"""
from core.dataset import Dataset
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from pipeline.predictor import ClinicalPredictor
from utils.preprocessing import PatientData


def main():
    # 1. Données : lecture, découpage 80 % / 20 %, normalisation
    data = Dataset.from_csv("data/patient_data.csv")
    train_set, test_set = data.split()
    train_set, scaler = train_set.standardize()
    test_set, _ = test_set.standardize(scaler)

    # 2. Entraînement avec trainer.py + 3. Évaluation avec evaluator.py (metrics.py)
    models = [LogisticRegression(), NeuralNetwork()]
    for model in models:
        Trainer(model).train(train_set)
        scores = Evaluator(model).evaluate(test_set)
        print(f"{type(model).__name__:20s}", {k: round(v, 3) for k, v in scores.items()})

    # 4. Prédiction avec predictor.py : probabilité >= 0.5 -> "Infecté", sinon "Sain"
    predictor = ClinicalPredictor(models[0], scaler)
    patients = {
        "Patient 1 (fièvre, toux, 39.2, PCR+)": PatientData(["fièvre", "toux"], 39.2, {"PCR": 1}),
        "Patient 2 (aucun symptôme, 36.8, PCR-)": PatientData([], 36.8, {"PCR": 0}),
    }
    for nom, patient in patients.items():
        proba = predictor.probability(patient)
        print(f"{nom} -> probabilité = {proba:.2f} -> {predictor.diagnose(patient)}")


if __name__ == "__main__":
    main()
