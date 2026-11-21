"""main.py (Ã  la racine) : teste tout le pipeline.
donnÃ©es -> trainer -> evaluator (metrics) -> predictor (seuil >= 0.5)
"""
from core.dataset import Dataset
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from core.knn import KNN
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator
from pipeline.predictor import ClinicalPredictor
from utils.preprocessing import PatientData


def main():
    # 1. DonnÃ©es : lecture, dÃ©coupage 80 % / 20 %, normalisation
    data = Dataset.from_csv("data/patient_data.csv")
    train_set, test_set = data.split()
    train_set, scaler = train_set.standardize()
    test_set, _ = test_set.standardize(scaler)

    # 2. EntraÃ®nement avec trainer.py + 3. Ã‰valuation avec evaluator.py (metrics.py)
    models = [LogisticRegression(), NeuralNetwork(), KNN()]
    for model in models:
        Trainer(model).train(train_set)
        scores = Evaluator(model).evaluate(test_set)
        print(f"{type(model).__name__:20s}", {k: round(v, 3) for k, v in scores.items()})

    # 4. PrÃ©diction avec predictor.py : probabilitÃ© >= 0.5 -> "InfectÃ©", sinon "Sain"
    predictor = ClinicalPredictor(models[0], scaler)
    patients = {
        "Patient 1 (fiÃ¨vre, toux, 39.2, PCR+)": PatientData(["fiÃ¨vre", "toux"], 39.2, {"PCR": 1}),
        "Patient 2 (aucun symptÃ´me, 36.8, PCR-)": PatientData([], 36.8, {"PCR": 0}),
    }
    for nom, patient in patients.items():
        proba = predictor.probability(patient)
        print(f"{nom} -> probabilitÃ© = {proba:.2f} -> {predictor.diagnose(patient)}")


if __name__ == "__main__":
    main()
