"""pipeline/predictor.py : EXERCICE p.147 - ClinicalPredictor.

Séparé du Trainer : il ne peut que diagnostiquer, jamais ré-entraîner.
Fonctionne avec N'IMPORTE QUEL modèle ayant une méthode predict() (polymorphisme).
"""
from utils.preprocessing import PatientData


class ClinicalPredictor:
    def __init__(self, model, scaler=None, threshold=0.5):
        self._model = model          # modèle déjà entraîné
        self._scaler = scaler        # même normalisation que pendant l'entraînement
        self._threshold = threshold  # seuil de décision : >= 0.5 -> "Infecté"

    def probability(self, patient_data):
        """Probabilité d'être infecté (nombre entre 0 et 1)."""
        x = patient_data.to_vector() if isinstance(patient_data, PatientData) else patient_data
        if self._scaler is not None:
            x = self._scaler.transform(x)
        return self._model.predict(x)

    def diagnose(self, patient_data):
        prediction = self.probability(patient_data)
        return "Infecté" if prediction >= self._threshold else "Sain"
