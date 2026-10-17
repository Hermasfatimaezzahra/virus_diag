"""pipeline/predictor.py : EXERCICE p.147 - ClinicalPredictor

Séparé du Trainer : il ne peut que diagnostiquer jamais ré-entraîner
Fonctionne avec N'IMPORTE QUEL modèle ayant une méthode predict() (polymorphisme)
"""
from utils.preprocessing import PatientData


class ClinicalPredictor:
    def __init__(self, model, scaler=None):
        self._model = model      # modèle déjà entraîné
        self._scaler = scaler    # même normalisation que pendant l'entraînement

    def diagnose(self, patient_data):
        x = patient_data.to_vector() if isinstance(patient_data, PatientData) else patient_data
        if self._scaler is not None:
            x = self._scaler.transform(x)
        prediction = self._model.predict(x)
        return "Infecté" if prediction >= 0.5 else "Sain"
