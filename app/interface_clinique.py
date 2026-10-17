"""app/interface_clinique.py : interface pour médecins (console)."""
from utils.preprocessing import PatientData


class InterfaceClinique:
    def __init__(self, predictor):
        self._predictor = predictor

    def run(self):
        print("=== Diagnostic viral - interface clinique ===")
        symptoms = [s.strip() for s in
                    input("Symptômes (fièvre, toux, fatigue) séparés par des virgules : ").split(",")
                    if s.strip()]
        temperature = float(input("Température (°C) : "))
        pcr = int(input("Test PCR (1 = positif, 0 = négatif) : "))
        patient = PatientData(symptoms, temperature, {"PCR": pcr})
        print("Diagnostic :", self._predictor.diagnose(patient))
