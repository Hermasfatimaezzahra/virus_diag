"""app/api.py : API pour applications externes (sans dépendance externe)."""
from utils.preprocessing import PatientData


class ClinicalAPI:
    def __init__(self, predictor):
        self._predictor = predictor

    def handle_request(self, request):
        """request = {"symptoms": [...], "temperature": float, "test_results": {...}}"""
        try:
            patient = PatientData(request["symptoms"], request["temperature"],
                                  request.get("test_results", {}))
            return {"status": "ok", "diagnosis": self._predictor.diagnose(patient)}
        except (KeyError, TypeError) as e:
            return {"status": "error", "message": f"Requête invalide : {e}"}
