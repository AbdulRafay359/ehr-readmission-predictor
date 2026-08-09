from app.core.model import ReadmissionModel
from app.core.features import extract_features
from app.db.mongodb import get_db

model = ReadmissionModel()

class PredictionService:
    @staticmethod
    def predict_single(patient_id):
        db = get_db()
        record = db.diabetes_patients_combined.find_one({"encounter_id": int(patient_id)})
        if not record:
            raise ValueError(f"Patient {patient_id} not found")
        features = extract_features(record)
        risk = model.predict(features)
        return {
            "patient_id": patient_id,
            "readmission_risk": risk,
            "risk_level": "High" if risk > 0.7 else "Medium" if risk > 0.4 else "Low"
        }

    @staticmethod
    def predict_batch(patient_ids):
        db = get_db()
        records = list(db.diabetes_patients_combined.find({"encounter_id": {"$in": [int(pid) for pid in patient_ids]}}))
        results = []
        for rec in records:
            feat = extract_features(rec)
            risk = model.predict(feat)
            results.append({
                "patient_id": rec["encounter_id"],
                "readmission_risk": risk,
                "risk_level": "High" if risk > 0.7 else "Medium" if risk > 0.4 else "Low"
            })
        return results