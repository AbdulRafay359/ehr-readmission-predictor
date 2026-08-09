# app/services/prediction_service.py
from app.core.model import ReadmissionModel
from app.core.features import extract_features
from app.db.mongodb import get_db
import logging
import json
import os
from app.config.settings import settings

logger = logging.getLogger(__name__)
model = ReadmissionModel()

# Load calibrated thresholds
def load_thresholds():
    try:
        with open("models/risk_thresholds.json", "r") as f:
            thresholds = json.load(f)
            return thresholds["risk_levels"]
    except:
        # Fallback thresholds
        return {
            "very_low": {"threshold": 0.0, "label": "Very Low Risk"},
            "low": {"threshold": 0.15, "label": "Low Risk"},
            "medium": {"threshold": 0.3, "label": "Medium Risk"},
            "high": {"threshold": 0.55, "label": "High Risk"},
            "very_high": {"threshold": 0.75, "label": "Very High Risk"}
        }

RISK_LEVELS = load_thresholds()

class PredictionService:
    
    @staticmethod
    def get_risk_level(risk_score):
        """Determine risk level based on calibrated thresholds"""
        risk_level = "very_low"
        for level, config in RISK_LEVELS.items():
            if risk_score >= config["threshold"]:
                risk_level = level
        
        return RISK_LEVELS[risk_level]["label"]
    
    @staticmethod
    def predict_single(patient_id):
        """
        Predict readmission risk for a single patient
        """
        db = get_db()
        
        try:
            patient_id_int = int(patient_id)
        except ValueError:
            raise ValueError(f"Invalid patient ID format: {patient_id}")
        
        record = db.diabetes_patients_combined.find_one({"encounter_id": patient_id_int})
        
        if not record:
            raise ValueError(f"Patient with ID {patient_id} not found")
        
        try:
            features = extract_features(record)
            risk = model.predict(features)
            
            risk_level = PredictionService.get_risk_level(risk)
            
            return {
                "patient_id": str(patient_id),
                "readmission_risk": risk,
                "risk_level": risk_level,
                "model_version": "2.0.0"
            }
        except Exception as e:
            logger.error(f"Error predicting for patient {patient_id}: {str(e)}")
            raise ValueError(f"Prediction failed for patient {patient_id}: {str(e)}")

    @staticmethod
    def predict_batch(patient_ids):
        """
        Predict readmission risk for multiple patients
        """
        db = get_db()
        results = []
        errors = []
        
        for pid in patient_ids:
            try:
                patient_id_int = int(pid)
                record = db.diabetes_patients_combined.find_one({"encounter_id": patient_id_int})
                
                if not record:
                    errors.append(f"Patient {pid} not found")
                    continue
                
                features = extract_features(record)
                risk = model.predict(features)
                
                risk_level = PredictionService.get_risk_level(risk)
                
                results.append({
                    "patient_id": str(pid),
                    "readmission_risk": risk,
                    "risk_level": risk_level,
                    "model_version": "2.0.0"
                })
                
            except Exception as e:
                logger.error(f"Error predicting for patient {pid}: {str(e)}")
                errors.append(f"Patient {pid}: {str(e)}")
        
        if len(results) == 0 and len(errors) > 0:
            raise ValueError(f"All predictions failed: {'; '.join(errors)}")
        
        if len(errors) > 0:
            logger.warning(f"Some predictions failed: {'; '.join(errors)}")
        
        return results