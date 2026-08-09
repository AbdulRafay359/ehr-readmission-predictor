from pydantic import BaseModel
from typing import List, Optional

class PatientIdRequest(BaseModel):
    patient_id: str

class BatchRequest(BaseModel):
    patient_ids: List[str]

class PredictionResponse(BaseModel):
    patient_id: str
    readmission_risk: float
    risk_level: str
    model_version: str = "1.0.0"