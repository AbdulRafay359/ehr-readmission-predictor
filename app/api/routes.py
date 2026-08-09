from typing import List
from fastapi import APIRouter, HTTPException
from app.api.schemas import PatientIdRequest, BatchRequest, PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PatientIdRequest):
    try:
        result = PredictionService.predict_single(request.patient_id)
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchRequest):
    try:
        results = PredictionService.predict_batch(request.patient_ids)
        return [PredictionResponse(**r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health():
    return {"status": "ok", "model_version": "1.0.0"}