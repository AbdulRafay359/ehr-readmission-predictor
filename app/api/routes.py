from fastapi import APIRouter, HTTPException, status
from app.api.schemas import PatientIdRequest, BatchRequest, PredictionResponse
from app.services.prediction_service import PredictionService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PatientIdRequest):
    """
    Predict readmission risk for a single patient
    """
    try:
        result = PredictionService.predict_single(request.patient_id)
        return PredictionResponse(**result)
    except ValueError as e:
        logger.warning(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(request: BatchRequest):
    """
    Predict readmission risk for multiple patients
    """
    try:
        results = PredictionService.predict_batch(request.patient_ids)
        return results
    except ValueError as e:
        logger.warning(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in batch prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )