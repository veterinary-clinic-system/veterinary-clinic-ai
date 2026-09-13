import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Body, HTTPException, status

from src.db import save_diagnosis_log
from src.extractor import DeepSeekSymptomExtractor
from src.matrix_engine import MatrixEngine
from src.schemas import (
    CompiledExtractionOutput,
    DiagnosisRequest,
    DiagnosisResponse,
    DiseasePrediction,
)

logger = logging.getLogger(__name__)

router = APIRouter()

extractor = DeepSeekSymptomExtractor()
matrix_engine = MatrixEngine()

MAX_IMAGE_COUNT = 5
MAX_VIDEO_COUNT = 2

async def _process_diagnosis_pipeline(
    request_data: DiagnosisRequest,
    image_files: Optional[List[Tuple[str, bytes]]] = None,
    video_files: Optional[List[Tuple[str, bytes]]] = None,
) -> DiagnosisResponse:
    """Run the diagnosis pipeline."""

    compiled_output = await extractor.process_inputs(
        pet_info=request_data.pet_info,
        initial_symptoms=request_data.symptoms,
        description=request_data.describe,
        images=request_data.images,
        videos=request_data.videos,
        image_files=image_files,
        video_files=video_files,
    )

    data_02 = compiled_output.model_dump(by_alias=True)

    predicted_diseases = matrix_engine.predict(
        pet_info=data_02["pet-info"],
        symptoms=data_02["symptoms"],
        top_k=5,
    )

    weight_adjustments = None

    if request_data.diseases:
        logger.info(
            "Updating model weights for diseases: %s",
            request_data.diseases,
        )

        weight_adjustments = matrix_engine.readjust_weights_batch(
            target_disease_ids=request_data.diseases,
            symptoms=data_02["symptoms"],
            pet_info=data_02["pet-info"],
        )

    response = DiagnosisResponse(
        diseases=[DiseasePrediction(**item) for item in predicted_diseases],
        triage_result=compiled_output.triage_result,
    )

    try:
        save_diagnosis_log(
            input_data=request_data.model_dump(by_alias=True),
            data_02_compiled=data_02,
            data_03_output=response.model_dump(),
            ground_truth_diseases=request_data.diseases,
            weight_adjustments=weight_adjustments,
        )
    except Exception as exc:
        logger.error("Failed to save diagnosis log: %s", exc)

    return response


@router.post(
    "/api/v1/diagnose",
    response_model=DiagnosisResponse,
    summary="Diagnose pet disease",
)
async def diagnose_endpoint(
    request_data: DiagnosisRequest = Body(...),
) -> DiagnosisResponse:
    if len(request_data.images or []) > MAX_IMAGE_COUNT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {MAX_IMAGE_COUNT} images are allowed.",
        )

    if len(request_data.videos or []) > MAX_VIDEO_COUNT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {MAX_VIDEO_COUNT} videos are allowed.",
        )

    try:
        return await _process_diagnosis_pipeline(request_data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Diagnosis processing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagnosis processing failed: {exc}",
        )


@router.get(
    "/api/v1/matrix",
    summary="Get symptom-disease matrix",
)
async def get_matrix_endpoint() -> Dict[str, Any]:
    return matrix_engine.matrix_data


@router.post(
    "/api/v1/matrix/predict",
    response_model=DiagnosisResponse,
    summary="Predict disease from matrix",
)
async def predict_matrix_from_data_02(
    compiled_data: CompiledExtractionOutput = Body(...),
) -> DiagnosisResponse:
    data_02 = compiled_data.model_dump(by_alias=True)

    predicted_diseases = matrix_engine.predict(
        pet_info=data_02["pet-info"],
        symptoms=data_02["symptoms"],
        top_k=5,
    )

    return DiagnosisResponse(
        diseases=[DiseasePrediction(**item) for item in predicted_diseases],
        triage_result=compiled_data.triage_result,
    )


@router.post(
    "/api/v1/extract-symptoms",
    response_model=DiagnosisResponse,
)
async def extract_symptoms_alias(
    request_data: DiagnosisRequest = Body(...),
) -> DiagnosisResponse:
    return await diagnose_endpoint(request_data)

