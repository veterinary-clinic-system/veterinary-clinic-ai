import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import APIRouter, Body, HTTPException, Path, status
from fastapi.responses import FileResponse

from src.db import (
    create_disease,
    create_symptom,
    delete_disease,
    delete_symptom,
    fetch_disease,
    fetch_diseases,
    fetch_symptom,
    fetch_symptoms,
    save_diagnosis_log,
    update_disease,
    update_symptom,
)
from src.extractor import DeepSeekSymptomExtractor
from src.matrix_engine import MatrixEngine
from src.schemas import (
    CompiledExtractionOutput,
    DiagnosisDetailResponse,
    DiagnosisRequest,
    DiagnosisResponse,
    Disease,
    DiseaseCreate,
    DiseasePrediction,
    DiseasePredictionDetail,
    DiseaseUpdate,
    MatrixCellBatchUpdate,
    MatrixCellUpdate,
    MatrixTable,
    Symptom,
    SymptomCreate,
    SymptomUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter()

matrix_engine = MatrixEngine()

try:
    extractor: Optional[DeepSeekSymptomExtractor] = DeepSeekSymptomExtractor()
except Exception as exc:  # pragma: no cover - missing/invalid OpenRouter config
    # The catalog and matrix endpoints stay usable without an LLM key; only
    # /diagnose needs the extractor, and it reports the problem at call time.
    logger.error("Could not initialise the symptom extractor: %s", exc)
    extractor = None

MAX_IMAGE_COUNT = 5
MAX_VIDEO_COUNT = 2


def _require_extractor() -> DeepSeekSymptomExtractor:
    if extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Chưa cấu hình được OpenRouter (kiểm tra OPENROUTER_API_KEY "
                "trong file .env)."
            ),
        )
    return extractor


def _db(action: Callable[[], Any]) -> Any:
    """Run a database call, turning connection errors into a 503."""
    try:
        return action()
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Database operation failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Không kết nối được cơ sở dữ liệu: {exc}",
        )


def _invalidate_symptom_cache() -> None:
    """Drop the extractor's cached symptom catalog after a catalog change."""
    if extractor is not None:
        extractor._symptoms_cache = None


async def _process_diagnosis_pipeline(
    request_data: DiagnosisRequest,
    image_files: Optional[List[Tuple[str, bytes]]] = None,
    video_files: Optional[List[Tuple[str, bytes]]] = None,
) -> Tuple[DiagnosisResponse, Dict[str, Any], Optional[List[Dict[str, Any]]]]:
    """Run the diagnosis pipeline.

    Returns the `data-03` response plus the intermediate `data-02` payload and
    any weight adjustments, so callers can expose the full audit trail.
    """

    compiled_output = await _require_extractor().process_inputs(
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

    return response, data_02, weight_adjustments


def _validate_media_counts(request_data: DiagnosisRequest) -> None:
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


@router.post(
    "/api/v1/diagnose",
    response_model=DiagnosisResponse,
    summary="Diagnose pet disease",
)
async def diagnose_endpoint(
    request_data: DiagnosisRequest = Body(...),
) -> DiagnosisResponse:
    _validate_media_counts(request_data)

    try:
        response, _, _ = await _process_diagnosis_pipeline(request_data)
        return response
    except HTTPException:
        raise
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


@router.post(
    "/api/v1/diagnose/detailed",
    response_model=DiagnosisDetailResponse,
    summary="Diagnose pet disease (with the full audit trail)",
)
async def diagnose_detailed_endpoint(
    request_data: DiagnosisRequest = Body(...),
) -> DiagnosisDetailResponse:
    """Same pipeline as `/api/v1/diagnose`, plus what the dashboard shows.

    The extra fields are the compiled `data-02` payload (normalised pet info,
    the symptoms the model actually extracted and the triage reasoning), the
    disease names resolved from the catalog, and any online-learning weight
    adjustments applied to the matrix.
    """
    _validate_media_counts(request_data)

    try:
        response, compiled, adjustments = await _process_diagnosis_pipeline(
            request_data,
        )
    except HTTPException:
        raise
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

    # Names are a nicety: a catalog outage must not fail a good prediction.
    names: Dict[str, str] = {}
    try:
        names = {
            row["disease_id"]: row["disease_name"] for row in fetch_diseases()
        }
    except Exception as exc:
        logger.warning("Could not resolve disease names: %s", exc)

    return DiagnosisDetailResponse(
        diseases=[
            DiseasePredictionDetail(
                disease=item.disease,
                prevalence_rate=item.prevalence_rate,
                disease_name=names.get(item.disease),
            )
            for item in response.diseases
        ],
        triage_result=response.triage_result,
        compiled=compiled,
        weight_adjustments=adjustments,
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


# ---------------------------------------------------------------------------
# Symptom catalog
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/symptoms",
    response_model=List[Symptom],
    summary="List symptoms",
    tags=["catalog"],
)
async def list_symptoms() -> List[Symptom]:
    rows = _db(fetch_symptoms)
    return [Symptom(**row) for row in rows]


@router.post(
    "/api/v1/symptoms",
    response_model=Symptom,
    status_code=status.HTTP_201_CREATED,
    summary="Create a symptom (also adds its matrix column)",
    tags=["catalog"],
)
async def create_symptom_endpoint(
    payload: SymptomCreate = Body(...),
) -> Symptom:
    row = _db(
        lambda: create_symptom(
            symptom_name=payload.symptom_name,
            describe=payload.describe,
            common_symptom=payload.common_symptom,
            symptom_id=payload.symptom_id,
        )
    )

    matrix_engine.add_symptom_column(row["symptom_id"])
    _invalidate_symptom_cache()

    return Symptom(**row)


@router.put(
    "/api/v1/symptoms/{symptom_id}",
    response_model=Symptom,
    summary="Update a symptom",
    tags=["catalog"],
)
async def update_symptom_endpoint(
    symptom_id: str = Path(...),
    payload: SymptomUpdate = Body(...),
) -> Symptom:
    row = _db(
        lambda: update_symptom(
            symptom_id=symptom_id,
            symptom_name=payload.symptom_name,
            describe=payload.describe,
            common_symptom=payload.common_symptom,
        )
    )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy triệu chứng '{symptom_id}'.",
        )

    _invalidate_symptom_cache()
    return Symptom(**row)


@router.delete(
    "/api/v1/symptoms/{symptom_id}",
    summary="Delete a symptom (also drops its matrix column)",
    tags=["catalog"],
)
async def delete_symptom_endpoint(
    symptom_id: str = Path(...),
) -> Dict[str, Any]:
    if not _db(lambda: delete_symptom(symptom_id)):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy triệu chứng '{symptom_id}'.",
        )

    removed_column = matrix_engine.remove_symptom_column(symptom_id)
    _invalidate_symptom_cache()

    return {
        "symptom_id": symptom_id,
        "deleted": True,
        "matrix_column_removed": removed_column,
    }


# ---------------------------------------------------------------------------
# Disease catalog
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/diseases",
    response_model=List[Disease],
    summary="List diseases",
    tags=["catalog"],
)
async def list_diseases() -> List[Disease]:
    rows = _db(fetch_diseases)
    return [Disease(**row) for row in rows]


@router.post(
    "/api/v1/diseases",
    response_model=Disease,
    status_code=status.HTTP_201_CREATED,
    summary="Create a disease (also adds its matrix row)",
    tags=["catalog"],
)
async def create_disease_endpoint(
    payload: DiseaseCreate = Body(...),
) -> Disease:
    row = _db(
        lambda: create_disease(
            disease_name=payload.disease_name,
            describe=payload.describe,
            disease_id=payload.disease_id,
        )
    )

    matrix_engine.add_disease_row(row["disease_id"])

    return Disease(**row)


@router.put(
    "/api/v1/diseases/{disease_id}",
    response_model=Disease,
    summary="Update a disease",
    tags=["catalog"],
)
async def update_disease_endpoint(
    disease_id: str = Path(...),
    payload: DiseaseUpdate = Body(...),
) -> Disease:
    row = _db(
        lambda: update_disease(
            disease_id=disease_id,
            disease_name=payload.disease_name,
            describe=payload.describe,
        )
    )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy bệnh '{disease_id}'.",
        )

    return Disease(**row)


@router.delete(
    "/api/v1/diseases/{disease_id}",
    summary="Delete a disease (also drops its matrix row)",
    tags=["catalog"],
)
async def delete_disease_endpoint(
    disease_id: str = Path(...),
) -> Dict[str, Any]:
    if not _db(lambda: delete_disease(disease_id)):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy bệnh '{disease_id}'.",
        )

    removed_row = matrix_engine.remove_disease_row(disease_id)

    return {
        "disease_id": disease_id,
        "deleted": True,
        "matrix_row_removed": removed_row,
    }


# ---------------------------------------------------------------------------
# Matrix editor
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/matrix/table",
    response_model=MatrixTable,
    summary="Get the matrix as a dense table (same layout as the CSV)",
    tags=["matrix"],
)
async def get_matrix_table() -> MatrixTable:
    return MatrixTable(**matrix_engine.get_table())


@router.put(
    "/api/v1/matrix/cell",
    summary="Update one matrix cell",
    tags=["matrix"],
)
async def update_matrix_cell(
    payload: MatrixCellUpdate = Body(...),
) -> Dict[str, Any]:
    try:
        return matrix_engine.set_cell(
            disease_id=payload.disease,
            column=payload.column,
            value=payload.value,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc).strip("'\""),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@router.put(
    "/api/v1/matrix/cells",
    summary="Update several matrix cells at once",
    tags=["matrix"],
)
async def update_matrix_cells(
    payload: MatrixCellBatchUpdate = Body(...),
) -> Dict[str, Any]:
    try:
        results = matrix_engine.set_cells(
            [item.model_dump() for item in payload.updates]
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc).strip("'\""),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return {"updated": len(results), "cells": results}


@router.post(
    "/api/v1/matrix/sync",
    summary="Add matrix rows/columns for catalog entries that are missing",
    tags=["matrix"],
)
async def sync_matrix_with_catalog() -> Dict[str, Any]:
    symptom_ids = [row["symptom_id"] for row in _db(fetch_symptoms)]
    disease_ids = [row["disease_id"] for row in _db(fetch_diseases)]

    added = matrix_engine.sync_with_catalog(
        symptom_ids=symptom_ids,
        disease_ids=disease_ids,
    )

    return {"added": added}


@router.get(
    "/api/v1/matrix/export",
    summary="Download the matrix CSV as it is stored on disk",
    tags=["matrix"],
    response_class=FileResponse,
)
async def export_matrix() -> FileResponse:
    """Serve the CSV file itself rather than re-serialising the in-memory copy.

    Every edit goes through `_save_matrix`, so the file on disk is current and
    the download is byte-for-byte what the engine reads back.
    """
    path = os.path.abspath(matrix_engine.csv_path)

    if not os.path.exists(path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy file ma trận: {path}",
        )

    return FileResponse(
        path,
        media_type="text/csv; charset=utf-8",
        filename=os.path.basename(path),
    )


@router.post(
    "/api/v1/matrix/reload",
    summary="Re-read the matrix CSV from disk",
    tags=["matrix"],
)
async def reload_matrix() -> Dict[str, Any]:
    matrix_engine.reload()
    table = matrix_engine.get_table()
    return {
        "reloaded": True,
        "diseases": len(table["rows"]),
        "columns": len(table["columns"]),
    }


# ---------------------------------------------------------------------------
# Meta / health
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/meta",
    summary="Reference codes used across the pipeline",
    tags=["meta"],
)
async def get_meta() -> Dict[str, Any]:
    return {
        "breeds": [
            {"code": "BR001", "label": "Chó"},
            {"code": "BR002", "label": "Mèo"},
            {"code": "BR003", "label": "Thỏ"},
            {"code": "BR004", "label": "Hamster"},
        ],
        "genders": [
            {"code": "M", "label": "Đực"},
            {"code": "F", "label": "Cái"},
        ],
        "weights": [
            {"code": "W01", "label": "Nhẹ cân"},
            {"code": "W02", "label": "Bình thường"},
            {"code": "W03", "label": "Thừa cân"},
        ],
        "ages": [
            {"code": "A01", "label": "Nhỏ (< 1 tuổi)"},
            {"code": "A02", "label": "Trưởng thành (1-7 tuổi)"},
            {"code": "A03", "label": "Già (> 7 tuổi)"},
        ],
        "triage_colors": [
            {"code": "RED", "label": "Tối khẩn"},
            {"code": "ORANGE", "label": "Khẩn cấp"},
            {"code": "YELLOW", "label": "Bán khẩn"},
            {"code": "GREEN", "label": "Ổn định"},
            {"code": "BLUE", "label": "Không khẩn cấp"},
        ],
        "limits": {
            "max_images": MAX_IMAGE_COUNT,
            "max_videos": MAX_VIDEO_COUNT,
        },
    }


@router.get(
    "/api/v1/health",
    summary="Service health (database, LLM, matrix)",
    tags=["meta"],
)
async def health() -> Dict[str, Any]:
    database: Dict[str, Any] = {"ok": True}
    try:
        fetch_symptom("SY001")
    except Exception as exc:
        database = {"ok": False, "error": str(exc)}

    return {
        "status": "ok",
        "database": database,
        "extractor": {
            "ok": extractor is not None,
            "model": getattr(extractor, "model_name", None),
        },
        "matrix": {
            "diseases": len(matrix_engine.matrix_data.get("diseases", {})),
            "symptom_columns": len(matrix_engine.symptom_cols),
            "csv_path": matrix_engine.csv_path,
        },
    }


@router.get(
    "/api/v1/diseases/{disease_id}",
    response_model=Disease,
    summary="Get a single disease",
    tags=["catalog"],
)
async def get_disease_endpoint(disease_id: str = Path(...)) -> Disease:
    row = _db(lambda: fetch_disease(disease_id))
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy bệnh '{disease_id}'.",
        )
    return Disease(**row)


@router.get(
    "/api/v1/symptoms/{symptom_id}",
    response_model=Symptom,
    summary="Get a single symptom",
    tags=["catalog"],
)
async def get_symptom_endpoint(symptom_id: str = Path(...)) -> Symptom:
    row = _db(lambda: fetch_symptom(symptom_id))
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy triệu chứng '{symptom_id}'.",
        )
    return Symptom(**row)
