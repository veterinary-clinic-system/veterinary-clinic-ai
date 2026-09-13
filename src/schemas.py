from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PetInfoInput(BaseModel):
    breed: str
    specie: Optional[str] = None
    gender: str
    weight: float
    age: float


class DiagnosisRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pet_info: PetInfoInput = Field(alias="pet-info")
    symptoms: List[str] = Field(default_factory=list)
    describe: Optional[str] = None
    diseases: List[str] = Field(
        default_factory=list,
        description=(
            "Optional ground-truth disease codes for this case. When "
            "provided, the matrix engine learns from them and adjusts "
            "its weights accordingly."
        ),
    )
    images: List[str] = Field(default_factory=list)
    videos: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_diseases(cls, data: Any) -> Any:
        if isinstance(data, dict) and "diseases" not in data and "disease" in data:
            value = data["disease"]
            data["diseases"] = value if isinstance(value, list) else [value]
        return data

    @model_validator(mode="before")
    @classmethod
    def normalize_media(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "image" in data and "images" not in data:
                value = data["image"]
                data["images"] = value if isinstance(value, list) else [value]

            if "video" in data and "videos" not in data:
                value = data["video"]
                data["videos"] = value if isinstance(value, list) else [value]

        return data


class PetInfoNormalized(BaseModel):
    breed: str
    gender: str
    weight: str
    age: str


class SymptomIntensity(BaseModel):
    symptom: str
    intensity: int = Field(ge=1, le=3)


class TriageResult(BaseModel):
    color_code: str
    reasoning: str


class CompiledExtractionOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    pet_info: PetInfoNormalized = Field(alias="pet-info")
    symptoms: List[SymptomIntensity]
    triage_result: TriageResult


class DiseasePrediction(BaseModel):
    disease: str
    prevalence_rate: float


class DiagnosisResponse(BaseModel):
    diseases: List[DiseasePrediction]
    triage_result: TriageResult


# ---------------------------------------------------------------------------
# Catalog (symptoms / diseases) - dashboard CRUD
# ---------------------------------------------------------------------------


class SymptomBase(BaseModel):
    symptom_name: str = Field(min_length=1, max_length=255)
    describe: Optional[str] = None
    common_symptom: bool = False


class SymptomCreate(SymptomBase):
    symptom_id: Optional[str] = Field(
        default=None,
        max_length=10,
        description="Bỏ trống để hệ thống tự sinh mã SYnnn tiếp theo.",
    )


class SymptomUpdate(SymptomBase):
    pass


class Symptom(SymptomBase):
    symptom_id: str


class DiseaseBase(BaseModel):
    disease_name: str = Field(min_length=1, max_length=255)
    describe: Optional[str] = None


class DiseaseCreate(DiseaseBase):
    disease_id: Optional[str] = Field(
        default=None,
        max_length=10,
        description="Bỏ trống để hệ thống tự sinh mã DInnn tiếp theo.",
    )


class DiseaseUpdate(DiseaseBase):
    pass


class Disease(DiseaseBase):
    disease_id: str


# ---------------------------------------------------------------------------
# Matrix editor
# ---------------------------------------------------------------------------


class MatrixColumn(BaseModel):
    key: str
    group: str
    kind: str


class MatrixRow(BaseModel):
    disease: str
    values: Dict[str, int]


class MatrixValueRange(BaseModel):
    min: int
    max: int


class MatrixTable(BaseModel):
    columns: List[MatrixColumn]
    rows: List[MatrixRow]
    id_column: str
    value_range: MatrixValueRange
    csv_path: str


class MatrixCellUpdate(BaseModel):
    disease: str = Field(min_length=1)
    column: str = Field(min_length=1)
    value: int


class MatrixCellBatchUpdate(BaseModel):
    updates: List[MatrixCellUpdate] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Detailed diagnosis response (dashboard) - data-03 plus the audit trail
# ---------------------------------------------------------------------------


class DiseasePredictionDetail(DiseasePrediction):
    disease_name: Optional[str] = None


class DiagnosisDetailResponse(BaseModel):
    diseases: List[DiseasePredictionDetail]
    triage_result: TriageResult
    compiled: Dict[str, Any]
    weight_adjustments: Optional[List[Dict[str, Any]]] = None
