from typing import Any, List, Optional

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
