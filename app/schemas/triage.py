"""Request/response models for POST /api/v1/triage."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

PriorityColor = Literal["RED", "ORANGE", "YELLOW", "GREEN", "BLUE"]


class TriageRequest(BaseModel):
    symptom_text: str = Field(
        default="", description="Vietnamese free text describing the pet's symptoms; may be empty."
    )
    photo_urls: List[str] = Field(
        default_factory=list, description="Publicly reachable image URLs of the pet/injury; may be empty."
    )
    pet_species: Optional[str] = Field(default=None, description="Optional context, e.g. 'dog', 'cat'.")
    pet_age_months: Optional[float] = Field(default=None, description="Optional context, pet age in months.")


class DiseaseGroupScore(BaseModel):
    name: str
    confidence: float = Field(ge=0.0, le=1.0)


class TriageResponse(BaseModel):
    priority_color: PriorityColor
    suspected_disease_groups: List[DiseaseGroupScore]
    extracted_symptom_keywords: List[str]
    nlp_confidence: float = Field(ge=0.0, le=1.0)
    cv_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    overall_confidence: float = Field(ge=0.0, le=1.0)
