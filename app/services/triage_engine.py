"""Combines the NLP and CV triage signals into the final API response.

Orchestration:
  1. Run the NLP step (always -- rule-based, optionally LLM-upgraded).
  2. Run the CV step (best-effort, only when photo_urls is non-empty; a
     blocking call, so it's off-loaded to a worker thread so it never
     blocks the event loop).
  3. `final_severity = max(nlp_severity, cv_severity or 0)` -- triage is
     worst-case driven: a single alarming signal from either channel
     should win, never get averaged away.
  4. Disease groups from both channels are merged, deduplicated by name
     (keeping the higher confidence on a collision), sorted by confidence
     descending, and capped at 5 entries.
  5. `overall_confidence` is a weighted combination of the two channels
     when both are present (NLP weighted higher since it is always
     "in-domain" Vietnamese text, whereas CV is a coarse, English-label
     zero-shot classifier): `0.6 * nlp_confidence + 0.4 * cv_confidence`.
     When CV was skipped, `overall_confidence == nlp_confidence`.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from app.schemas.triage import DiseaseGroupScore, TriageRequest, TriageResponse
from app.services import cv_triage, nlp_triage
from app.services.cv_triage import CvResult
from app.services.nlp_triage import SEVERITY_TO_COLOR, DiseaseGroupMatch, NlpResult

# Weights for combining NLP + CV confidence into overall_confidence when
# both channels produced a result. See module docstring point 5.
_NLP_WEIGHT = 0.6
_CV_WEIGHT = 0.4


def _merge_disease_groups(
    nlp_groups: List[DiseaseGroupMatch], cv_result: Optional[CvResult]
) -> List[DiseaseGroupScore]:
    merged: Dict[str, float] = {}
    for group in nlp_groups:
        merged[group.name] = max(merged.get(group.name, 0.0), group.confidence)
    if cv_result is not None:
        merged[cv_result.disease_group] = max(merged.get(cv_result.disease_group, 0.0), cv_result.confidence)

    scored = [DiseaseGroupScore(name=name, confidence=round(confidence, 4)) for name, confidence in merged.items()]
    scored.sort(key=lambda g: g.confidence, reverse=True)
    return scored[:5]


async def run_triage(request: TriageRequest) -> TriageResponse:
    nlp_result: NlpResult = await nlp_triage.extract(request.symptom_text)

    cv_result: Optional[CvResult] = None
    if request.photo_urls:
        # classify_photos() is a synchronous, potentially slow call
        # (model load + HTTP downloads + inference); run it off the event
        # loop so one request's CV step can't stall the whole service.
        cv_result = await asyncio.to_thread(cv_triage.classify_photos, request.photo_urls)

    final_severity = max(nlp_result.severity, cv_result.severity if cv_result else 0)
    priority_color = SEVERITY_TO_COLOR[final_severity]

    suspected_disease_groups = _merge_disease_groups(nlp_result.disease_groups, cv_result)

    if cv_result is not None:
        overall_confidence = _NLP_WEIGHT * nlp_result.confidence + _CV_WEIGHT * cv_result.confidence
    else:
        overall_confidence = nlp_result.confidence
    overall_confidence = max(0.0, min(1.0, overall_confidence))

    return TriageResponse(
        priority_color=priority_color,  # type: ignore[arg-type]
        suspected_disease_groups=suspected_disease_groups,
        extracted_symptom_keywords=nlp_result.matched_keywords,
        nlp_confidence=nlp_result.confidence,
        cv_confidence=cv_result.confidence if cv_result is not None else None,
        overall_confidence=overall_confidence,
    )
