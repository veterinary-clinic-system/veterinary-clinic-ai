"""POST /api/v1/triage -- AI pre-screening endpoint (bearer-token protected)."""

from fastapi import APIRouter, Depends

from app.core.security import verify_service_token
from app.schemas.triage import TriageRequest, TriageResponse
from app.services.triage_engine import run_triage

router = APIRouter(dependencies=[Depends(verify_service_token)])


@router.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest) -> TriageResponse:
    return await run_triage(request)
