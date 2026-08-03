"""POST /api/v1/chat -- safety-scoped chat assistant endpoint (bearer-token protected)."""

from fastapi import APIRouter, Depends

from app.core.security import verify_service_token
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_engine import run_chat

router = APIRouter(dependencies=[Depends(verify_service_token)])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await run_chat(request)
