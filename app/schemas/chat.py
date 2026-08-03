"""Request/response models for POST /api/v1/chat."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

ChatRole = Literal["user", "assistant"]


class ChatMessage(BaseModel):
    role: ChatRole
    content: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Existing session id, or null to start a new one.")
    message: str = Field(description="Latest user message, Vietnamese.")
    history: Optional[List[ChatMessage]] = Field(default=None, description="Prior turns, oldest first.")


class ChatResponse(BaseModel):
    reply: str
    suggest_booking: bool
    session_id: str
