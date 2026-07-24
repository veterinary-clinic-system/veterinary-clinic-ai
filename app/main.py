"""FastAPI application entrypoint.

Run locally with:

    uvicorn app.main:app --reload --port 8000

This is an internal, server-to-server API called only by the NestJS
backend over the private network / docker network -- no browser calls it
directly, so CORS is intentionally NOT enabled.
"""

from fastapi import FastAPI

from app.api import chat, health, triage

app = FastAPI(
    title="veterinary-clinic-ai",
    description="AI pre-screening/triage + chat microservice for the veterinary clinic system.",
    version="0.1.0",
)

# /health has no auth (docker healthchecks / uptime probes).
app.include_router(health.router)

# /api/v1/triage and /api/v1/chat both require the AI_SERVICE_TOKEN bearer
# token, enforced by app.core.security.verify_service_token on each router.
app.include_router(triage.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
