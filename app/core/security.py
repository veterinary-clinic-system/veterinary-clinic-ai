"""Bearer-token authentication for the internal service-to-service API.

The NestJS backend is the only expected caller. It must send:

    Authorization: Bearer <AI_SERVICE_TOKEN>

`verify_service_token` is a FastAPI dependency raising 401 whenever the
header is missing, malformed, or does not match the configured secret.
It is applied to the triage and chat routers only -- `/health` stays
unauthenticated for container/uptime probes.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings

# auto_error=False so we can control the status code (401, not FastAPI's
# default 403) and error payload ourselves for both the "missing header"
# and "wrong token" cases.
_bearer_scheme = HTTPBearer(auto_error=False)


async def verify_service_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> None:
    """Raise 401 unless a valid `Bearer <AI_SERVICE_TOKEN>` header is present."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.scheme.lower() != "bearer" or credentials.credentials != settings.AI_SERVICE_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
