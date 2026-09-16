import logging
import os
import secrets
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from src.db import fetch_diseases, fetch_symptoms, init_db
from src.router import matrix_engine, router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure DB tables and Matrix Engine are ready
    logger.info("Initializing database connection and tables...")
    try:
        init_db()
    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL database: {e}")

    # The matrix engine itself is built at import time in src.router; here we
    # only reconcile it with the catalog so the dashboard's matrix grid always
    # has a row per disease and a column per symptom.
    logger.info("Syncing symptom-disease matrix with the catalog...")
    try:
        matrix_engine.sync_with_catalog(
            symptom_ids=[row["symptom_id"] for row in fetch_symptoms()],
            disease_ids=[row["disease_id"] for row in fetch_diseases()],
        )
    except Exception as e:
        logger.error(f"Error syncing matrix with catalog: {e}")

    yield
    logger.info("Shutting down application...")


app = FastAPI(
    title="Pet Disease Diagnosis & Triage API",
    description="API dự đoán bệnh thú cưng và phân loại cấp cứu - DeepSeek V4.1-Flash (OpenRouter) & Symptom-Disease Matrix",
    version="2.0.0",
    lifespan=lifespan,
)

# The dashboard is a separate Vite app, so it talks to this API cross-origin.
_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        ["*"] if _origins == "*"
        else [origin.strip() for origin in _origins.split(",") if origin.strip()]
    ),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def authenticate_api(request: Request, call_next):
    """Require the shared service token for non-public AI endpoints."""
    service_token = os.getenv("AI_SERVICE_TOKEN", "").strip()
    public_paths = {"/api/v1/health", "/api/v1/meta"}

    if (
        service_token
        and request.method != "OPTIONS"
        and request.url.path.startswith("/api/v1/")
        and request.url.path not in public_paths
    ):
        authorization = request.headers.get("Authorization", "")
        scheme, _, supplied_token = authorization.partition(" ")
        valid = (
            scheme.lower() == "bearer"
            and supplied_token
            and secrets.compare_digest(supplied_token, service_token)
        )
        if not valid:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing AI service token."},
                headers={"WWW-Authenticate": "Bearer"},
            )

    return await call_next(request)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
