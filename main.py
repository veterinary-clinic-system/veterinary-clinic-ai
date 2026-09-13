import logging
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)