import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from src.db import init_db
from src.matrix_engine import MatrixEngine
from src.router import router

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

    logger.info("Initializing symptom-disease matrix...")
    try:
        MatrixEngine()
    except Exception as e:
        logger.error(f"Error initializing matrix: {e}")

    yield
    logger.info("Shutting down application...")


app = FastAPI(
    title="Pet Disease Diagnosis & Triage API",
    description="API dự đoán bệnh thú cưng và phân loại cấp cứu - DeepSeek V4.1-Flash (OpenRouter) & Symptom-Disease Matrix",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)