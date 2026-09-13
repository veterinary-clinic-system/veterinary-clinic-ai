import json
import logging
import os
from typing import Any, Dict, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def get_db_connection():
    """Create a new connection to PostgreSQL using .env variables."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        user=os.getenv("DB_USERNAME", "vetclinic"),
        password=os.getenv("DB_PASSWORD", "vetclinic"),
        dbname=os.getenv("DB_DATABASE", "veterinary_clinic"),
    )


def init_db() -> None:
    """Initialize additional tables such as diagnosis_logs if they do not already exist."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnosis_logs (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    input_data JSONB,
                    data_02_compiled JSONB,
                    data_03_output JSONB,
                    ground_truth_diseases JSONB,
                    weight_adjustments JSONB
                );
                """
            )

            # Migrate legacy single-disease column (pre "diseases" array) if present.
            cur.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'diagnosis_logs'
                  AND column_name = 'ground_truth_disease';
                """
            )
            if cur.fetchone():
                cur.execute(
                    """
                    ALTER TABLE diagnosis_logs
                    ALTER COLUMN ground_truth_disease TYPE JSONB
                    USING CASE
                        WHEN ground_truth_disease IS NULL THEN NULL
                        ELSE to_jsonb(ARRAY[ground_truth_disease])
                    END;
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE diagnosis_logs
                    RENAME COLUMN ground_truth_disease TO ground_truth_diseases;
                    """
                )
                logger.info(
                    "Migrated diagnosis_logs.ground_truth_disease -> "
                    "ground_truth_diseases (JSONB array)."
                )

            conn.commit()
            logger.info("Database initialized successfully (diagnosis_logs table ready).")
    finally:
        conn.close()


def fetch_symptoms() -> List[Dict[str, Any]]:
    """Query all symptoms from the PostgreSQL symptoms table."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT symptom_id, symptom_name, describe, common_symptom
                FROM symptoms
                ORDER BY symptom_id;
                """
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch_diseases() -> List[Dict[str, Any]]:
    """Query all diseases from the PostgreSQL diseases table."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT disease_id, disease_name, describe
                FROM diseases
                ORDER BY disease_id;
                """
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    finally:
        conn.close()


def save_diagnosis_log(
    input_data: Dict[str, Any],
    data_02_compiled: Dict[str, Any],
    data_03_output: Dict[str, Any],
    ground_truth_diseases: Optional[List[str]] = None,
    weight_adjustments: Optional[Any] = None,
) -> int:
    """Record diagnosis interaction and learning adjustments in database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO diagnosis_logs (
                    input_data,
                    data_02_compiled,
                    data_03_output,
                    ground_truth_diseases,
                    weight_adjustments
                ) VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    json.dumps(input_data, ensure_ascii=False),
                    json.dumps(data_02_compiled, ensure_ascii=False),
                    json.dumps(data_03_output, ensure_ascii=False),
                    json.dumps(ground_truth_diseases, ensure_ascii=False) if ground_truth_diseases else None,
                    json.dumps(weight_adjustments, ensure_ascii=False) if weight_adjustments else None,
                ),
            )
            inserted_id = cur.fetchone()[0]
            conn.commit()
            return inserted_id
    finally:
        conn.close()
