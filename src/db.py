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


# ---------------------------------------------------------------------------
# Symptom / disease catalog CRUD (used by the dashboard)
# ---------------------------------------------------------------------------

_CATALOG_META = {
    "symptom": {
        "table": "symptoms",
        "id_col": "symptom_id",
        "name_col": "symptom_name",
        "prefix": "SY",
        "columns": "symptom_id, symptom_name, describe, common_symptom",
    },
    "disease": {
        "table": "diseases",
        "id_col": "disease_id",
        "name_col": "disease_name",
        "prefix": "DI",
        "columns": "disease_id, disease_name, describe",
    },
}


def _next_code(cur, meta: Dict[str, str]) -> str:
    """Return the next free code (e.g. ``SY033``) for a catalog table."""
    cur.execute(
        f"""
        SELECT {meta['id_col']} FROM {meta['table']}
        WHERE {meta['id_col']} ~ %s
        """,
        (f"^{meta['prefix']}[0-9]+$",),
    )
    # The caller's cursor is a RealDictCursor, so rows are keyed by name.
    numbers = [
        int(row[meta["id_col"]][len(meta["prefix"]):])
        for row in cur.fetchall()
    ]
    return f"{meta['prefix']}{(max(numbers) + 1) if numbers else 1:03d}"


def fetch_disease(disease_id: str) -> Optional[Dict[str, Any]]:
    """Return a single disease row, or None when it does not exist."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT disease_id, disease_name, describe
                FROM diseases WHERE disease_id = %s;
                """,
                (disease_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def fetch_symptom(symptom_id: str) -> Optional[Dict[str, Any]]:
    """Return a single symptom row, or None when it does not exist."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT symptom_id, symptom_name, describe, common_symptom
                FROM symptoms WHERE symptom_id = %s;
                """,
                (symptom_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def create_symptom(
    symptom_name: str,
    describe: Optional[str] = None,
    common_symptom: bool = False,
    symptom_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert a symptom, generating the next ``SYnnn`` code when none is given."""
    meta = _CATALOG_META["symptom"]
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            code = (symptom_id or "").strip().upper() or _next_code(cur, meta)
            cur.execute(
                "SELECT 1 FROM symptoms WHERE symptom_id = %s;", (code,)
            )
            if cur.fetchone():
                raise ValueError(f"Mã triệu chứng '{code}' đã tồn tại.")

            cur.execute(
                """
                INSERT INTO symptoms (symptom_id, symptom_name, describe, common_symptom)
                VALUES (%s, %s, %s, %s)
                RETURNING symptom_id, symptom_name, describe, common_symptom;
                """,
                (code, symptom_name.strip(), describe, bool(common_symptom)),
            )
            row = dict(cur.fetchone())
            conn.commit()
            return row
    finally:
        conn.close()


def update_symptom(
    symptom_id: str,
    symptom_name: str,
    describe: Optional[str] = None,
    common_symptom: bool = False,
) -> Optional[Dict[str, Any]]:
    """Update a symptom in place. Returns None when the code is unknown."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE symptoms
                SET symptom_name = %s, describe = %s, common_symptom = %s
                WHERE symptom_id = %s
                RETURNING symptom_id, symptom_name, describe, common_symptom;
                """,
                (symptom_name.strip(), describe, bool(common_symptom), symptom_id),
            )
            row = cur.fetchone()
            conn.commit()
            return dict(row) if row else None
    finally:
        conn.close()


def delete_symptom(symptom_id: str) -> bool:
    """Delete a symptom. Returns False when the code is unknown."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM symptoms WHERE symptom_id = %s RETURNING symptom_id;",
                (symptom_id,),
            )
            deleted = cur.fetchone() is not None
            conn.commit()
            return deleted
    finally:
        conn.close()


def create_disease(
    disease_name: str,
    describe: Optional[str] = None,
    disease_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert a disease, generating the next ``DInnn`` code when none is given."""
    meta = _CATALOG_META["disease"]
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            code = (disease_id or "").strip().upper() or _next_code(cur, meta)
            cur.execute(
                "SELECT 1 FROM diseases WHERE disease_id = %s;", (code,)
            )
            if cur.fetchone():
                raise ValueError(f"Mã bệnh '{code}' đã tồn tại.")

            cur.execute(
                """
                INSERT INTO diseases (disease_id, disease_name, describe)
                VALUES (%s, %s, %s)
                RETURNING disease_id, disease_name, describe;
                """,
                (code, disease_name.strip(), describe),
            )
            row = dict(cur.fetchone())
            conn.commit()
            return row
    finally:
        conn.close()


def update_disease(
    disease_id: str,
    disease_name: str,
    describe: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Update a disease in place. Returns None when the code is unknown."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE diseases
                SET disease_name = %s, describe = %s
                WHERE disease_id = %s
                RETURNING disease_id, disease_name, describe;
                """,
                (disease_name.strip(), describe, disease_id),
            )
            row = cur.fetchone()
            conn.commit()
            return dict(row) if row else None
    finally:
        conn.close()


def delete_disease(disease_id: str) -> bool:
    """Delete a disease. Returns False when the code is unknown."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM diseases WHERE disease_id = %s RETURNING disease_id;",
                (disease_id,),
            )
            deleted = cur.fetchone() is not None
            conn.commit()
            return deleted
    finally:
        conn.close()
