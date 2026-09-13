# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

There is no build step, linter, or test suite configured in this repository yet.

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API (auto-reload dev server on http://0.0.0.0:8000)
python main.py
# or
uvicorn main:app --reload
```

Configuration is read from a `.env` file (via `python-dotenv`) at the repo root. Required variables:
- `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_MODEL` (optionally `OPENROUTER_REFERER`, `OPENROUTER_TITLE`) — used by `src/extractor.py` to call the DeepSeek model through OpenRouter.
- `DB_HOST`, `DB_PORT`, `DB_USERNAME`, `DB_PASSWORD`, `DB_DATABASE` — PostgreSQL connection used by `src/db.py`.
- `CORS_ALLOW_ORIGINS` (optional, default `*`) — comma-separated origins allowed to call the API. The `veterinary-clinic-ai-dashboard` front-end is a separate Vite app, so CORS middleware is always installed in `main.py`.

The app will still start if the database or OpenRouter key is unavailable (startup errors are caught and logged), but the corresponding endpoints will fail at request time.

## Architecture

This is a FastAPI service that diagnoses pet diseases and triages severity from symptoms, using a two-stage pipeline: an LLM extraction/normalization stage followed by a deterministic scoring stage against a hand-authored symptom-disease matrix.

### Request pipeline (`src/router.py: _process_diagnosis_pipeline`)

1. **Input (`data-01.jsonc` shape)** — raw pet info, freeform `symptoms` codes, a text `describe`, optional media, and an optional `diseases` array (ground-truth labels for online learning).
2. **Extraction (`src/extractor.py: DeepSeekSymptomExtractor`)** — sends pet info, symptoms, description, and any images/video frames to a DeepSeek model via the OpenAI SDK pointed at OpenRouter. The system prompt (built in `_build_system_instruction`) embeds the live symptom catalog fetched from Postgres (`src/db.py: fetch_symptoms`) and instructs the model to return normalized pet-info codes (breed `BR00x`, gender `M/F`, weight `W0x`, age `A0x`), a symptom list with 1–3 intensity, and a triage `color_code`. The raw model output is parsed into `CompiledExtractionOutput` (`data-02.jsonc` shape).
3. **Prediction (`src/matrix_engine.py: MatrixEngine.predict`)** — scores every disease in the matrix against the normalized pet info and symptom intensities (weighted dot product + coverage ratio + cosine similarity + a small demographic bonus for age/weight bucket), maps the score through a sigmoid to a 0.05–0.98 prevalence rate, and returns the top-k. Breed/gender act as hard filters (a disease's breed/gender lists, if non-empty, exclude non-matching pets); a disease with no overlapping symptoms is skipped entirely.
4. **Online learning (optional)** — if the caller passed `diseases` (ground-truth codes), `MatrixEngine.readjust_weights_batch` reinforces every listed disease's symptom weights (+`learning_rate`, capped at 5) and unions in the reported breed/gender, writing the CSV once for the whole batch.
5. **Output (`data-03.jsonc` shape)** plus a full audit trail — is persisted to `diagnosis_logs` via `save_diagnosis_log` (input, compiled extraction, final output, ground-truth diseases, and any weight adjustments), best-effort (a DB failure here is logged, not raised).

### The symptom-disease matrix (`data/symptom_disease_matrix.csv`)

A single flat CSV is the entire model: one row per disease, one column per breed/gender/weight-bucket/age-bucket/symptom. On disk every cell is always populated:
- `BR01..BR04`, `M`, `F` are binary flags (`1` = applicable, `0` = not). The four breed columns are the four supported species — `BR01` dog, `BR02` cat, `BR03` rabbit, `BR04` hamster — which the LLM reports as `BR001..BR004` and `MatrixEngine.BREED_ALIASES` maps back onto the column names.
- `W01..W03`, `A01..A03`, `SY001..SYnnn` are integer weights (typically 1–3, occasionally negative to penalize a match); `0` means no association.

`MatrixEngine._load_matrix` parses this into a **sparse** in-memory structure (`matrix_data["diseases"][id] = {breeds, genders, weights, ages, symptoms}`) — zero/blank values are dropped so `predict()`'s coverage/cosine math only sees real associations. `_save_matrix` writes the dense zero-filled format back out, so the on-disk format and the in-memory format intentionally differ; when touching either loader or writer, keep both in sync (see `_parse_binary_flag` / `_parse_int_cols`).

### Data-contract files

`data-01.jsonc`, `data-02.jsonc`, `data-03.jsonc` at the repo root are annotated example payloads documenting the three pipeline stages above (request → compiled extraction → final response) and are the source of truth for the JSON shapes `src/schemas.py` must match — update them together with any schema change.

### Other endpoints (`src/router.py`)

`GET /api/v1/matrix` dumps the in-memory matrix; `POST /api/v1/matrix/predict` runs scoring only, from an already-compiled `data-02` payload (skips the LLM call). `MatrixEngine` and `DeepSeekSymptomExtractor` are constructed once as module-level singletons in `src/router.py`, so the matrix CSV is read from disk at import time, not per-request. The extractor singleton is built inside a `try` — a missing/invalid `OPENROUTER_API_KEY` leaves `extractor is None` and only `/diagnose*` fails (with a 503), so the catalog and matrix endpoints stay usable.

### Dashboard-facing endpoints

These exist for `veterinary-clinic-ai-dashboard`; the three groups map one-to-one onto its three tabs.

- `POST /api/v1/diagnose/detailed` — same pipeline as `/api/v1/diagnose`, but the response adds the compiled `data-02` payload, disease names resolved from the catalog, and the weight adjustments applied by online learning. `/api/v1/diagnose` keeps the exact `data-03` contract; extend the detailed endpoint, not that one.
- `GET/POST/PUT/DELETE /api/v1/symptoms[/{id}]` and `/api/v1/diseases[/{id}]` — CRUD over the Postgres `symptoms` / `diseases` tables. `POST` generates the next `SYnnn`/`DInnn` code when the caller omits one. **The catalog and the matrix are kept in sync here**: creating a symptom adds a zero-filled matrix column, creating a disease adds a zero-filled row, and deleting either drops the corresponding column/row. Every symptom write also clears `extractor._symptoms_cache` so the next LLM prompt sees the new catalog.
- `GET /api/v1/matrix/table` — the matrix as a **dense** table (`columns` + `rows` with every cell present), mirroring the CSV layout. This is deliberately different from `GET /api/v1/matrix`, which returns the sparse in-memory form.
- `PUT /api/v1/matrix/cell` and `PUT /api/v1/matrix/cells` — write one cell / several cells and persist the CSV. Breed and gender columns are binary (0/1); weight, age and symptom columns accept `MatrixEngine.CELL_MIN..CELL_MAX` (−3…3). Note the CSV still contains a few legacy out-of-range values (a `4`, a `5`) that load fine but can no longer be written through the API.
- `POST /api/v1/matrix/sync` — add-only reconciliation of the matrix against the catalog (also run at startup from `main.py`). It never deletes, so an unreachable or out-of-date database cannot wipe hand-authored weights; removals only happen through the explicit DELETE endpoints.
- `GET /api/v1/matrix/export` — download the CSV as an attachment. It streams the file from disk rather than re-serialising `matrix_data`, so the download is byte-for-byte what the engine will read back.
- `POST /api/v1/matrix/reload` — re-read the CSV from disk, for when the file is edited outside the API.
- `GET /api/v1/meta` (reference codes and media limits) and `GET /api/v1/health` (database / extractor / matrix status).
