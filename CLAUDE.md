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

Configuration is read from a `.env` file (via `python-dotenv`) at the repo root. Keep `.env` free of comments (the owner's convention) and never commit it — it is git- and docker-ignored. Variables (a blank value falls back to the default):
- `LLM_API_KEY` — required: the OpenRouter key. `LLM_BASE_URL` (default `https://openrouter.ai/api/v1`) is the OpenAI-compatible endpoint used for every generative call: vision, the fallback extraction and `/ask`. The account needs credits — DeepSeek is a paid model and answers HTTP 402 on an empty balance, while Jev keeps working.
- `JEV_URL` (default `https://openrouter.ai/api/alpha/decisions`), `JEV_API_KEY` (default `LLM_API_KEY`), `JEV_MODEL` (default `typesafe/jev-1.14`), `JEV_SYMPTOM_THRESHOLD` (default `0.5`) — the Jev decision model on OpenRouter's Decisions API.
- `LLM_VISION_MODELS`, `LLM_FALLBACK_MODELS`, `LLM_CHAT_MODELS` — comma-separated model chains, tried in order until one gives a usable answer. Defaults (in `src/extractor.py`): `deepseek/deepseek-v4.1-flash-1017` for all three. Keep each chain short: the backend allows the whole diagnosis 60 s, and every failing model in a chain spends part of it.
- Legacy names, still read when the new ones are blank so older deployments keep working: `OPENROUTER_BASE_URL`, `OPENROUTER_API_KEY`, `OPENROUTER_DECISIONS_URL`, `OPENROUTER_JEV_MODEL`. `OPENROUTER_CHAT_MODEL` is ignored; use `LLM_CHAT_MODELS`.
- `DB_HOST`, `DB_PORT`, `DB_USERNAME`, `DB_PASSWORD`, `DB_DATABASE` — PostgreSQL connection used by `src/db.py`.
- `CORS_ALLOW_ORIGINS` (optional, default `*`) — comma-separated origins allowed to call the API. The `veterinary-clinic-ai-dashboard` front-end is a separate Vite app, so CORS middleware is always installed in `main.py`.
- `AI_SERVICE_TOKEN` (optional) — when set, every `/api/v1/*` call except `/health` and `/meta` must send `Authorization: Bearer <token>`.
- `MEDIA_BASE_URLS` (optional) — comma-separated `https://` URL prefixes the service may download attachments from; empty means no remote URL is fetched (production passes the backend's `S3_PUBLIC_BASE_URL`, the CloudFront domain serving the S3 bucket).
- `MATRIX_CSV_PATH` (optional) — where the live matrix is read and written; defaults to the bundled `data/symptom_disease_matrix.csv`. The Docker image points it at the `/app/state` volume (see the matrix section).

The app will still start if the database or the OpenRouter key is unavailable (startup errors are caught and logged), but the corresponding endpoints will fail at request time.

## Architecture

This is a FastAPI service that diagnoses pet diseases and triages severity from symptoms, using a two-stage pipeline: an extraction/normalization stage (a vision LLM for attachments, the Jev decision model, and a generative fallback) followed by a deterministic scoring stage against a hand-authored symptom-disease matrix.

### Request pipeline (`src/router.py: _process_diagnosis_pipeline`)

1. **Input (`data-01.jsonc` shape)** — raw pet info, freeform `symptoms` codes, a text `describe`, optional media, and an optional `diseases` array (ground-truth labels for online learning).
2. **Extraction (`src/extractor.py: JevSymptomExtractor.process_inputs`)** — three model steps that all reduce to the same validated `CompiledExtractionOutput` (`data-02.jsonc` shape):
   - **Media → text (`src/media.py: collect_media`, then `_describe_media`)** — attachments (data URLs, URLs on the clinic's media store — the `MEDIA_BASE_URLS` prefixes (S3 via CloudFront), the only remote URLs fetched —, local paths, raw base64 or multipart uploads; at most 5 images and 2 videos) are loaded in a worker thread. Images are EXIF-rotated and shrunk to a ≤1568 px JPEG. DeepSeek V4.1 Flash reads text and images only, so a video is sent as 6 evenly spaced still frames (no sound). The vision chain answers `VISION_PROMPT` with objective per-file findings, which the case text carries as `media_findings`. Jev never receives images.
   - **Jev (`_jev_extract`)** — `POST {model, state, questions}` to `JEV_URL` via `httpx` (the Decisions API, not `/chat/completions`); `state` is the case as one JSON text item. Jev writes no text; it answers typed questions with probabilities, so the extractor builds `data-02` itself. Questions (`_build_questions`): `choice` questions for breed `BR00x`, gender `M/F/U`, weight `W0x` (not asked when no weight is given → `W00`) and triage `color_code`, plus one `score` question per symptom in the live Postgres catalog (`src/db.py: fetch_symptoms`), so Jev can only report known codes. Owner-selected symptoms get a 3-level severity scale; every other symptom gets a 4-level scale whose level 0 means "absent", and counts as present when `1 − P(absent) ≥ JEV_SYMPTOM_THRESHOLD` (`_symptom_intensity`). The triage `reasoning` is composed in code from the chosen color's probability and the strongest symptoms.
   - **Fallback (`_llm_extract`)** — on any Jev failure the fallback chain writes `data-02` as JSON (`_fallback_prompt`, JSON mode), so triage keeps working and attachments stay read. `_compile_llm_answer` rejects an invalid breed or color (the next model is tried), drops unknown symptom codes, clamps intensity to 1–3 and always keeps the owner-selected symptoms; the model's own reasoning is kept (≤400 chars).
   - **Computed in code** — the age bucket `A0x` (a plain threshold check, which Jev's docs say to keep out of the model). If attachments were sent but none could be read, `MEDIA_UNREAD_NOTE` is appended to the reasoning, which the backend shows to the owner.
   - It returns `(compiled, analysis)`: `analysis` records `extractor` (`jev` or `llm`), `model`, `jev_error`, and `media` (counts plus the vision model and findings, or its error). Only `/diagnose/detailed` returns it (the backend stores that whole response as `rawAiResponse`). When every model fails, `/diagnose*` answers 500 and the backend falls back to its keyword triage, which reads no attachments.
3. **Prediction (`src/matrix_engine.py: MatrixEngine.predict`)** — scores every disease in the matrix against the normalized pet info and symptom intensities (weighted dot product + coverage ratio + cosine similarity + a small demographic bonus for age/weight bucket), maps the score through a sigmoid to a 0.05–0.98 prevalence rate, and returns the top-k. Breed/gender act as hard filters (a disease's breed/gender lists, if non-empty, exclude non-matching pets); a disease with no overlapping symptoms is skipped entirely.
4. **Online learning (optional)** — if the caller passed `diseases` (ground-truth codes), `MatrixEngine.readjust_weights_batch` reinforces every listed disease's symptom weights (+`learning_rate`, capped at 5) and unions in the reported breed/gender, writing the CSV once for the whole batch.
5. **Output (`data-03.jsonc` shape)** plus a full audit trail — is persisted to `diagnosis_logs` via `save_diagnosis_log` (input, compiled extraction, final output, ground-truth diseases, and any weight adjustments), best-effort (a DB failure here is logged, not raised).

### The symptom-disease matrix (`data/symptom_disease_matrix.csv`)

A single flat CSV is the entire model: one row per disease, one column per breed/gender/weight-bucket/age-bucket/symptom. On disk every cell is always populated:
- `BR01..BR04`, `M`, `F` are binary flags (`1` = applicable, `0` = not). The four breed columns are the four supported species — `BR01` dog, `BR02` cat, `BR03` rabbit, `BR04` hamster — which the extractor reports as `BR001..BR004` and `MatrixEngine.BREED_ALIASES` maps back onto the column names.
- `W01..W03`, `A01..A03`, `SY001..SYnnn` are integer weights (3 = hallmark sign, 2 = common, 1 = occasional, `-1` to penalize a match); `0` means no association.

Rows and columns are keyed by catalog codes, so the matrix only means something while `DInnn`/`SYnnn` mean what the catalog (`seed_catalog.py`, then the Postgres tables) says: check the disease and symptom names when editing, since a row describing another disease makes the reference-disease list contradict the symptoms (vomiting suggesting mange, for example). `symptom-disease matrix.xlsx` at the repo root mirrors the CSV for human review (an `X` per non-zero cell) — keep the two in sync.

`MatrixEngine._load_matrix` parses this into a **sparse** in-memory structure (`matrix_data["diseases"][id] = {breeds, genders, weights, ages, symptoms}`) — zero/blank values are dropped so `predict()`'s coverage/cosine math only sees real associations. `_save_matrix` writes the dense zero-filled format back out, so the on-disk format and the in-memory format intentionally differ; when touching either loader or writer, keep both in sync (see `_parse_binary_flag` / `_parse_int_cols`).

When `MATRIX_CSV_PATH` points elsewhere (the Docker volume), `_initialize_external_matrix` seeds that copy from the bundled CSV and records the bundled file's SHA-256 in `<csv>.seed`. While the bundled file is unchanged, the live copy — with its dashboard edits and online learning — is kept across restarts; when a new image ships a changed bundled matrix, the live copy is replaced and the previous one is kept as `<csv>.bak-<timestamp>`.

### Data-contract files

`data-01.jsonc`, `data-02.jsonc`, `data-03.jsonc` at the repo root are annotated example payloads documenting the three pipeline stages above (request → compiled extraction → final response) and are the source of truth for the JSON shapes `src/schemas.py` must match — update them together with any schema change.

### Other endpoints (`src/router.py`)

`GET /api/v1/matrix` dumps the in-memory matrix; `POST /api/v1/matrix/predict` runs scoring only, from an already-compiled `data-02` payload (skips the model calls). `POST /api/v1/ask` answers short Vietnamese pet-care questions; Jev cannot write text, so this endpoint calls `/chat/completions` through the async OpenAI SDK with the `LLM_CHAT_MODELS` chain. `MatrixEngine` and `JevSymptomExtractor` are constructed once as module-level singletons in `src/router.py`, so the matrix CSV is read from disk at import time, not per-request. The extractor singleton is built inside a `try` — a missing `LLM_API_KEY` leaves `extractor is None` and only `/diagnose*` and `/ask` fail (with a 503), so the catalog and matrix endpoints stay usable.

### Dashboard-facing endpoints

These exist for `veterinary-clinic-ai-dashboard`; the three groups map one-to-one onto its three tabs.

- `POST /api/v1/diagnose/detailed` — same pipeline as `/api/v1/diagnose`, but the response adds the compiled `data-02` payload, disease names resolved from the catalog, the weight adjustments applied by online learning, and the `analysis` record. `/api/v1/diagnose` keeps the exact `data-03` contract; extend the detailed endpoint, not that one.
- `GET/POST/PUT/DELETE /api/v1/symptoms[/{id}]` and `/api/v1/diseases[/{id}]` — CRUD over the Postgres `symptoms` / `diseases` tables. `POST` generates the next `SYnnn`/`DInnn` code when the caller omits one. **The catalog and the matrix are kept in sync here**: creating a symptom adds a zero-filled matrix column, creating a disease adds a zero-filled row, and deleting either drops the corresponding column/row. Every symptom write also clears `extractor._symptoms_cache` so the next extraction uses the new catalog.
- `GET /api/v1/matrix/table` — the matrix as a **dense** table (`columns` + `rows` with every cell present), mirroring the CSV layout. This is deliberately different from `GET /api/v1/matrix`, which returns the sparse in-memory form.
- `PUT /api/v1/matrix/cell` and `PUT /api/v1/matrix/cells` — write one cell / several cells and persist the CSV. Breed and gender columns are binary (0/1); weight, age and symptom columns accept `MatrixEngine.CELL_MIN..CELL_MAX` (−3…3). Online learning can still push a weight to 5, which loads fine but can no longer be written through the API.
- `POST /api/v1/matrix/sync` — add-only reconciliation of the matrix against the catalog (also run at startup from `main.py`). It never deletes, so an unreachable or out-of-date database cannot wipe hand-authored weights; removals only happen through the explicit DELETE endpoints.
- `GET /api/v1/matrix/export` — download the CSV as an attachment. It streams the file from disk rather than re-serialising `matrix_data`, so the download is byte-for-byte what the engine will read back.
- `POST /api/v1/matrix/reload` — re-read the CSV from disk, for when the file is edited outside the API.
- `GET /api/v1/meta` (reference codes and media limits) and `GET /api/v1/health` (database / extractor model chains / matrix status; never the gateway URL or keys).
