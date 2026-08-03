# veterinary-clinic-ai

Standalone Python/FastAPI microservice providing AI pre-screening (triage)
and a safety-scoped chat assistant for the veterinary clinic management
system built for this graduation thesis (KLTN).

## What this service does, and how it fits into the system

`veterinary-clinic-ai` is a small internal service with exactly two
functional endpoints:

- **Triage** (`POST /api/v1/triage`): given free-text symptom description
  (Vietnamese) and optionally some photos, produces a 5-color priority
  rating, a list of suspected disease groups, and confidence scores --
  meant to help clinic staff / the booking flow prioritize incoming
  appointment requests.
- **Chat** (`POST /api/v1/chat`): a conversational assistant for pet
  owners that answers general questions and gently nudges towards
  booking an appointment when appropriate, while explicitly never
  attempting to diagnose (see **Safety design** below).

The **NestJS backend** (`veterinary-clinic-backend`) is the only expected
caller of this service, over plain internal REST (e.g. inside the same
docker network / private network -- there is no browser client calling
this service directly, which is why CORS is intentionally not enabled).
Every request to `/api/v1/*` must carry:

```
Authorization: Bearer <AI_SERVICE_TOKEN>
```

where `AI_SERVICE_TOKEN` is a shared secret configured identically on
both sides. `GET /health` is unauthenticated, for container/uptime
probes.

## API contract

### `POST /api/v1/triage`

Request:

```json
{
  "symptom_text": "Bé nhà em bị sốt cao và nôn nhiều hai ngày nay",
  "photo_urls": [],
  "pet_species": "dog",
  "pet_age_months": 24
}
```

- `symptom_text`: Vietnamese free text, may be empty.
- `photo_urls`: list of publicly reachable image URLs, may be empty.
- `pet_species`, `pet_age_months`: optional context, currently informational.

Response:

```json
{
  "priority_color": "YELLOW",
  "suspected_disease_groups": [
    { "name": "Rối loạn tiêu hóa", "confidence": 0.8 },
    { "name": "Sốt cao", "confidence": 0.6 }
  ],
  "extracted_symptom_keywords": ["sốt cao", "nôn nhiều"],
  "nlp_confidence": 0.6,
  "cv_confidence": null,
  "overall_confidence": 0.6
}
```

`priority_color` is one of `RED | ORANGE | YELLOW | GREEN | BLUE` (see the
triage table below). `cv_confidence` is `null` whenever no photos were
supplied or the CV step was skipped/failed.

### `POST /api/v1/chat`

Request:

```json
{
  "session_id": null,
  "message": "Chó nhà em bị co giật liên tục, phải làm sao đây?",
  "history": [{ "role": "user", "content": "..." }]
}
```

Response:

```json
{
  "reply": "Những gì bạn mô tả nghe có vẻ là tình huống khẩn cấp...",
  "suggest_booking": true,
  "session_id": "b3f1c2a4-... (generated uuid4 since none was given)"
}
```

## The 5-color triage table

| Color  | Meaning                                   | Example symptoms/keywords                                      |
| ------ | ------------------------------------------ | ---------------------------------------------------------------- |
| RED    | Immediate stabilization for survival       | cardiac/respiratory arrest, massive hemorrhage, unconscious      |
| ORANGE | Urgent care to stabilize                   | severe respiratory distress, shock, continuous seizures          |
| YELLOW | Stable but potentially serious             | frequent vomiting/diarrhea, abdominal pain, high fever           |
| GREEN  | Non-life-threatening injury/illness        | lameness, skin rash, ear infection, eye discharge                |
| BLUE   | Routine/ongoing follow-up                  | routine consultation, minor recheck, non-urgent questions        |

Internally this is represented as a severity integer 0-4 (`BLUE=0` ...
`RED=4`); see `app/services/nlp_triage.py::SEVERITY_TO_COLOR`.

## Model choices, and why

### NLP: rule-based Vietnamese keyword engine (always-on default)

`app/services/nlp_triage.py` implements a deterministic, dependency-free
keyword-matching + severity-scoring engine over a hand-built dictionary of
Vietnamese symptom phrases (with automatic diacritics-stripped matching,
since Vietnamese mobile users very often type without accents). This is
the **reliable default path**: it needs no API keys, no network calls,
and no ML runtime, so the service is meaningfully testable and usable out
of the box, and a demo/thesis environment never depends on an external
LLM being reachable.

If `LLM_PROVIDER` is configured with an API key, the service *additionally*
attempts an LLM-based structured extraction (`app/services/llm_client.py`
+ `nlp_triage.llm_extract`) and prefers that richer result -- but on any
timeout/error/malformed-JSON response it transparently falls back to the
rule-based result. This is a pragmatic, pluggable upgrade path rather than
a hard dependency.

### CV: CLIP zero-shot image classification (optional, graceful degradation)

`app/services/cv_triage.py` uses Hugging Face `transformers`'
`zero-shot-image-classification` pipeline with a pretrained CLIP model
(`openai/clip-vit-base-patch32` by default) against a fixed list of
English veterinary-issue labels (CLIP performs best in English), mapped
back to a Vietnamese disease-group name and an implied severity weight.
No fine-tuning/training was in scope for this student project -- a
pretrained zero-shot classifier is a defensible, honest choice that still
provides a real (if coarse) visual signal rather than a fabricated one.

This entire step is **best-effort**: `classify_photos()` returns `None`
(never raises) when `ENABLE_CV=false`, when `transformers`/`torch` are not
installed, when no photos were given, or when model loading / downloading
/ inference fails for any reason. The triage endpoint always still
returns a valid, well-formed NLP-only response in that case
(`cv_confidence: null`).

### Chat replies

Rule-based Vietnamese templates by default (varied per case -- see
**Safety design**), optionally upgraded to LLM-generated phrasing when
`LLM_PROVIDER` is configured, with the same fallback-on-failure behavior
as triage.

## Environment variables (`.env.example`)

| Variable                | Default                          | Meaning                                                                 |
| ------------------------ | --------------------------------- | ------------------------------------------------------------------------ |
| `AI_SERVICE_TOKEN`      | *(required)*                     | Shared bearer-token secret the NestJS backend must send.                |
| `PORT`                  | `8000`                           | Port for local `uvicorn` runs.                                          |
| `LLM_PROVIDER`          | `none`                           | `none` \| `openai` \| `anthropic`.                                      |
| `LLM_API_KEY`           | *(empty)*                        | API key for the selected provider; required if `LLM_PROVIDER != none`. |
| `LLM_API_BASE_URL`      | *(provider default)*             | Override for OpenAI-compatible custom/self-hosted endpoints.            |
| `LLM_MODEL`             | *(provider default)*             | e.g. `gpt-4o-mini`, `claude-3-5-haiku-20241022`.                        |
| `ENABLE_CV`             | `true`                           | Set `false` to skip the CV step entirely (no torch/transformers load). |
| `CV_MODEL_NAME`         | `openai/clip-vit-base-patch32`   | Hugging Face model id for zero-shot image classification.               |
| `HTTP_TIMEOUT_SECONDS`  | `10`                             | Timeout for outbound HTTP calls (LLM requests, photo downloads).       |

## Running locally

```powershell
cd veterinary-clinic-ai
pip install -r requirements.txt
copy .env.example .env    # then edit AI_SERVICE_TOKEN etc.
uvicorn app.main:app --reload --port 8000
```

Run the test suite (no external services/API keys required -- the tests
force `ENABLE_CV=false` so they never need torch/transformers installed):

```powershell
pytest
```

## Safety design

The chat assistant (`app/services/chat_engine.py`) is a
**safety-relevant product decision**, documented here explicitly and not
just as an implementation detail:

1. **Informational support only -- never a diagnosis.** Every reply,
   whether LLM-generated or from the built-in canned templates, is
   required to avoid definitive diagnostic claims ("your pet has X") and
   instead talk about possibilities and recommended next steps (see a
   vet, book an exam, seek emergency care).

2. **Deterministic red-flag detection, independent of the LLM.** The
   decision of *whether a message sounds urgent* is always computed by
   the same rule-based Vietnamese keyword/severity engine used by the
   triage endpoint (`nlp_triage.rule_based_extract`) -- **not** the
   optional LLM-assisted extractor. This means the safety-critical
   "should we tell this person to seek emergency care" decision never
   depends on an external API being reachable, and can never be
   prompt-injected away by adversarial input to an LLM.

3. **Three response cases, in order of precedence:**
   - **Urgent** (matched keyword severity is ORANGE/RED, i.e. `>= 3`):
     `suggest_booking = true`, and the reply clearly and unambiguously
     recommends immediate in-person/emergency care, with no hedging.
   - **Symptom-like** (any symptom or appointment-type keyword matched,
     but not urgent): `suggest_booking = true`, and the reply
     acknowledges what was described and gently recommends booking a
     proper exam, while staying conversationally helpful.
   - **General** (no keywords matched at all -- opening hours, services,
     generic pet-care questions): `suggest_booking = false`, and the
     reply just answers helpfully and briefly, without a fabricated,
     possibly-wrong claim about specific clinic details the service
     doesn't actually have data for.

4. **The LLM only ever phrases the reply, never decides `suggest_booking`.**
   When configured, the LLM's system prompt (in Vietnamese) explicitly
   states it is not a veterinarian, must not diagnose, and must reflect
   the urgency level the rule-based engine already determined. If the LLM
   call fails or times out, a real (not a single generic placeholder),
   case-specific canned Vietnamese reply is used instead -- the fallback
   always reflects both which case was hit and the `suggest_booking`
   value.

## Design decisions worth a reviewer's attention

- **Severity-scoring formula**: `nlp_confidence = 0.2` for blank input,
  else `min(1.0, 0.3 + 0.15 * number_of_matched_keywords)`. `text_severity`
  is the **max** severity among matched keywords (worst case wins, not an
  average), defaulting to `0` (BLUE) when nothing matched.
- **Diacritics handling**: the keyword dictionary stores one canonical,
  accented Vietnamese phrase per entry; matching is attempted both
  against the raw normalized text and against a diacritics-stripped
  version of both the dictionary phrase and the input text
  (`unicodedata` NFD decomposition + explicit `đ`/`Đ` handling). This
  avoids hand-duplicating every entry while still matching common
  no-diacritics typing.
- **CV label -> disease-group mapping / severity**: `severe bleeding` ->
  RED(4); `labored breathing` -> ORANGE(3); `open wound or laceration` and
  `swelling or lump` -> YELLOW(2); `skin rash or allergy`, `ear
  infection`, `eye discharge or conjunctivitis` -> GREEN(1); `healthy
  skin`, `normal healthy pet` -> BLUE(0). When multiple photos are given,
  the **single highest-confidence label across all photos** is used
  (not an average), consistent with triage being worst-case-driven.
- **Combining NLP + CV**: `final_severity = max(nlp_severity, cv_severity
  or 0)`; `overall_confidence = 0.6 * nlp_confidence + 0.4 *
  cv_confidence` when both are present (NLP weighted higher since it's
  always in-domain Vietnamese text; CV is a coarser, English-label
  zero-shot signal), else just `nlp_confidence`.
- **`CommonSymptom` enum coverage**: `VOMITING` / `DIARRHEA` / `HEMATURIA`
  / `LOSS_OF_APPETITE` / `WEIGHT_LOSS` are mapped to YELLOW (can mask
  serious underlying disease); `SKIN_ALLERGY` / `EAR_INFECTION` to GREEN;
  `HYPERACTIVITY` to GREEN rather than pure BLUE, since a sudden
  behavioural change can itself be a pain/medical signal worth a
  light checkup even though it's rarely urgent -- a judgment call.
