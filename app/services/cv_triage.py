"""Zero-shot image classification (CLIP) for photo-based triage signals.

Optional and best-effort: the triage endpoint must keep working (NLP-only)
even when this entire step is unavailable. `classify_photos()` returns
`None` -- never raises -- whenever:
  - `settings.ENABLE_CV` is `False`,
  - no `photo_urls` were provided,
  - `transformers`/`torch` are not installed (heavy, optional deps), or
  - the model fails to download, or every photo fails to download/decode.

The Hugging Face zero-shot-image-classification pipeline is lazy-loaded
once per process (module-level cache) since loading model weights is slow
and must not happen on every request.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import httpx
from PIL import Image

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# English works best with CLIP even though the rest of the product is
# Vietnamese-facing; we translate the winning label to a Vietnamese
# disease-group name (and an implied severity weight, 0-4, matching the
# same triage scale used by nlp_triage) before returning it.
#
# label -> (vietnamese_disease_group, severity_weight)
CANDIDATE_LABELS: Dict[str, Tuple[str, int]] = {
    "severe bleeding": ("Xuất huyết nặng", 4),
    "labored breathing": ("Suy hô hấp cấp", 3),
    "open wound or laceration": ("Vết thương hở", 2),
    "swelling or lump": ("Sưng - nổi cục bất thường", 2),
    "skin rash or allergy": ("Da liễu", 1),
    "ear infection": ("Bệnh về tai", 1),
    "eye discharge or conjunctivitis": ("Bệnh về mắt", 1),
    "healthy skin": ("Da khỏe mạnh", 0),
    "normal healthy pet": ("Thú cưng khỏe mạnh", 0),
}


@dataclass
class CvResult:
    disease_group: str
    confidence: float
    severity: int  # 0-4


@lru_cache(maxsize=1)
def _load_pipeline() -> Any:
    """Lazily import transformers/torch and build the zero-shot pipeline.

    Raises ImportError if the heavy ML deps aren't installed; raises
    whatever transformers/huggingface_hub raise on a download failure.
    Callers are responsible for catching and degrading gracefully.
    """
    from transformers import pipeline  # heavy import, deferred on purpose

    settings = get_settings()
    logger.info("Loading CV zero-shot classification pipeline: %s", settings.CV_MODEL_NAME)
    return pipeline("zero-shot-image-classification", model=settings.CV_MODEL_NAME)


def _download_image(url: str, timeout_seconds: float) -> Image.Image:
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.get(url)
        response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def classify_photos(photo_urls: List[str]) -> Optional[CvResult]:
    """Run zero-shot classification over one or more photo URLs.

    Aggregation strategy when multiple photos are given: take the single
    highest-confidence label across all photos (rather than averaging
    per-label scores). Triage is worst-case driven -- one photo that
    clearly shows e.g. an open wound should not be diluted by other,
    less-clear photos of the same pet. Returns `None` on any failure.
    """
    settings = get_settings()
    if not settings.ENABLE_CV or not photo_urls:
        return None

    try:
        classifier = _load_pipeline()
    except ImportError:
        logger.warning("transformers/torch not installed; skipping CV triage step")
        return None
    except Exception:  # noqa: BLE001 - model download/load failure, degrade gracefully
        logger.warning("Failed to load CV model; skipping CV triage step", exc_info=True)
        return None

    candidate_labels = list(CANDIDATE_LABELS.keys())
    best: Optional[CvResult] = None

    for url in photo_urls:
        try:
            image = _download_image(url, settings.HTTP_TIMEOUT_SECONDS)
            predictions = classifier(image, candidate_labels=candidate_labels)
            if not predictions:
                continue
            top = predictions[0]
            label = top["label"]
            score = float(top["score"])
            group_name, severity = CANDIDATE_LABELS.get(label, (label, 0))
            candidate = CvResult(disease_group=group_name, confidence=score, severity=severity)
            if best is None or candidate.confidence > best.confidence:
                best = candidate
        except Exception:  # noqa: BLE001 - one bad photo must not fail the whole request
            logger.warning("CV inference failed for photo url=%s", url, exc_info=True)
            continue

    return best
