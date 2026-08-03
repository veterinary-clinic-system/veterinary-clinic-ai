"""Rule-based Vietnamese symptom keyword extraction + severity scoring.

This is the *always-available* default triage path: pure Python, zero
external dependencies, zero API keys, deterministic. It is what keeps the
service meaningfully testable and functional out of the box.

If an LLM provider is configured (`LLM_PROVIDER` + `LLM_API_KEY`), callers
may additionally try `llm_extract()` / `extract()` for a richer,
model-assisted extraction that *prefers* the LLM result but transparently
falls back to this rule-based engine on any failure -- see `extract()`.

Severity scale (matches the 5-color triage table, worst case wins):
    0 = BLUE   (routine / follow-up)
    1 = GREEN  (non-life-threatening)
    2 = YELLOW (stable but potentially serious)
    3 = ORANGE (urgent stabilization)
    4 = RED    (immediate, life-threatening)
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)

SEVERITY_TO_COLOR = {
    0: "BLUE",
    1: "GREEN",
    2: "YELLOW",
    3: "ORANGE",
    4: "RED",
}


@dataclass(frozen=True)
class KeywordEntry:
    """One recognizable Vietnamese symptom phrase.

    `phrase` is stored with full diacritics (the canonical, "pretty" form
    returned to callers as `extracted_symptom_keywords`). Matching against
    diacritics-free / misspelled user input is handled generically by
    `_strip_diacritics()` at match time instead of hand-duplicating every
    entry -- see `rule_based_extract()`. This keeps the dictionary a single
    source of truth while still matching inputs like "kho tho nang" typed
    without accents (a very common pattern for Vietnamese mobile users).
    """

    phrase: str
    severity: int  # 0-4
    disease_group: str


@dataclass
class DiseaseGroupMatch:
    name: str
    confidence: float


@dataclass
class NlpResult:
    matched_keywords: List[str] = field(default_factory=list)
    disease_groups: List[DiseaseGroupMatch] = field(default_factory=list)
    severity: int = 0  # 0-4, worst case among matched keywords (default: routine)
    confidence: float = 0.0
    source: str = "rule_based"  # "rule_based" | "llm"


# ---------------------------------------------------------------------------
# Keyword dictionary
#
# Covers every row of the 5-color triage table with real Vietnamese
# vocabulary, plus the clinic's own `CommonSymptom` enum concepts
# (SKIN_ALLERGY, EAR_INFECTION, VOMITING, DIARRHEA, HEMATURIA,
# LOSS_OF_APPETITE, WEIGHT_LOSS, HYPERACTIVITY) mapped to a defensible
# severity:
#   - VOMITING / DIARRHEA / HEMATURIA / LOSS_OF_APPETITE / WEIGHT_LOSS
#     -> YELLOW (2): stable but need proper evaluation, can hide serious
#        underlying disease (e.g. renal failure, parvo, GI obstruction).
#   - SKIN_ALLERGY / EAR_INFECTION -> GREEN (1): uncomfortable but not
#     life-threatening, routine outpatient care.
#   - HYPERACTIVITY -> GREEN (1) rather than pure BLUE: sudden behavioural
#     change can be a pain/anxiety/medical signal worth a light checkup,
#     even though it is rarely urgent. This is a judgment call.
# ---------------------------------------------------------------------------

KEYWORDS: List[KeywordEntry] = [
    # ---- RED (4): immediate stabilization for survival --------------------
    KeywordEntry("ngừng thở", 4, "Cấp cứu hô hấp - tuần hoàn"),
    KeywordEntry("ngừng tim", 4, "Cấp cứu hô hấp - tuần hoàn"),
    KeywordEntry("tim ngừng đập", 4, "Cấp cứu hô hấp - tuần hoàn"),
    KeywordEntry("không thở được", 4, "Cấp cứu hô hấp - tuần hoàn"),
    KeywordEntry("nghẹt thở", 4, "Tắc nghẽn đường thở"),
    KeywordEntry("tắc nghẽn đường thở", 4, "Tắc nghẽn đường thở"),
    KeywordEntry("bất tỉnh", 4, "Hôn mê - bất tỉnh"),
    KeywordEntry("hôn mê", 4, "Hôn mê - bất tỉnh"),
    KeywordEntry("mất ý thức", 4, "Hôn mê - bất tỉnh"),
    KeywordEntry("chảy máu nhiều", 4, "Xuất huyết nặng"),
    KeywordEntry("chảy máu ồ ạt", 4, "Xuất huyết nặng"),
    KeywordEntry("máu chảy không ngừng", 4, "Xuất huyết nặng"),
    # ---- ORANGE (3): urgent care to stabilize ------------------------------
    KeywordEntry("khó thở nặng", 3, "Suy hô hấp cấp"),
    KeywordEntry("thở gấp nặng", 3, "Suy hô hấp cấp"),
    KeywordEntry("thở khò khè nặng", 3, "Suy hô hấp cấp"),
    KeywordEntry("co giật", 3, "Co giật - rối loạn thần kinh"),
    KeywordEntry("động kinh", 3, "Co giật - rối loạn thần kinh"),
    KeywordEntry("run rẩy toàn thân", 3, "Co giật - rối loạn thần kinh"),
    KeywordEntry("sốc", 3, "Sốc"),
    KeywordEntry("trụy tim mạch", 3, "Sốc"),
    KeywordEntry("chấn thương nặng", 3, "Chấn thương nặng"),
    KeywordEntry("gãy xương hở", 3, "Chấn thương nặng"),
    KeywordEntry("tai nạn xe", 3, "Chấn thương nặng"),
    KeywordEntry("ngã từ trên cao", 3, "Chấn thương nặng"),
    KeywordEntry("ngộ độc", 3, "Ngộ độc"),
    KeywordEntry("trúng độc", 3, "Ngộ độc"),
    KeywordEntry("ăn phải bả", 3, "Ngộ độc"),
    KeywordEntry("nôn ra máu", 3, "Xuất huyết tiêu hóa"),
    KeywordEntry("đi ngoài ra máu", 3, "Xuất huyết tiêu hóa"),
    KeywordEntry("đi ỉa ra máu", 3, "Xuất huyết tiêu hóa"),
    # ---- YELLOW (2): stable but potentially serious ------------------------
    KeywordEntry("nôn nhiều", 2, "Rối loạn tiêu hóa"),
    KeywordEntry("nôn liên tục", 2, "Rối loạn tiêu hóa"),
    KeywordEntry("nôn mửa", 2, "Rối loạn tiêu hóa"),
    KeywordEntry("nôn ói", 2, "Rối loạn tiêu hóa"),
    KeywordEntry("tiêu chảy", 2, "Rối loạn tiêu hóa"),
    KeywordEntry("đi phân lỏng", 2, "Rối loạn tiêu hóa"),
    KeywordEntry("đau bụng", 2, "Đau bụng - nội tạng"),
    KeywordEntry("bụng chướng", 2, "Đau bụng - nội tạng"),
    KeywordEntry("bụng phình to", 2, "Đau bụng - nội tạng"),
    KeywordEntry("sốt cao", 2, "Sốt cao"),
    KeywordEntry("sốt cao liên tục", 2, "Sốt cao"),
    KeywordEntry("thân nhiệt cao", 2, "Sốt cao"),
    KeywordEntry("tiểu ra máu", 2, "Bất thường tiết niệu"),
    KeywordEntry("đi tiểu ra máu", 2, "Bất thường tiết niệu"),
    KeywordEntry("nước tiểu có máu", 2, "Bất thường tiết niệu"),
    KeywordEntry("bỏ ăn", 2, "Chán ăn - suy nhược"),
    KeywordEntry("chán ăn", 2, "Chán ăn - suy nhược"),
    KeywordEntry("không chịu ăn", 2, "Chán ăn - suy nhược"),
    KeywordEntry("sụt cân", 2, "Sụt cân bất thường"),
    KeywordEntry("giảm cân nhanh", 2, "Sụt cân bất thường"),
    KeywordEntry("gầy đi trông thấy", 2, "Sụt cân bất thường"),
    # ---- GREEN (1): non-life-threatening injury/illness --------------------
    KeywordEntry("khập khiễng", 1, "Chấn thương nhẹ - vận động"),
    KeywordEntry("đi khập khiễng", 1, "Chấn thương nhẹ - vận động"),
    KeywordEntry("đi cà nhắc", 1, "Chấn thương nhẹ - vận động"),
    KeywordEntry("đau chân nhẹ", 1, "Chấn thương nhẹ - vận động"),
    KeywordEntry("ngứa da", 1, "Da liễu"),
    KeywordEntry("dị ứng da", 1, "Da liễu"),
    KeywordEntry("mẩn đỏ", 1, "Da liễu"),
    KeywordEntry("nổi mẩn", 1, "Da liễu"),
    KeywordEntry("rụng lông", 1, "Da liễu"),
    KeywordEntry("chảy dịch mắt", 1, "Bệnh về mắt"),
    KeywordEntry("chảy nước mắt", 1, "Bệnh về mắt"),
    KeywordEntry("đỏ mắt", 1, "Bệnh về mắt"),
    KeywordEntry("ghèn mắt", 1, "Bệnh về mắt"),
    KeywordEntry("viêm tai", 1, "Bệnh về tai"),
    KeywordEntry("chảy dịch tai", 1, "Bệnh về tai"),
    KeywordEntry("có mùi hôi ở tai", 1, "Bệnh về tai"),
    KeywordEntry("hiếu động quá mức", 1, "Hành vi - tăng động"),
    KeywordEntry("tăng động", 1, "Hành vi - tăng động"),
    # ---- BLUE (0): routine / ongoing follow-up -----------------------------
    KeywordEntry("khám định kỳ", 0, "Khám định kỳ"),
    KeywordEntry("khám sức khỏe định kỳ", 0, "Khám định kỳ"),
    KeywordEntry("tái khám", 0, "Tái khám"),
    KeywordEntry("tư vấn", 0, "Tư vấn chung"),
    KeywordEntry("hỏi thông tin", 0, "Tư vấn chung"),
    KeywordEntry("tiêm phòng", 0, "Tiêm phòng - phòng bệnh"),
    KeywordEntry("tiêm vắc xin", 0, "Tiêm phòng - phòng bệnh"),
]


def _strip_diacritics(text: str) -> str:
    """Best-effort ASCII-fold of Vietnamese text (accents removed).

    NFD-decomposes accented Latin letters and drops the combining marks;
    "đ"/"Đ" don't decompose that way in Vietnamese, so they're mapped
    explicitly. Used as a *secondary* match pass so users typing without
    diacritics (very common on phones, e.g. "kho tho nang") still match.
    """
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def compute_nlp_confidence(text: str, matches: List[KeywordEntry]) -> float:
    """Heuristic confidence in the rule-based extraction.

    - Empty/blank input carries no signal at all -> fixed low confidence.
    - Otherwise: base 0.3 (we at least received *some* text) + 0.15 per
      distinct matched keyword, capped at 1.0. More keyword hits means
      more corroborating evidence that we understood the complaint.
    """
    if not text or not text.strip():
        return 0.2
    return min(1.0, 0.3 + 0.15 * len(matches))


def rule_based_extract(text: str) -> NlpResult:
    """Deterministic, dependency-free Vietnamese symptom extraction."""
    normalized = _normalize(text or "")
    stripped = _strip_diacritics(normalized)

    matched_entries: List[KeywordEntry] = []
    seen_phrases: set[str] = set()
    for entry in KEYWORDS:
        entry_norm = entry.phrase.lower()
        entry_stripped = _strip_diacritics(entry_norm)
        if entry_norm in normalized or entry_stripped in stripped:
            if entry.phrase not in seen_phrases:
                matched_entries.append(entry)
                seen_phrases.add(entry.phrase)

    severity = max((e.severity for e in matched_entries), default=0)

    # Aggregate per disease group: more corroborating keyword hits for the
    # same group -> higher confidence in that group, capped at 0.95 (never
    # absolute certainty from keyword matching alone).
    group_hit_counts: dict[str, int] = {}
    for entry in matched_entries:
        group_hit_counts[entry.disease_group] = group_hit_counts.get(entry.disease_group, 0) + 1

    disease_groups = [
        DiseaseGroupMatch(name=name, confidence=min(0.95, 0.4 + 0.2 * count))
        for name, count in group_hit_counts.items()
    ]
    disease_groups.sort(key=lambda g: g.confidence, reverse=True)

    confidence = compute_nlp_confidence(text, matched_entries)

    return NlpResult(
        matched_keywords=[e.phrase for e in matched_entries],
        disease_groups=disease_groups,
        severity=severity,
        confidence=confidence,
        source="rule_based",
    )


async def llm_extract(text: str) -> Optional[NlpResult]:
    """Best-effort LLM-assisted structured extraction.

    Returns None (never raises) on missing config, network error, timeout,
    or malformed response so callers can transparently fall back to
    `rule_based_extract()`.
    """
    settings = get_settings()
    if settings.LLM_PROVIDER == "none" or not settings.LLM_API_KEY:
        return None

    # Imported lazily to avoid an import-time dependency on httpx being
    # exercised when no LLM provider is configured.
    from app.services import llm_client

    system_prompt = (
        "Bạn là bộ trích xuất triệu chứng thú y cho một hệ thống phòng khám thú cưng. "
        "Đọc mô tả triệu chứng bằng tiếng Việt của chủ nuôi và trả lời DUY NHẤT một đối "
        "tượng JSON hợp lệ, không kèm giải thích, không markdown, đúng định dạng:\n"
        '{"keywords": ["tu khoa trieu chung 1", ...], '
        '"disease_groups": ["nhom benh nghi ngo 1", ...], '
        '"severity": 0, '
        '"confidence": 0.0}\n'
        "Trong đó severity là số nguyên 0-4 theo thang độ nặng: "
        "0=theo doi dinh ky/khong khan cap, 1=nhe khong nguy hiem tinh mang, "
        "2=on dinh nhung can kham them, 3=can can thiep khan cap de on dinh, "
        "4=nguy hiem tinh mang can cap cuu ngay lap tuc. "
        "confidence la so thuc 0-1 the hien do tin cay cua ban."
    )

    try:
        raw = await llm_client.complete(
            messages=[{"role": "user", "content": text}],
            settings=settings,
            system_prompt=system_prompt,
            json_mode=True,
            max_tokens=400,
        )
        if not raw:
            return None

        data = llm_client.parse_json_object(raw)
        if data is None:
            return None

        keywords = [str(k) for k in data.get("keywords", []) if str(k).strip()]
        groups_raw = [str(g) for g in data.get("disease_groups", []) if str(g).strip()]
        severity = int(data.get("severity", 0))
        severity = max(0, min(4, severity))
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        disease_groups = [DiseaseGroupMatch(name=g, confidence=confidence) for g in groups_raw]

        return NlpResult(
            matched_keywords=keywords,
            disease_groups=disease_groups,
            severity=severity,
            confidence=confidence,
            source="llm",
        )
    except Exception:  # noqa: BLE001 - any failure here must degrade gracefully
        logger.warning("LLM triage extraction failed; falling back to rule-based engine", exc_info=True)
        return None


async def extract(text: str) -> NlpResult:
    """Top-level entry point used by the triage engine.

    Tries the LLM-assisted extractor first (only if configured); falls
    back to the always-available rule-based engine on any failure or when
    no LLM provider is configured.
    """
    settings = get_settings()
    if settings.LLM_PROVIDER != "none" and settings.LLM_API_KEY:
        llm_result = await llm_extract(text)
        if llm_result is not None:
            return llm_result
    return rule_based_extract(text)
