"""Safety-scoped conversational assistant.

*** SAFETY DESIGN (also documented prominently in README.md) ***

This assistant is **informational support only -- it never diagnoses**.
Concretely:

1. Red-flag detection reuses the exact same deterministic, rule-based
   Vietnamese keyword/severity engine as the triage endpoint
   (`nlp_triage.rule_based_extract`) -- NOT the optional LLM path -- so the
   safety-critical decision of "is this urgent" never depends on an
   external API being reachable or behaving.
2. Three cases, decided purely by that rule-based severity/keyword signal:
     a. `severity >= 3` (ORANGE/RED-level language, e.g. "khó thở nặng",
        "co giật", "sốc") -> `suggest_booking=True` and the reply clearly
        recommends immediate in-person / emergency care.
     b. Any symptom/appointment keyword matched but severity < 3 (GREEN/
        YELLOW/BLUE keyword hits, e.g. "ngứa da", "nôn nhiều", "tái
        khám") -> `suggest_booking=True`, reply acknowledges what was
        described and gently recommends booking a proper exam.
     c. No keywords matched at all (general questions: opening hours,
        services offered, generic pet-care questions) ->
        `suggest_booking=False`, reply just answers helpfully/briefly.
3. Whether the reply text is produced by the optional LLM or by the
   built-in canned templates, it must never state a definitive diagnosis
   ("your pet has X"); it should talk in terms of possibilities and next
   steps, and explicitly hand off to a human vet / booking / emergency
   care. The LLM system prompt spells this out; the canned templates are
   themselves written that way by construction.
4. The LLM (if configured) is only ever used to *phrase* the reply -- the
   `suggest_booking` decision itself is always computed by the
   deterministic rule-based path, never left to the LLM, so it cannot be
   prompt-injected away from recommending emergency care.
"""

from __future__ import annotations

import logging
from typing import List, Optional
from uuid import uuid4

from app.core.config import get_settings
from app.schemas.chat import ChatRequest, ChatResponse
from app.services import nlp_triage
from app.services.nlp_triage import NlpResult

logger = logging.getLogger(__name__)

_URGENT_SEVERITY_THRESHOLD = 3  # ORANGE (3) and RED (4)

_SYSTEM_PROMPT_TEMPLATE = """Bạn là trợ lý ảo hỗ trợ thông tin cho một phòng khám thú y.
QUY TẮC AN TOÀN BẮT BUỘC (không được vi phạm):
- Bạn KHÔNG PHẢI là bác sĩ thú y và TUYỆT ĐỐI KHÔNG được đưa ra chẩn đoán xác định \
(không nói kiểu "bé nhà bạn chắc chắn bị bệnh X"). Chỉ được nói về khả năng và các \
bước nên làm tiếp theo.
- Hệ thống đã phân tích tin nhắn và xác định: mức độ khẩn cấp = {urgency_label}; \
từ khóa triệu chứng phát hiện được = {keywords}.
- Nếu mức độ khẩn cấp là KHẨN CẤP: PHẢI khuyến cáo rõ ràng, dứt khoát đưa thú cưng \
đến cơ sở cấp cứu thú y hoặc phòng khám gần nhất NGAY LẬP TỨC, không trì hoãn.
- Nếu có từ khóa triệu chứng nhưng không khẩn cấp: hãy thể hiện sự đồng cảm, tóm tắt \
lại ngắn gọn điều người dùng vừa mô tả, và nhẹ nhàng khuyến nghị đặt lịch khám để \
bác sĩ thú y kiểm tra trực tiếp -- vì bạn không thể chẩn đoán qua tin nhắn.
- Nếu là câu hỏi chung chung (giờ làm việc, dịch vụ, chăm sóc thú cưng cơ bản...): \
trả lời ngắn gọn, hữu ích, thân thiện; không cần gợi ý đặt lịch nếu không liên quan.
- Luôn trả lời bằng tiếng Việt, giọng điệu thân thiện, súc tích (tối đa vài câu).
"""


def _case_for(nlp_result: NlpResult) -> str:
    if nlp_result.severity >= _URGENT_SEVERITY_THRESHOLD:
        return "urgent"
    if nlp_result.matched_keywords:
        return "symptom"
    return "general"


def _canned_reply(case: str, nlp_result: NlpResult) -> str:
    """Rule-based fallback reply, always real and case-specific -- never a
    single hardcoded sentence regardless of input. Used whenever the LLM
    is not configured or fails.
    """
    keyword_list = "、".join(nlp_result.matched_keywords) if nlp_result.matched_keywords else ""

    if case == "urgent":
        signal = f" (dấu hiệu ghi nhận: {keyword_list})" if keyword_list else ""
        return (
            f"Những gì bạn mô tả{signal} nghe có vẻ là tình huống khẩn cấp, có thể nguy hiểm "
            "đến tính mạng của thú cưng. Vui lòng đưa bé đến phòng khám gần nhất hoặc cơ sở "
            "cấp cứu thú y NGAY BÂY GIỜ, hoặc gọi điện trước để được hướng dẫn sơ cứu. "
            "Đây chỉ là thông tin tham khảo ban đầu, không thay thế cho việc thăm khám trực "
            "tiếp của bác sĩ thú y."
        )

    if case == "symptom":
        signal = f" như: {keyword_list}" if keyword_list else ""
        return (
            f"Cảm ơn bạn đã chia sẻ. Mình ghi nhận các biểu hiện bạn mô tả{signal}. "
            "Mình chỉ có thể đưa ra thông tin tham khảo chứ không thể chẩn đoán chính xác "
            "qua tin nhắn, nên để yên tâm hơn, bạn nên đặt lịch khám để bác sĩ thú y kiểm "
            "tra trực tiếp cho bé nhé. Bạn có muốn đặt lịch ngay bây giờ không?"
        )

    return (
        "Mình có thể hỗ trợ thông tin chung về phòng khám, dịch vụ và chăm sóc thú cưng cơ "
        "bản. Bạn có thể xem giờ làm việc và danh sách dịch vụ trong ứng dụng, hoặc liên hệ "
        "trực tiếp phòng khám để được tư vấn chi tiết hơn. Nếu bé nhà bạn đang có biểu hiện "
        "bất thường, hãy mô tả cụ thể để mình hỗ trợ tốt hơn nhé."
    )


async def _llm_reply(request: ChatRequest, nlp_result: NlpResult, case: str) -> Optional[str]:
    settings = get_settings()
    if settings.LLM_PROVIDER == "none" or not settings.LLM_API_KEY:
        return None

    from app.services import llm_client  # lazy import, see nlp_triage.llm_extract for rationale

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        urgency_label="KHẨN CẤP" if case == "urgent" else "không khẩn cấp",
        keywords=", ".join(nlp_result.matched_keywords) if nlp_result.matched_keywords else "không có",
    )

    messages: List[dict] = []
    if request.history:
        messages.extend({"role": m.role, "content": m.content} for m in request.history)
    messages.append({"role": "user", "content": request.message})

    try:
        reply = await llm_client.complete(
            messages=messages,
            settings=settings,
            system_prompt=system_prompt,
            max_tokens=400,
        )
        return reply.strip() if reply and reply.strip() else None
    except Exception:  # noqa: BLE001 - must degrade gracefully to canned reply
        logger.warning("LLM chat reply generation failed; falling back to canned reply", exc_info=True)
        return None


async def run_chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid4())

    # Safety-critical: always the deterministic rule-based engine, never
    # the (optional, LLM-assisted) `nlp_triage.extract()` -- see module
    # docstring point 4.
    nlp_result = nlp_triage.rule_based_extract(request.message)
    case = _case_for(nlp_result)
    suggest_booking = case in ("urgent", "symptom")

    reply = await _llm_reply(request, nlp_result, case)
    if not reply:
        reply = _canned_reply(case, nlp_result)

    return ChatResponse(reply=reply, suggest_booking=suggest_booking, session_id=session_id)
