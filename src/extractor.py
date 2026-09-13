import base64
import io
import json
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from src.db import fetch_symptoms
from src.schemas import CompiledExtractionOutput, PetInfoInput

load_dotenv()
logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class DeepSeekSymptomExtractor:
    """Pet symptom compiler and triage extractor using DeepSeek V4.1-Flash served via OpenRouter."""

    def __init__(
        self,
        max_video_frames: int = 5,
        max_image_side: int = 2048,
    ):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing from .env")

        default_headers = {}
        if os.getenv("OPENROUTER_REFERER"):
            default_headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERER")
        if os.getenv("OPENROUTER_TITLE"):
            default_headers["X-OpenRouter-Title"] = os.getenv("OPENROUTER_TITLE")

        self.client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENROUTER_BASE_URL"),
            default_headers=default_headers or None,
        )
        self.model_name = os.getenv("OPENROUTER_MODEL")
        self.max_video_frames = max_video_frames
        self.max_image_side = max_image_side

        # Fetch symptoms from database
        self._symptoms_cache: Optional[List[Dict[str, Any]]] = None

    def get_symptoms_context(self) -> List[Dict[str, Any]]:
        """Fetch and cache symptoms from PostgreSQL database."""
        if self._symptoms_cache is None:
            try:
                self._symptoms_cache = fetch_symptoms()
            except Exception as e:
                logger.error(f"Failed to fetch symptoms from PostgreSQL: {e}")
                # Fallback to empty list or static if DB is temporarily unreachable
                self._symptoms_cache = []
        return self._symptoms_cache

    def _build_system_instruction(self) -> str:
        symptoms = self.get_symptoms_context()
        symptoms_str = json.dumps(symptoms, ensure_ascii=False, indent=2)

        return f"""
Bạn là một chuyên gia thú y và chuyên gia phân tích triệu chứng thú cưng (Pet Triage & Symptom Extractor).
Nhiệm vụ của bạn là tiếp nhận thông tin thú cưng (loài, giống, giới tính, cân nặng, tuổi), các triệu chứng người dùng cung cấp ban đầu, đoạn văn mô tả lâm sàng, cùng hình ảnh hoặc video (nếu có), sau đó:

1. CHUẨN HÓA THÔNG TIN THÚ CƯNG (`pet-info`):
- `breed`: Chuẩn hóa loài/giống sang mã: "BR001" (nếu là Chó), "BR002" (nếu là Mèo).
- `gender`: "M" (đực / male), "F" (cái / female).
- `weight`: Phân loại thể trạng cân nặng:
  * "W01": Nhẹ cân (Underweight)
  * "W02": Bình thường (Normal)
  * "W03": Thừa cân / Béo phì (Overweight)
- `age`: Phân loại độ tuổi:
  * "A01": Nhỏ / con non (dưới 1 tuổi)
  * "A02": Trưởng thành (từ 1 đến 7 tuổi)
  * "A03": Già / cao tuổi (trên 7 tuổi)

2. TỔNG HỢP VÀ ĐÁNH GIÁ MỨC ĐỘ TRIỆU CHỨNG (`symptoms`):
- Kết hợp toàn diện: danh sách mã triệu chứng input người dùng đã chọn + triệu chứng trích xuất từ văn bản mô tả (`describe`) + triệu chứng quan sát được qua ảnh và video.
- Đối với mỗi triệu chứng phát hiện được, gán mức độ nghiêm trọng `intensity`:
  * 1: Nhẹ (Mild)
  * 2: Vừa (Moderate)
  * 3: Nặng (Severe)
- CHỈ ĐƯỢC DÙNG mã `symptom_id` (ví dụ: "SY001", "SY004", "SY008", ...) có trong DANH MỤC TRIỆU CHỨNG dưới đây. Tuyệt đối không tự bịa mã mới.

3. PHÂN LOẠI CẤP CỨU (`triage_result`):
- `color_code`: Đánh giá thang màu cấp cứu thú y:
  * "RED": Cực kỳ nguy kịch / Cấp cứu tối khẩn (đe dọa tính mạng ngay lập tức: co giật nặng, khó thở tím tái, sùi bọt mép, xuất huyết cấp, sốt sữa cấp...).
  * "ORANGE": Nguy cơ cao / Khẩn cấp (mệt lả nặng, đau đớn dữ dội, mất nước nghiêm trọng...).
  * "YELLOW": Bán khẩn (cần thăm khám trong ngày: nôn tiêu chảy nhẹ, sốt vừa, mẩn ngứa...).
  * "GREEN": Ít khẩn cấp / Ổn định (triệu chứng nhẹ, tiến triển mạn tính).
  * "BLUE": Không khẩn cấp (chăm sóc thông thường).
- `reasoning`: Giải thích ngắn gọn lý do phân loại cấp cứu bằng tiếng Việt (ví dụ: "Phát hiện triệu chứng co giật mức độ cao").

=== DANH MỤC TRIỆU CHỨNG HỆ THỐNG (TỪ DATABASE) ===
{symptoms_str}
=================================================

QUY TẮC ĐẦU RA:
- Trả về DUY NHẤT một chuỗi JSON hợp lệ đúng chuẩn theo định dạng mẫu sau (phải chứa từ khóa json):
{{
  "pet-info": {{
    "breed": "BR001",
    "gender": "F",
    "weight": "W02",
    "age": "A02"
  }},
  "symptoms": [
    {{
      "symptom": "SY004",
      "intensity": 1
    }},
    {{
      "symptom": "SY008",
      "intensity": 3
    }}
  ],
  "triage_result": {{
    "color_code": "RED",
    "reasoning": "Phát hiện triệu chứng co giật mức độ cao"
  }}
}}
"""

    def _encode_image(self, file_bytes: bytes) -> Optional[str]:
        """Convert image bytes to a base64 JPEG data URL."""
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                img.load()
                img = img.convert("RGB")
                if max(img.size) > self.max_image_side:
                    ratio = self.max_image_side / max(img.size)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            logger.warning(f"Could not open or process image bytes: {e}")
            return None

    def _extract_video_frames(self, video_path: str) -> List[str]:
        """Sample up to max_video_frames frames from video and return as base64 JPEG strings."""
        import cv2

        frames = []
        cap = cv2.VideoCapture(video_path)
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            max_frames = self.max_video_frames
            if total > 0:
                step = max(1, total // max_frames)
                indices = {i * step for i in range(max_frames)} | {total - 1}
            else:
                indices = set(range(max_frames))

            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in indices and len(frames) < max_frames:
                    ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok_enc:
                        frames.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
                idx += 1
        finally:
            cap.release()
        return frames

    async def process_inputs(
        self,
        pet_info: PetInfoInput,
        initial_symptoms: Optional[List[str]] = None,
        description: Optional[str] = None,
        images: Optional[List[str]] = None,
        videos: Optional[List[str]] = None,
        image_files: Optional[List[Tuple[str, bytes]]] = None,
        video_files: Optional[List[Tuple[str, bytes]]] = None,
    ) -> CompiledExtractionOutput:
        """
        Process multimodal inputs through OpenRouter DeepSeek V4.1-Flash and return CompiledExtractionOutput (data-02.jsonc).
        """
        content_parts: List[Dict[str, Any]] = []
        temp_paths: List[str] = []

        try:
            # 1. Add Pet Info & initial symptoms & text description
            pet_info_dict = pet_info.model_dump()
            input_text_summary = {
                "pet-info": pet_info_dict,
                "selected_symptoms": initial_symptoms or [],
                "describe": description or "",
            }
            content_parts.append({
                "type": "text",
                "text": f"Dữ liệu đầu vào ca bệnh:\n{json.dumps(input_text_summary, ensure_ascii=False, indent=2)}"
            })

            # 2. Add Images from string list (base64 data URIs, URLs, or local paths)
            if images:
                for img_item in images[:5]:
                    if not img_item or not isinstance(img_item, str):
                        continue
                    img_str = img_item.strip()
                    if img_str.startswith("data:image/") or img_str.startswith("http://") or img_str.startswith("https://"):
                        content_parts.append({"type": "image_url", "image_url": {"url": img_str}})
                    elif os.path.exists(img_str):
                        try:
                            with open(img_str, "rb") as f:
                                data_url = self._encode_image(f.read())
                                if data_url:
                                    content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
                        except Exception as e:
                            logger.warning(f"Error reading image from path {img_str}: {e}")
                    else:
                        # Raw base64 string
                        content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}})

            # 3. Add Images from multipart files (max 5)
            if image_files:
                for filename, img_bytes in image_files[:5]:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in SUPPORTED_IMAGE_EXTS:
                        data_url = self._encode_image(img_bytes)
                        if data_url:
                            content_parts.append({"type": "image_url", "image_url": {"url": data_url}})
                    else:
                        logger.warning(f"Skipping unsupported image format: {filename}")

            # 4. Add Videos from string list (local paths or base64)
            if videos:
                for vid_item in videos[:2]:
                    if not vid_item or not isinstance(vid_item, str):
                        continue
                    vid_str = vid_item.strip()
                    if os.path.exists(vid_str):
                        frames = self._extract_video_frames(vid_str)
                        for frame_b64 in frames:
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                            })
                    elif vid_str.startswith("data:video/") or len(vid_str) > 100:
                        # Decode base64 to temp file
                        try:
                            raw_b64 = vid_str.split(",", 1)[-1]
                            raw_bytes = base64.b64decode(raw_b64)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                                tmp.write(raw_bytes)
                                tmp_path = tmp.name
                                temp_paths.append(tmp_path)
                            frames = self._extract_video_frames(tmp_path)
                            for frame_b64 in frames:
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                                })
                        except Exception as e:
                            logger.warning(f"Error decoding video base64: {e}")

            # 5. Add Videos from multipart files (max 2)
            if video_files:
                for filename, vid_bytes in video_files[:2]:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in SUPPORTED_VIDEO_EXTS:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            tmp.write(vid_bytes)
                            tmp_path = tmp.name
                            temp_paths.append(tmp_path)
                        frames = self._extract_video_frames(tmp_path)
                        for frame_b64 in frames:
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                            })
                    else:
                        logger.warning(f"Skipping unsupported video format: {filename}")

            messages = [
                {"role": "system", "content": self._build_system_instruction()},
                {"role": "user", "content": content_parts},
            ]

            request_kwargs = dict(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=3500,
                extra_body={"reasoning": {"effort": "low"}},
            )

            logger.info(f"Calling OpenRouter model {self.model_name}...")
            response = self.client.chat.completions.create(**request_kwargs)

            choice = response.choices[0]
            content = choice.message.content

            # Fallback if model placed answer in reasoning or choice.message
            if not content:
                reasoning = getattr(choice.message, "reasoning", "") or ""
                m = re.search(r"(\{.*\})", reasoning, re.DOTALL)
                if m:
                    content = m.group(1)

            if not content:
                raise ValueError("DeepSeek returned an empty response.")

            # Clean markdown code blocks if any
            clean_content = content.strip()
            if clean_content.startswith("```"):
                clean_content = re.sub(r"^```[a-zA-Z]*\n?", "", clean_content)
                clean_content = re.sub(r"\n?```$", "", clean_content).strip()

            logger.info("DeepSeek returned output. Validating into CompiledExtractionOutput...")
            return CompiledExtractionOutput.model_validate_json(clean_content)

        finally:
            for p in temp_paths:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass