"""Load the images and videos attached to a case and package them for the vision model.

Everything goes to the vision model as an `image_url` JPEG data URL. The
OpenRouter vision model (DeepSeek V4.1 Flash) accepts text and images only, so
a video is sent as a few evenly spaced still frames - without its sound.
"""
import base64
import io
import logging
import os
import tempfile
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_IMAGES = 5
MAX_VIDEOS = 2
MAX_REMOTE_IMAGE_BYTES = 5 * 1024 * 1024
MAX_REMOTE_VIDEO_BYTES = 100 * 1024 * 1024

IMAGE_MAX_SIDE = 1568

VIDEO_FRAMES = 6


@dataclass
class MediaItem:
    label: str  # "Ảnh 1", "Video 1", ... - the vision model cites these
    kind: str  # "image" (video frames are images too)
    data_url: str


@dataclass
class MediaBundle:
    items: List[MediaItem] = field(default_factory=list)
    images: int = 0  # images that could be read
    videos: int = 0  # videos that could be read
    skipped: int = 0  # attachments that could not be loaded


def encode_image(file_bytes: bytes) -> Optional[str]:
    """EXIF-rotate and downscale an image, returning it as a JPEG data URL."""
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.load()
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
    except Exception as exc:
        logger.warning("Could not open or process image bytes: %s", exc)
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _trusted_bases() -> List[str]:
    """URL prefixes of the clinic's own media store (S3 via CloudFront), from MEDIA_BASE_URLS."""
    return [
        base.strip().rstrip("/").lower() + "/"
        for base in os.getenv("MEDIA_BASE_URLS", "").split(",")
        if base.strip().lower().startswith("https://")
    ]


def is_trusted_media_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    lowered = url.lower()
    return any(lowered.startswith(base) for base in _trusted_bases())


def download_media(url: str, max_bytes: int) -> Optional[bytes]:
    """Download a URL from a trusted media store, refusing anything over max_bytes."""
    if not is_trusted_media_url(url):
        logger.warning("Skipping media URL outside the trusted stores")
        return None

    limit_mb = max_bytes // (1024 * 1024)
    request = urllib.request.Request(url, headers={"User-Agent": "VetCare-AI/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if int(response.headers.get("Content-Length") or 0) > max_bytes:
            raise ValueError(f"Remote file exceeds the {limit_mb} MB limit")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Remote file exceeds the {limit_mb} MB limit")
    return data


def _image_bytes(value: str) -> Optional[bytes]:
    """Load an image given as a data URL, trusted store URL, local path or raw base64."""
    if value.startswith("data:image/"):
        return base64.b64decode(value.split(",", 1)[-1])
    if value.startswith(("http://", "https://")):
        return download_media(value, MAX_REMOTE_IMAGE_BYTES)
    if os.path.exists(value):
        with open(value, "rb") as file:
            return file.read()
    return base64.b64decode(value)


def _write_temp(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name


def _video_file(value: str) -> Tuple[Optional[str], str, bool]:
    """Resolve a video string to (path, extension, is_temporary)."""
    if value.startswith("https://"):
        ext = os.path.splitext(urlparse(value).path)[1].lower()
        if ext not in SUPPORTED_VIDEO_EXTS:
            logger.warning("Skipping unsupported remote video format: %s", ext or "unknown")
            return None, ext, False
        data = download_media(value, MAX_REMOTE_VIDEO_BYTES)
        return (_write_temp(data, ext), ext, True) if data else (None, ext, False)
    if os.path.exists(value):
        return value, os.path.splitext(value)[1].lower(), False
    if value.startswith("data:video/") or len(value) > 100:
        return _write_temp(base64.b64decode(value.split(",", 1)[-1]), ".mp4"), ".mp4", True
    return None, "", False


def _shrink(frame, max_side: int):
    """Resize a cv2 frame so its longer side is at most max_side (even dimensions)."""
    import cv2

    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    new_width = max(2, int(width * scale) // 2 * 2)
    new_height = max(2, int(height * scale) // 2 * 2)
    if (new_width, new_height) == (width, height):
        return frame
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _frame_indices(path: str, wanted_fps: float, max_frames: int) -> Tuple[List[int], float]:
    """Pick the frame indices to keep: wanted_fps over at most max_frames frames."""
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
    finally:
        cap.release()
    fps = fps if fps > 0 else 30.0
    step = max(1, round(fps / wanted_fps))
    if total <= 0:
        # Unknown length (some webm files): keep the first max_frames samples.
        return [i * step for i in range(max_frames)], fps
    indices = list(range(0, total, step))
    if len(indices) > max_frames:
        last = total - 1
        indices = sorted({round(i * last / (max_frames - 1)) for i in range(max_frames)})
    return indices, fps


def _read_frames(path: str, indices: List[int]):
    """Yield the frames at the given indices, decoding the video once."""
    import cv2

    wanted = set(indices)
    last = max(indices) if indices else -1
    cap = cv2.VideoCapture(path)
    try:
        index = 0
        while index <= last and cap.grab():
            if index in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    yield frame
            index += 1
    finally:
        cap.release()


def sample_frames(path: str, count: int = VIDEO_FRAMES) -> List[bytes]:
    """Sample `count` evenly spaced frames as JPEG bytes."""
    import cv2

    indices, _ = _frame_indices(path, 1000.0, count)
    frames = []
    for frame in _read_frames(path, indices):
        ok, buffer = cv2.imencode(".jpg", _shrink(frame, IMAGE_MAX_SIDE), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            frames.append(buffer.tobytes())
    return frames


def _video_items(path: str, ext: str, label: str) -> List[MediaItem]:
    """Package one video as evenly spaced still frames (the vision model reads no video)."""
    frames = [url for url in (encode_image(frame) for frame in sample_frames(path)) if url]
    return [
        MediaItem(f"{label} - khung hình {i}/{len(frames)}", "image", url)
        for i, url in enumerate(frames, start=1)
    ]


def collect_media(
    images: Optional[List[str]] = None,
    videos: Optional[List[str]] = None,
    image_files: Optional[List[Tuple[str, bytes]]] = None,
    video_files: Optional[List[Tuple[str, bytes]]] = None,
) -> MediaBundle:
    """Load every attachment of a case (blocking: run it in a worker thread)."""
    bundle = MediaBundle()

    def add_image(url: Optional[str]) -> None:
        if url:
            bundle.images += 1
            bundle.items.append(MediaItem(f"Ảnh {bundle.images}", "image", url))
        else:
            bundle.skipped += 1

    def add_video(path: Optional[str], ext: str) -> None:
        items = _video_items(path, ext, f"Video {bundle.videos + 1}") if path else []
        if items:
            bundle.videos += 1
            bundle.items.extend(items)
        else:
            bundle.skipped += 1

    temp_paths: List[str] = []
    try:
        for value in [v.strip() for v in (images or []) if isinstance(v, str) and v.strip()][:MAX_IMAGES]:
            try:
                data = _image_bytes(value)
            except Exception as exc:
                logger.warning("Could not load image input: %s", exc)
                data = None
            add_image(encode_image(data) if data else None)

        for filename, data in (image_files or [])[:MAX_IMAGES]:
            if os.path.splitext(filename)[1].lower() in SUPPORTED_IMAGE_EXTS:
                add_image(encode_image(data))
            else:
                logger.warning("Skipping unsupported image format: %s", filename)
                bundle.skipped += 1

        for value in [v.strip() for v in (videos or []) if isinstance(v, str) and v.strip()][:MAX_VIDEOS]:
            try:
                path, ext, is_temp = _video_file(value)
                if path and is_temp:
                    temp_paths.append(path)
                add_video(path, ext)
            except Exception as exc:
                logger.warning("Could not load video input: %s", exc)
                bundle.skipped += 1

        for filename, data in (video_files or [])[:MAX_VIDEOS]:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_VIDEO_EXTS:
                logger.warning("Skipping unsupported video format: %s", filename)
                bundle.skipped += 1
                continue
            path = _write_temp(data, ext)
            temp_paths.append(path)
            try:
                add_video(path, ext)
            except Exception as exc:
                logger.warning("Could not process video file %s: %s", filename, exc)
                bundle.skipped += 1
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass

    return bundle
