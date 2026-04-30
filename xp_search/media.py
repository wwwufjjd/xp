from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps

from .config import (
    COVER_DIR,
    IMAGE_EXTENSIONS,
    MEDIA_EXTENSIONS,
    THUMB_DIR,
    VIDEO_EXTENSIONS,
    XP_FFMPEG,
    XP_FFPROBE,
)


KNOWN_FFMPEG = Path(r"D:\1\ffmpeg-6.1.1-essentials_build\bin\ffmpeg.exe")
KNOWN_FFPROBE = Path(r"D:\1\ffmpeg-6.1.1-essentials_build\bin\ffprobe.exe")


@dataclass(frozen=True)
class CoverInfo:
    media_type: str
    source_path: Path
    cover_path: Path
    cover_source: str
    title: str
    source_site: str = "本地"
    page_url: str = ""
    cover_url: str = ""


def is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def is_video_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def is_media_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS


def iter_image_files(folder: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(folder).expanduser()
    if not root.exists() or not root.is_dir():
        return []

    iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    return sorted((path.resolve() for path in iterator if is_image_path(path)), key=lambda p: str(p).lower())


def iter_media_files(folder: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(folder).expanduser()
    if not root.exists() or not root.is_dir():
        return []

    iterator: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    return sorted((path.resolve() for path in iterator if is_media_path(path)), key=lambda p: str(p).lower())


def uploaded_path(item: Any) -> Path | None:
    if item is None:
        return None
    if isinstance(item, (str, Path)):
        return Path(item)
    if isinstance(item, dict):
        value = item.get("name") or item.get("path")
        return Path(value) if value else None
    name = getattr(item, "name", None)
    return Path(name) if name else None


def collect_reference_media(
    uploaded_files: Any,
    folder_path: str | None,
    max_count: int,
) -> list[Path]:
    candidates: list[Path] = []

    if uploaded_files:
        files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
        for item in files:
            path = uploaded_path(item)
            if path and is_media_path(path):
                candidates.append(path.resolve())

    if folder_path and folder_path.strip():
        candidates.extend(iter_media_files(folder_path.strip(), recursive=True))

    seen: set[str] = set()
    result: list[Path] = []
    for path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
        if len(result) >= max(1, int(max_count)):
            break

    return result


def resolve_media_cover(path: str | Path) -> CoverInfo:
    source_path = Path(path).resolve()
    if is_image_path(source_path):
        return CoverInfo(
            media_type="image",
            source_path=source_path,
            cover_path=source_path,
            cover_source="image",
            title=source_path.name,
        )
    if is_video_path(source_path):
        cover_path, cover_source = resolve_video_cover(source_path)
        return CoverInfo(
            media_type="video",
            source_path=source_path,
            cover_path=cover_path,
            cover_source=cover_source,
            title=source_path.name,
        )
    raise ValueError(f"不支持的媒体格式：{source_path.suffix}")


def resolve_video_cover(video_path: Path) -> tuple[Path, str]:
    COVER_DIR.mkdir(parents=True, exist_ok=True)
    digest = media_digest(video_path)
    embedded_path = COVER_DIR / f"{digest}_embedded.jpg"
    frame_path = COVER_DIR / f"{digest}_frame.jpg"

    if embedded_path.exists() and embedded_path.stat().st_size > 0:
        return embedded_path, "embedded"
    if extract_attached_picture(video_path, embedded_path):
        return embedded_path, "embedded"

    if frame_path.exists() and frame_path.stat().st_size > 0:
        return frame_path, "frame"
    if extract_video_frame(video_path, frame_path):
        return frame_path, "frame"

    raise RuntimeError(f"无法从视频读取封面或抽帧：{video_path}")


def media_digest(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha1(f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")).hexdigest()


def ffmpeg_path() -> str:
    return executable_path(XP_FFMPEG, "ffmpeg", KNOWN_FFMPEG)


def ffprobe_path() -> str:
    return executable_path(XP_FFPROBE, "ffprobe", KNOWN_FFPROBE)


def executable_path(configured: str, name: str, known_path: Path) -> str:
    if configured and Path(configured).exists():
        return configured
    found = shutil.which(name)
    if found:
        return found
    if known_path.exists():
        return str(known_path)
    raise FileNotFoundError(f"找不到 {name}，可用 XP_{name.upper()} 环境变量指定路径。")


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        ffprobe_path(),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "stream=index,codec_type,disposition:format=duration",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"ffprobe 失败：{path}")
    return json.loads(completed.stdout or "{}")


def attached_picture_stream_index(path: Path) -> int | None:
    data = probe_video(path)
    for stream in data.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        disposition = stream.get("disposition") or {}
        if int(disposition.get("attached_pic") or 0) == 1:
            return int(stream["index"])
    return None


def extract_attached_picture(video_path: Path, output_path: Path) -> bool:
    try:
        stream_index = attached_picture_stream_index(video_path)
    except Exception:
        return False
    if stream_index is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-map",
        f"0:{stream_index}",
        "-frames:v",
        "1",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0


def extract_video_frame(video_path: Path, output_path: Path) -> bool:
    timestamp = fallback_timestamp(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0


def fallback_timestamp(video_path: Path) -> float:
    try:
        data = probe_video(video_path)
        duration = float((data.get("format") or {}).get("duration") or 0.0)
    except Exception:
        duration = 0.0
    if duration <= 0:
        return 1.0
    if duration <= 1.0:
        return max(duration * 0.1, 0.0)
    return min(max(duration * 0.1, 1.0), max(duration - 0.1, 0.0), 30.0)


def make_thumbnail(image_path: str | Path, max_size: int = 420) -> tuple[str, int, int]:
    path = Path(image_path)
    stat = path.stat()
    digest = hashlib.sha1(f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")).hexdigest()
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    thumb_path = THUMB_DIR / f"{digest}.jpg"

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        if not thumb_path.exists():
            thumb = image.copy()
            thumb.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            if thumb.mode in {"RGBA", "LA", "P"}:
                thumb = thumb.convert("RGBA")
                canvas = Image.new("RGBA", thumb.size, (255, 255, 255, 255))
                canvas.alpha_composite(thumb)
                thumb = canvas.convert("RGB")
            else:
                thumb = thumb.convert("RGB")
            thumb.save(thumb_path, format="JPEG", quality=88, optimize=True)

    return str(thumb_path), width, height
