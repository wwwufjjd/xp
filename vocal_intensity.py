"""
视频音频激烈度批量分析器

默认不使用模型，完整扫描视频音轨，按绝对音量、相对峰值、人声频带、
事件突起度和片段稳定性排序，并可截取候选高峰片段。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".wmv", ".flv", ".ts", ".rmvb"}
SR = 16000
FRAME_SEC = 0.75
HOP_SEC = 0.25
EVENT_MIN_SEC = 1.0
EVENT_MERGE_GAP_SEC = 1.25
MIN_F0 = 150
MAX_F0 = 800

KNOWN_FFMPEG = [
    Path(r"D:\xiazai\ffmpeg-2024-08-21-git-9d15fe77e3-full_build\bin\ffmpeg.exe"),
    Path(r"D:\1\ffmpeg-6.1.1-essentials_build\bin\ffmpeg.exe"),
]

LEVELS = [
    (0.00, "[--] 安静"),
    (0.08, "[低] 轻声"),
    (0.18, "[中] 中等"),
    (0.30, "[高] 激烈"),
    (0.48, "[烈] 很激烈"),
    (0.68, "[极] 极烈"),
]


@dataclass
class VideoResult:
    path: str
    filename: str
    duration_sec: float
    overall: float
    peak: float
    peak_time: float
    label: str
    loud_pct: float
    top_moments: list[dict[str, Any]] = field(default_factory=list)
    peak_features: dict[str, Any] = field(default_factory=dict)
    clip_path: str = ""
    thumb_path: str = ""
    engine: str = "signal"
    error: str = ""


def find_ffmpeg() -> str:
    candidates = [os.environ.get("XP_FFMPEG", ""), shutil.which("ffmpeg") or ""]
    candidates.extend(str(path) for path in KNOWN_FFMPEG)
    for item in candidates:
        if item and Path(item).exists():
            return item
    raise FileNotFoundError("找不到 ffmpeg。请安装 ffmpeg，或设置 XP_FFMPEG。")


def ffprobe_of(ffmpeg: str) -> str:
    ffmpeg_path = Path(ffmpeg)
    sibling = ffmpeg_path.parent / ffmpeg_path.name.replace("ffmpeg", "ffprobe")
    if sibling.exists():
        return str(sibling)
    return shutil.which("ffprobe") or str(sibling)


def get_duration(video: Path, ffmpeg: str) -> float:
    try:
        result = subprocess.run(
            [
                ffprobe_of(ffmpeg),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def extract_audio(video: Path, ffmpeg: str) -> np.ndarray:
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(SR),
            "-ac",
            "1",
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
        timeout=900,
    )
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"ffmpeg 提取音频失败: {result.stderr[:200]!r}")
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def rms_of(seg: np.ndarray) -> float:
    if len(seg) == 0:
        return 0.0
    return float(np.sqrt(np.mean(seg * seg)))


def dbfs_from_rms(value: float) -> float:
    return 20.0 * float(np.log10(max(value, 1e-9)))


def band_ratio(seg: np.ndarray, lo_hz: float, hi_hz: float) -> float:
    n = min(8192, len(seg))
    if n < 64:
        return 0.0
    windowed = seg[:n] * np.hanning(n)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    freq = np.fft.rfftfreq(n, 1.0 / SR)
    total = float(np.sum(power))
    if total < 1e-12:
        return 0.0
    mask = (freq >= lo_hz) & (freq <= hi_hz)
    return float(np.sum(power[mask]) / total)


def high_freq_ratio(seg: np.ndarray) -> float:
    return band_ratio(seg, 1000, SR / 2)


def zcr_of(seg: np.ndarray) -> float:
    if len(seg) < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(np.sign(seg))) > 0) / (len(seg) - 1))


def pitch_fft(seg: np.ndarray) -> float:
    n = len(seg)
    if n < SR // MIN_F0:
        return 0.0
    x = seg - seg.mean()
    fft_n = 1
    while fft_n < 2 * n:
        fft_n <<= 1
    spec = np.fft.rfft(x, n=fft_n)
    corr = np.fft.irfft(spec * np.conj(spec))[:n]
    lo = max(1, SR // MAX_F0)
    hi = min(n - 1, SR // MIN_F0)
    if lo >= hi or corr[0] <= 1e-10:
        return 0.0
    region = corr[lo : hi + 1]
    if float(region.max()) <= 1e-10:
        return 0.0
    peak = lo + int(np.argmax(region))
    if corr[peak] / corr[0] < 0.15:
        return 0.0
    return float(SR / peak)


def hnr_estimate(seg: np.ndarray) -> float:
    n = len(seg)
    if n < SR // 80:
        return 0.0
    x = seg - seg.mean()
    fft_n = 1
    while fft_n < 2 * n:
        fft_n <<= 1
    spec = np.fft.rfft(x, n=fft_n)
    corr = np.fft.irfft(spec * np.conj(spec))[:n]
    if corr[0] <= 1e-10:
        return 0.0
    lo = max(1, SR // MAX_F0)
    hi = min(n - 1, SR // MIN_F0)
    if lo >= hi:
        return 0.0
    return float(np.clip(np.max(corr[lo : hi + 1]) / corr[0], 0, 1))


def rhythmicity(seg: np.ndarray, frame_ms: int = 50) -> float:
    frame_len = SR * frame_ms // 1000
    n_frames = len(seg) // frame_len
    if n_frames < 10:
        return 0.0
    frames = seg[: n_frames * frame_len].reshape(n_frames, frame_len)
    energy = np.sqrt(np.mean(frames * frames, axis=1))
    energy -= energy.mean()
    std = float(np.std(energy))
    if std < 1e-6:
        return 0.0
    energy /= std
    corr = np.correlate(energy, energy, mode="full")
    corr = corr[len(corr) // 2 :]
    corr /= corr[0] + 1e-10
    min_lag = 3
    max_lag = min(40, len(corr) - 1)
    if min_lag >= max_lag:
        return 0.0
    return float(np.clip(np.max(corr[min_lag : max_lag + 1]), 0, 1))


def smooth_1d(values: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0 or len(values) <= radius * 2:
        return values.astype(np.float32)
    padded = np.pad(values.astype(np.float32), (radius, radius), mode="edge")
    kernel = np.ones(radius * 2 + 1, dtype=np.float32) / float(radius * 2 + 1)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def signal_frame_score(seg: np.ndarray, noise_db: float) -> tuple[float, dict[str, Any]]:
    rms_value = rms_of(seg)
    db = dbfs_from_rms(rms_value)
    voice_ratio = band_ratio(seg, 180, 4200)
    breath_ratio = band_ratio(seg, 900, 5200)
    hfr = high_freq_ratio(seg)
    zcr = zcr_of(seg)

    absolute = float(np.clip((db + 52.0) / 24.0, 0, 1))
    relative = float(np.clip((db - noise_db - 5.0) / 18.0, 0, 1))
    spectral = float(np.clip(0.70 * voice_ratio + 0.30 * min(breath_ratio / 0.55, 1.0), 0, 1))
    texture = float(np.clip((zcr - 0.015) / 0.12, 0, 1))
    score = absolute * (0.70 * relative + 0.20 * spectral * np.sqrt(relative) + 0.10 * texture * relative)

    if db < -48:
        score = 0.0
    elif relative < 0.08:
        score = min(score, 0.08)
    elif relative < 0.18:
        score = min(score, 0.18)
    elif db < -42:
        score = min(score, 0.22)
    elif db < -36:
        score = min(score, 0.55)
    if voice_ratio < 0.20 and breath_ratio < 0.18:
        score *= 0.45

    details = {
        "engine": "signal",
        "rms": round(rms_value, 5),
        "dbfs": round(db, 1),
        "noise_dbfs": round(noise_db, 1),
        "voice_ratio": round(voice_ratio, 3),
        "breath_ratio": round(breath_ratio, 3),
        "hfr": round(hfr, 3),
        "zcr": round(zcr, 4),
        "absolute": round(absolute, 3),
        "relative": round(relative, 3),
        "spectral": round(spectral, 3),
    }
    return float(np.clip(score, 0, 1)), details


def label_of(score: float) -> str:
    label = LEVELS[0][1]
    for threshold, item in LEVELS:
        if score >= threshold:
            label = item
    return label


def build_frames(audio: np.ndarray) -> list[tuple[int, int, float]]:
    frame = max(SR // 4, int(round(FRAME_SEC * SR)))
    hop = max(SR // 20, int(round(HOP_SEC * SR)))
    ranges: list[tuple[int, int, float]] = []
    start = 0
    while start < len(audio):
        end = min(len(audio), start + frame)
        if end - start >= SR // 4:
            ranges.append((start, end, (start + end) / (2 * SR)))
        if end == len(audio):
            break
        start += hop
    return ranges


def find_events(
    scores: np.ndarray,
    centers: list[float],
    details: list[dict[str, Any]],
    duration: float,
) -> list[dict[str, Any]]:
    if len(scores) == 0 or float(scores.max()) < 0.05:
        return []

    max_score = float(scores.max())
    enter = max(0.12, min(0.40, max_score * 0.45))
    exit_threshold = max(0.06, enter * 0.55)
    min_frames = max(1, int(round(EVENT_MIN_SEC / HOP_SEC)))

    raw: list[tuple[int, int]] = []
    start: int | None = None
    for idx, score in enumerate(scores):
        if score >= exit_threshold:
            if start is None:
                start = idx
        elif start is not None:
            raw.append((start, idx - 1))
            start = None
    if start is not None:
        raw.append((start, len(scores) - 1))

    filtered: list[tuple[int, int]] = []
    for a, b in raw:
        segment = scores[a : b + 1]
        if len(segment) >= min_frames and float(segment.max()) >= enter:
            filtered.append((a, b))

    merged: list[tuple[int, int]] = []
    for a, b in filtered:
        if not merged:
            merged.append((a, b))
            continue
        prev_a, prev_b = merged[-1]
        if centers[a] - centers[prev_b] <= EVENT_MERGE_GAP_SEC:
            merged[-1] = (prev_a, b)
        else:
            merged.append((a, b))

    events: list[dict[str, Any]] = []
    for a, b in merged:
        segment = scores[a : b + 1]
        peak_rel = int(np.argmax(segment))
        peak_idx = a + peak_rel
        peak = float(scores[peak_idx])
        mean = float(segment.mean())
        top_k = max(1, int(np.ceil(len(segment) * 0.25)))
        top_mean = float(np.sort(segment)[-top_k:].mean())
        contrast = float(np.clip((peak - mean) / 0.25, 0, 1))
        start_sec = max(0.0, centers[a] - FRAME_SEC / 2)
        end_sec = min(duration, centers[b] + FRAME_SEC / 2)
        event_dur = max(0.0, end_sec - start_sec)
        peak_details = dict(details[peak_idx])
        db = float(peak_details.get("dbfs", -60.0))

        base_score = 0.52 * peak + 0.20 * top_mean + 0.10 * mean + 0.18 * contrast
        duration_penalty = 1.0
        if event_dur > 45:
            duration_penalty = max(0.78, 1.0 - (event_dur - 45.0) / 180.0 * 0.22)
        stability_penalty = 1.0 if contrast >= 0.30 else 0.82 + 0.18 * (contrast / 0.30)
        loud_penalty = 1.0
        if db > -20.0:
            loud_penalty = max(0.86, 1.0 - (db + 20.0) / 8.0 * 0.20)
        event_score = float(np.clip(base_score * duration_penalty * stability_penalty * loud_penalty, 0, 1))

        events.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": event_dur,
                "peak_idx": peak_idx,
                "peak_time_sec": centers[peak_idx],
                "score": event_score,
                "peak": peak,
                "mean": mean,
                "top_mean": top_mean,
                "contrast": contrast,
                "duration_penalty": duration_penalty,
                "stability_penalty": stability_penalty,
                "loud_penalty": loud_penalty,
                "details": peak_details,
            }
        )

    events.sort(key=lambda item: (item["score"], item["peak"], item["duration_sec"]), reverse=True)
    return events


def adjust_event_texture(events: list[dict[str, Any]], audio: np.ndarray) -> None:
    for event in events:
        center = int(float(event["peak_time_sec"]) * SR)
        half = int(0.75 * SR / 2)
        seg = audio[max(0, center - half) : min(len(audio), center + half)]
        hnr = hnr_estimate(seg)
        f0 = pitch_fft(seg)
        rhythm = rhythmicity(seg)
        tonal_penalty = 1.0
        if hnr > 0.82:
            tonal_penalty = max(0.82, 1.0 - (hnr - 0.82) / 0.18 * 0.18)
        event["tonal_penalty"] = tonal_penalty
        event["score"] = float(np.clip(float(event["score"]) * tonal_penalty, 0, 1))
        event["details"]["hnr"] = round(hnr, 3)
        event["details"]["f0"] = round(f0, 1)
        event["details"]["rhythm"] = round(rhythm, 3)
    events.sort(key=lambda item: (item["score"], item["peak"], item["duration_sec"]), reverse=True)


def analyze_video(video_path: str, ffmpeg: str) -> VideoResult:
    path = Path(video_path)
    try:
        probe_duration = get_duration(path, ffmpeg)
        audio = extract_audio(path, ffmpeg)
        duration = round(probe_duration if probe_duration > 0 else len(audio) / SR, 1)
        if len(audio) < SR:
            raise RuntimeError("音频过短或为空")

        frames = build_frames(audio)
        if not frames:
            raise RuntimeError("未生成有效分析窗口")
        frame_rms = np.array([rms_of(audio[a:b]) for a, b, _ in frames], dtype=np.float32)
        db_values = np.array([dbfs_from_rms(float(v)) for v in frame_rms], dtype=np.float32)
        noise_db = float(np.percentile(db_values, 20))

        raw_scores: list[float] = []
        details: list[dict[str, Any]] = []
        centers: list[float] = []
        for a, b, center in frames:
            score, item = signal_frame_score(audio[a:b], noise_db)
            raw_scores.append(score)
            details.append(item)
            centers.append(center)

        scores = smooth_1d(np.array(raw_scores, dtype=np.float32), radius=2)
        active = scores[scores > 0.08]
        max_score = float(scores.max()) if len(scores) else 0.0
        events = find_events(scores, centers, details, duration)
        adjust_event_texture(events, audio)

        if not events and len(active) == 0:
            overall = 0.0
        elif events:
            event_scores = np.array([float(item["score"]) for item in events], dtype=np.float32)
            top_event_mean = float(event_scores[: min(3, len(event_scores))].mean())
            overall = 0.58 * float(event_scores[0]) + 0.24 * top_event_mean + 0.08 * float(active.mean() if len(active) else 0.0) + 0.10 * max_score
        else:
            top_k = max(1, int(np.ceil(len(active) * 0.10)))
            top_mean = float(np.sort(active)[-top_k:].mean())
            overall = 0.40 * max_score + 0.38 * top_mean + 0.22 * float(active.mean())
        if duration > 20 * 60:
            duration_min = duration / 60.0
            overall *= max(0.88, 1.0 - (duration_min - 20.0) / 80.0 * 0.12)
        overall = float(np.clip(overall, 0, 1))
        loud_pct = float(np.sum(scores >= 0.30) / len(scores)) if len(scores) else 0.0

        top_moments: list[dict[str, Any]] = []
        for event in events[:5]:
            peak_idx = int(event["peak_idx"])
            peak_time = float(event["peak_time_sec"])
            item = event["details"]
            start_sec = float(event["start_sec"])
            end_sec = float(event["end_sec"])
            top_moments.append(
                {
                    "time": f"{int(peak_time // 60)}:{peak_time % 60:04.1f}",
                    "time_sec": round(peak_time, 1),
                    "start_sec": round(start_sec, 1),
                    "end_sec": round(end_sec, 1),
                    "duration_sec": round(float(event["duration_sec"]), 1),
                    "clip_start_sec": round(max(0.0, max(start_sec, peak_time - 2.0)), 1),
                    "score": round(float(event["score"]), 3),
                    "peak": round(float(event["peak"]), 3),
                    "mean": round(float(event["mean"]), 3),
                    "contrast": round(float(event["contrast"]), 3),
                    "duration_penalty": round(float(event["duration_penalty"]), 3),
                    "stability_penalty": round(float(event["stability_penalty"]), 3),
                    "loud_penalty": round(float(event["loud_penalty"]), 3),
                    "tonal_penalty": round(float(event.get("tonal_penalty", 1.0)), 3),
                    "rms": round(float(item["rms"]), 5),
                    "classes": [
                        {"name": "dbFS", "prob": item["dbfs"]},
                        {"name": "voice_band", "prob": item["voice_ratio"]},
                        {"name": "relative", "prob": item["relative"]},
                        {"name": "HNR", "prob": item.get("hnr", 0)},
                    ],
                    "penalties": [] if scores[peak_idx] >= 0.05 else ["低于绝对音量/相对峰值门槛"],
                }
            )

        peak_idx = int(events[0]["peak_idx"]) if events else (int(np.argmax(scores)) if len(scores) else 0)
        peak_details = events[0]["details"] if events else (details[peak_idx] if details else {})
        return VideoResult(
            path=str(path),
            filename=path.name,
            duration_sec=duration,
            overall=round(overall, 3),
            peak=round(max_score, 3),
            peak_time=round(centers[peak_idx], 1) if centers else 0,
            label=label_of(overall),
            loud_pct=round(loud_pct, 3),
            top_moments=top_moments,
            peak_features=peak_details,
        )
    except Exception as exc:
        return VideoResult(str(path), path.name, 0, 0, 0, 0, "[X] 错误", 0, error=str(exc))


def collect_videos(target: str) -> list[Path]:
    path = Path(target)
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return [path.resolve()]
    if path.is_dir():
        return sorted(
            (
                item.resolve()
                for item in path.rglob("*")
                if item.is_file()
                and item.suffix.lower() in VIDEO_EXTENSIONS
                and "_clips" not in {part.lower() for part in item.relative_to(path).parts}
            ),
            key=lambda item: str(item).lower(),
        )
    return []


def safe_filename(name: str, max_len: int = 80) -> str:
    name = re.sub(r'[<>:"/\\|?*%#]', "_", name)
    name = re.sub(r"[\x00-\x1f]", "", name).strip(". ")
    if len(name) > max_len:
        stem, ext = os.path.splitext(name)
        name = stem[: max_len - len(ext)] + ext
    return name or "video"


def extract_clip(video_path: str, start_sec: float, duration: float, output_path: str, ffmpeg: str) -> bool:
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{max(0.0, start_sec):.1f}",
                "-i",
                video_path,
                "-t",
                f"{duration:.1f}",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                output_path,
            ],
            capture_output=True,
            check=False,
            timeout=120,
        )
        return result.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 0
    except Exception:
        return False


def extract_thumb(video_path: str, time_sec: float, output_path: str, ffmpeg: str) -> bool:
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{time_sec:.1f}",
                "-i",
                video_path,
                "-frames:v",
                "1",
                "-vf",
                "scale=320:-2",
                "-q:v",
                "3",
                output_path,
            ],
            capture_output=True,
            check=False,
            timeout=30,
        )
        return result.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 0
    except Exception:
        return False


def html_escape(value: Any) -> str:
    import html

    return html.escape(str(value), quote=True)


def generate_html_report(results: list[VideoResult], html_path: Path) -> None:
    import base64

    def img_data(path: str) -> str:
        try:
            encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
        except Exception:
            return ""

    def score_color(score: float) -> str:
        if score >= 0.68:
            return "#ff1744"
        if score >= 0.48:
            return "#ff5722"
        if score >= 0.30:
            return "#ff9800"
        if score >= 0.18:
            return "#ffc107"
        if score >= 0.08:
            return "#8bc34a"
        return "#9e9e9e"

    rows: list[str] = []
    for index, result in enumerate(results, start=1):
        if result.error:
            continue
        peak = result.top_moments[0] if result.top_moments else {}
        thumb = f'<img src="{img_data(result.thumb_path)}" style="width:160px;border-radius:6px;">' if result.thumb_path else ""
        clip = ""
        if result.clip_path and Path(result.clip_path).exists():
            clip = f'<a href="{Path(result.clip_path).name}" style="color:#4fc3f7;text-decoration:none;">播放片段</a>'
        sig = result.peak_features
        color = score_color(result.overall)
        dur = f"{result.duration_sec/60:.0f}分钟" if result.duration_sec >= 60 else f"{result.duration_sec:.0f}秒"
        rows.append(
            f"""
<tr>
  <td style="text-align:center;color:#aaa;">{index}</td>
  <td style="text-align:center;">{thumb}</td>
  <td>
    <div style="font-weight:600;color:#eee;margin-bottom:4px;word-break:break-all;">{html_escape(result.filename)}</div>
    <div style="color:#999;font-size:12px;">{dur} | 高分窗口 {result.loud_pct:.0%} | 引擎 signal</div>
    <div style="color:#888;font-size:12px;">最激烈: {peak.get('time', '')} (分数 {peak.get('score', 0):.2f})</div>
    <div style="color:#888;font-size:12px;">候选段: {peak.get('start_sec', 0):.1f}s - {peak.get('end_sec', 0):.1f}s | 峰值 {peak.get('peak', 0):.2f} | 均值 {peak.get('mean', 0):.2f}</div>
    <div style="color:#888;font-size:11px;">dbFS {sig.get('dbfs', 0)} | relative {sig.get('relative', 0)} | voice {sig.get('voice_ratio', 0)} | HNR {sig.get('hnr', 0)}</div>
    <div style="margin-top:4px;">{clip}</div>
  </td>
  <td style="text-align:center;"><div style="font-size:24px;font-weight:bold;color:{color};">{result.overall:.0%}</div><div style="font-size:11px;color:#aaa;">峰值 {result.peak:.0%}</div></td>
  <td style="text-align:center;color:{color};font-weight:600;">{result.label}</td>
</tr>"""
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>音频激烈度分析报告</title>
<style>
body {{ background:#1a1a2e; color:#eee; font-family:'Microsoft YaHei',sans-serif; padding:20px; }}
h1 {{ text-align:center; color:#e94560; margin-bottom:5px; }}
.subtitle {{ text-align:center; color:#888; margin-bottom:20px; font-size:14px; }}
table {{ width:100%; border-collapse:collapse; }}
th {{ background:#16213e; color:#e94560; padding:12px 8px; text-align:left; position:sticky; top:0; }}
td {{ padding:10px 8px; border-bottom:1px solid #2a2a4a; vertical-align:middle; }}
tr:hover {{ background:#16213e; }}
img {{ display:block; margin:0 auto; }}
</style>
</head>
<body>
<h1>音频激烈度分析报告</h1>
<div class="subtitle">共 {len([r for r in results if not r.error])} 个视频 | signal 完整扫描</div>
<table>
<tr><th style="width:40px;">#</th><th style="width:170px;text-align:center;">封面</th><th>标题 / 音频证据</th><th style="width:100px;text-align:center;">激烈度</th><th style="width:90px;text-align:center;">等级</th></tr>
{''.join(rows)}
</table>
</body>
</html>"""
    html_path.write_text(html, encoding="utf-8")


def print_progress(done: int, total: int, result: VideoResult, path: Path) -> None:
    if result.error:
        print(f"  [{done:>3}/{total}] ----  [X] {result.error[:60]}  {path.name}")
        return
    dur = f"{result.duration_sec/60:.0f}m" if result.duration_sec >= 60 else f"{result.duration_sec:.0f}s"
    print(f"  [{done:>3}/{total}] {result.overall:.2f}  {result.label}  ({dur})  {path.name}")


def print_summary(show: list[VideoResult], results: list[VideoResult], ok: int, failed: int, elapsed: float, output_file: str | None) -> None:
    print()
    print("=" * 100)
    print(f"  完成 | {len(results)}个 | 成功{ok} | 失败{failed} | {elapsed:.1f}s", end="")
    print(f" | {elapsed / ok:.2f}s/个" if ok > 0 else "")
    print("=" * 100)
    print()
    print(f"  {'#':>3}  {'综合':>5}  {'峰值':>5}  {'高分%':>5}  {'时长':>5}  {'等级':<10}  文件名")
    print("  " + "-" * 94)
    for index, result in enumerate(show, start=1):
        if result.error:
            print(f"  {index:>3}  ----   ----   ---   ----  [X]         {result.filename}")
            continue
        dur = f"{result.duration_sec/60:.0f}m" if result.duration_sec >= 60 else f"{result.duration_sec:.0f}s"
        print(f"  {index:>3}  {result.overall:.2f}   {result.peak:.2f}  {result.loud_pct:>4.0%}  {dur:>5}  {result.label:<10}  {result.filename}")
        if result.top_moments:
            peak = result.top_moments[0]
            print(f"       -> 最烈: {peak['time']} (分{peak['score']:.2f})")
            print(
                f"       候选段: {peak['start_sec']:.1f}s - {peak['end_sec']:.1f}s | "
                f"峰值={peak.get('peak', 0):.2f} | 均值={peak.get('mean', 0):.2f} | 突起={peak.get('contrast', 0):.2f}"
            )
        features = result.peak_features
        if features:
            print(
                f"       信号: dbFS={features.get('dbfs')} | relative={features.get('relative')} | "
                f"voice={features.get('voice_ratio')} | HNR={features.get('hnr')}"
            )
    print()
    if output_file:
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total": len(results),
            "ok": ok,
            "failed": failed,
            "elapsed": round(elapsed, 2),
            "videos": [asdict(item) for item in results],
        }
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [+] JSON: {out}")


def run_batch(
    target: str,
    output_file: str | None = None,
    top_n: int = 0,
    sort_by: str = "intensity",
    clip: bool = False,
    clip_sec: float = 10.0,
    clip_top: int = 1,
    min_score: float = 0.3,
    engine: str = "signal",
) -> list[VideoResult]:
    ffmpeg = find_ffmpeg()
    videos = collect_videos(target)
    if not videos:
        print(f"[!] 未找到视频: {target}")
        return []

    if engine not in {"auto", "signal"}:
        print(f"[!] 当前版本不再使用 {engine} 模型，已改用 signal。")
    print(f"[*] {len(videos)} 个视频 | 引擎 signal")
    print(f"    ffmpeg: {ffmpeg}")
    print(f"    signal: 无模型 / {FRAME_SEC:g}s帧 / {HOP_SEC:g}s步进 / 绝对音量+相对峰值+语音频带")
    if clip:
        print(f"    截取高峰片段: {clip_sec:.0f}s | 每视频前{clip_top}段 | 最低分: {min_score}")
    print()

    started = time.time()
    results: list[VideoResult] = []
    for done, video in enumerate(videos, start=1):
        result = analyze_video(str(video), ffmpeg)
        results.append(result)
        print_progress(done, len(videos), result, video)

    failed = sum(1 for item in results if item.error)
    ok = len(results) - failed
    sort_keys = {
        "intensity": lambda item: item.overall,
        "max": lambda item: item.peak,
        "loud_ratio": lambda item: item.loud_pct,
        "name": lambda item: item.filename.lower(),
    }
    results.sort(key=sort_keys.get(sort_by, sort_keys["intensity"]), reverse=(sort_by != "name"))

    clip_dir: Path | None = None
    if clip:
        clip_dir = Path(target)
        if clip_dir.is_file():
            clip_dir = clip_dir.parent
        clip_dir = clip_dir / "_clips"
        clip_dir.mkdir(exist_ok=True)
        thumb_dir = clip_dir / "thumbs"
        thumb_dir.mkdir(exist_ok=True)
        print()
        print(f"[*] 截取高峰片段到: {clip_dir}")
        count = 0
        for index, result in enumerate(results):
            if result.error or result.overall < min_score or not result.top_moments:
                continue
            safe_stem = safe_filename(Path(result.filename).stem, 50)
            pct = int(result.overall * 100)
            for rank, peak in enumerate(result.top_moments[: max(1, clip_top)], start=1):
                peak_time = float(peak["time_sec"])
                clip_start = float(peak.get("clip_start_sec", peak_time))
                time_text = f"{int(peak_time // 60)}m{peak_time % 60:04.1f}s"
                out_path = str(clip_dir / f"[{pct}]_[top{rank}]_[{time_text}]_{safe_stem}.mp4")
                if extract_clip(result.path, clip_start, clip_sec, out_path, ffmpeg):
                    if rank == 1:
                        result.clip_path = out_path
                    count += 1
                    print(f"    [{pct}%] top{rank} {peak['time']} <- {result.filename}")
            thumb_path = str(thumb_dir / f"{index:03d}_{safe_stem}.jpg")
            if extract_thumb(result.path, float(result.top_moments[0]["time_sec"]), thumb_path, ffmpeg):
                result.thumb_path = thumb_path
        report_path = clip_dir / "report.html"
        generate_html_report(results, report_path)
        print(f"  截取完成: {count} 个片段")
        print(f"  HTML报告: {report_path}")

    elapsed = time.time() - started
    show = results[:top_n] if top_n > 0 else results
    print_summary(show, results, ok, failed, elapsed, output_file)
    if clip and clip_dir:
        print(f"\n  [*] 高峰片段目录: {clip_dir}")
    return results


def main() -> None:
    if len(sys.argv) == 1:
        print("=" * 56)
        print("  视频音频激烈度分析器")
        print("  signal | 无模型 | 绝对音量+相对峰值+语音频带")
        print("=" * 56)
        print()
        target = input("视频文件/目录: ").strip().strip('"\'')
        if not target:
            return
        clip = input("截取高峰片段? (y/回车跳过): ").strip().lower() in {"y", "yes", "是"}
        output = input("JSON输出 (回车跳过): ").strip().strip('"\'')
        run_batch(target=target, output_file=output or None, clip=clip)
        input("\n回车退出...")
        return

    parser = argparse.ArgumentParser(
        description="视频音频激烈度分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
示例:
  python vocal_intensity.py "D:\videos"
  python vocal_intensity.py "D:\videos" -o result.json
  python vocal_intensity.py "D:\videos" --clip --clip-top 3 --clip-sec 10
""",
    )
    parser.add_argument("target", help="视频文件/目录")
    parser.add_argument("-o", "--output", type=str, default=None, help="JSON输出")
    parser.add_argument("-t", "--top", type=int, default=0, help="只显示前N个")
    parser.add_argument("--sort", choices=["intensity", "max", "loud_ratio", "name"], default="intensity")
    parser.add_argument("--clip", action="store_true", help="截取每个视频最激烈的片段")
    parser.add_argument("--clip-sec", type=float, default=10.0, help="截取片段长度(秒, 默认10)")
    parser.add_argument("--clip-top", type=int, default=1, help="每个视频截取前N个候选高峰段(默认1)")
    parser.add_argument("--min-score", type=float, default=0.3, help="截取最低分数阈值(默认0.3)")
    parser.add_argument("--engine", choices=["auto", "signal", "clap", "yamnet", "light"], default="signal", help="兼容旧参数；当前统一使用 signal")
    parser.add_argument("-w", "--workers", type=int, default=1, help="兼容旧参数，signal 引擎不使用多进程")
    parser.add_argument("--seg-sec", type=float, default=15.0, help="兼容旧参数，signal 引擎不使用")
    parser.add_argument("--hop-sec", type=float, default=5.0, help="兼容旧参数，signal 引擎不使用")
    args = parser.parse_args()
    run_batch(
        target=args.target,
        output_file=args.output,
        top_n=args.top,
        sort_by=args.sort,
        clip=args.clip,
        clip_sec=args.clip_sec,
        clip_top=args.clip_top,
        min_score=args.min_score,
        engine=args.engine,
    )


if __name__ == "__main__":
    main()
