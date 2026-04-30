from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "data"
THUMB_DIR = DATA_DIR / "thumbs"
COVER_DIR = DATA_DIR / "covers"
ONLINE_COVER_DIR = DATA_DIR / "online_covers"
DB_PATH = DATA_DIR / "library.db"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

FEATURE_FIELDS = [
    "人物主体",
    "人物细节",
    "体型轮廓",
    "身材细节",
    "肤色观感",
    "发型发色",
    "五官表情",
    "服装配饰",
    "动作姿态",
    "拍摄角度",
    "场景背景",
    "画风质感",
]

DEFAULT_SELECTED_FIELDS = [
    "人物主体",
    "人物细节",
    "体型轮廓",
    "身材细节",
    "肤色观感",
    "发型发色",
    "五官表情",
    "服装配饰",
    "动作姿态",
    "画风质感",
]

DEFAULT_FIELD_WEIGHTS = {
    "人物主体": 1.2,
    "人物细节": 1.4,
    "体型轮廓": 1.5,
    "身材细节": 1.7,
    "肤色观感": 1.3,
    "发型发色": 1.2,
    "五官表情": 1.1,
    "服装配饰": 1.3,
    "动作姿态": 1.4,
    "拍摄角度": 0.8,
    "场景背景": 0.6,
    "画风质感": 1.0,
}

FEATURE_MODES = ["细", "粗"]
DEFAULT_FEATURE_MODE = "细"
SPEED_MODES = ["极速", "快速", "详细"]
DEFAULT_SPEED_MODE = "极速"

CAMIE_BACKEND = "Camie v2 标签"
QWEN_BACKEND = "Qwen3-VL v4.5 精描"
FEATURE_BACKENDS = [QWEN_BACKEND, CAMIE_BACKEND]
DEFAULT_FEATURE_BACKEND = os.environ.get("XP_FEATURE_BACKEND", QWEN_BACKEND)
if DEFAULT_FEATURE_BACKEND not in FEATURE_BACKENDS:
    DEFAULT_FEATURE_BACKEND = QWEN_BACKEND

CAMIE_MODEL_ID = os.environ.get("XP_CAMIE_MODEL_ID", "Camais03/camie-tagger-v2")
CAMIE_PROMPT_VERSION = "camie-v2-media-tags-zh-v4-direct-body-strict"
QWEN_MODEL_ID = os.environ.get("XP_QWEN_MODEL_ID", "Disty0/Qwen3-VL-8B-NSFW-Caption-V4.5")
QWEN_PROMPT_VERSION = "qwen3-vl-v45-zh-xp-v1-max512"
QWEN_MAX_PIXELS = int(os.environ.get("XP_QWEN_MAX_PIXELS", str(512 * 512)))
QWEN_LOAD_IN_4BIT = os.environ.get("XP_QWEN_LOAD_IN_4BIT", "1") == "1"

CACHE_MODEL_ID = f"{CAMIE_MODEL_ID}@onnxruntime-gpu"
MODEL_CACHE_DIR = Path(os.environ.get("XP_MODEL_CACHE_DIR", str(APP_ROOT / "models" / "hf-cache")))
PROMPT_VERSION = QWEN_PROMPT_VERSION
USE_STUB_MODEL = os.environ.get("XP_USE_STUB_MODEL", "0") == "1"

XP_FFMPEG = os.environ.get("XP_FFMPEG", "")
XP_FFPROBE = os.environ.get("XP_FFPROBE", "")
