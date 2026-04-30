from __future__ import annotations

import re
import threading
from pathlib import Path

from .config import (
    CAMIE_BACKEND,
    FEATURE_BACKENDS,
    MODEL_CACHE_DIR,
    QWEN_BACKEND,
    QWEN_LOAD_IN_4BIT,
    QWEN_MAX_PIXELS,
    QWEN_MODEL_ID,
    QWEN_PROMPT_VERSION,
    USE_STUB_MODEL,
)
from .features import format_feature_map, normalize_feature_conflicts, parse_feature_text
from .tagger import CamieOnnxFeatureExtractor


class StubFeatureExtractor:
    """Fast deterministic extractor for tests and UI smoke checks."""

    model_id = "stub"
    prompt_version = "stub-tags-v2"

    def extract(self, image_path: str | Path, mode: str = "细", speed: str = "快速") -> str:
        stem = Path(image_path).stem.lower()
        fields = {
            "人物主体": "单人、清晰人像",
            "人物细节": "单人主体、清晰面部、自然姿态、人物居中",
            "体型轮廓": "纤细轮廓" if "slim" in stem or "thin" in stem else "普通体型",
            "身材细节": "腰线清晰、四肢线条清晰" if "slim" in stem or "thin" in stem else "自然体态",
            "肤色观感": "白皙肤色" if "white" in stem or "fair" in stem else "自然肤色",
            "发型发色": "长发、深色头发" if "long" in stem else "发型清晰",
            "五官表情": "微笑、正脸" if "smile" in stem else "自然表情",
            "服装配饰": "浅色服装" if "white" in stem or "fair" in stem else "日常服装",
            "动作姿态": "站姿" if "stand" in stem else "自然姿态",
            "拍摄角度": "正面视角",
            "场景背景": "室内" if "indoor" in stem else "普通背景",
            "画风质感": "自拍风格" if "selfie" in stem else "清晰人像",
        }
        if mode == "粗":
            fields = {field: "、".join(value.split("、")[:2]) for field, value in fields.items()}
        return format_feature_map(fields)


def normalize_model_output(output: str, mode: str = "细") -> str:
    output = strip_wrappers(output)
    parsed = parse_feature_text(output)
    if any(value.strip() for value in parsed.values()):
        return format_feature_map(parsed)

    compact = re.sub(r"\s+", " ", output).strip()
    if not compact:
        return format_feature_map({})
    if mode == "粗":
        compact = "、".join(compact.split("、")[:8])
    return format_feature_map({"人物主体": compact})


def feature_detail_score(feature_text: str) -> int:
    parsed = parse_feature_text(feature_text)
    score = 0
    for value in parsed.values():
        cleaned = value.strip()
        if not cleaned or cleaned == "无":
            continue
        score += 1
        score += len([term for term in re.split(r"[、，,；;\n]+", cleaned) if term.strip()])
    return score


def is_sparse_feature_text(feature_text: str) -> bool:
    parsed = parse_feature_text(feature_text)
    non_empty_fields = 0
    term_count = 0
    for value in parsed.values():
        cleaned = value.strip()
        if not cleaned or cleaned == "无":
            continue
        non_empty_fields += 1
        term_count += len([term for term in re.split(r"[、，,；;\n]+", cleaned) if term.strip()])
    return non_empty_fields < 6 or term_count < 16


def strip_wrappers(output: str) -> str:
    output = output.strip()
    output = re.sub(r"^```(?:json|text|markdown)?\s*", "", output, flags=re.IGNORECASE)
    output = re.sub(r"\s*```$", "", output)
    return output.strip()


def limit_feature_terms(feature_text: str, mode: str) -> str:
    if mode != "粗":
        return feature_text

    parsed = parse_feature_text(feature_text)
    trimmed: dict[str, str] = {}
    for field, value in parsed.items():
        terms = [term.strip() for term in re.split(r"[、，,；;\n]+", value) if term.strip() and term.strip() != "无"]
        trimmed[field] = "、".join(terms[:4]) if terms else "无"
    return format_feature_map(trimmed)


class QwenVlFeatureExtractor:
    model_id = f"{QWEN_MODEL_ID}@transformers-bnb4" if QWEN_LOAD_IN_4BIT else f"{QWEN_MODEL_ID}@transformers-fp16"
    prompt_version = QWEN_PROMPT_VERSION

    def __init__(self, model_id: str = QWEN_MODEL_ID) -> None:
        self.repo_id = model_id
        self.model_id = f"{model_id}@transformers-bnb4" if QWEN_LOAD_IN_4BIT else f"{model_id}@transformers-fp16"
        self.prompt_version = QWEN_PROMPT_VERSION
        self._model = None
        self._processor = None
        self._process_vision_info = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None and self._processor is not None and self._process_vision_info is not None:
            return

        with self._lock:
            if self._model is not None and self._processor is not None and self._process_vision_info is not None:
                return

            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            quantization_config = None
            if QWEN_LOAD_IN_4BIT:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

            self._processor = AutoProcessor.from_pretrained(
                self.repo_id,
                cache_dir=MODEL_CACHE_DIR,
                trust_remote_code=True,
            )
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.repo_id,
                cache_dir=MODEL_CACHE_DIR,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                attn_implementation="eager",
                trust_remote_code=True,
            ).eval()
            self._process_vision_info = process_vision_info

    def extract(self, image_path: str | Path, mode: str = "细", speed: str = "详细") -> str:
        self._load()
        assert self._model is not None
        assert self._processor is not None
        assert self._process_vision_info is not None

        import torch

        path = Path(image_path)
        prompt = self._build_prompt(mode)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(path), "max_pixels": QWEN_MAX_PIXELS},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = self._vision_inputs(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(self._model.device)

        max_new_tokens = 180 if mode == "粗" else 320
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        normalized = normalize_model_output(output, mode=mode)
        normalized = limit_feature_terms(normalized, mode=mode)
        return normalize_feature_conflicts(normalized)

    def _vision_inputs(self, messages: list[dict[str, object]]) -> tuple[object, object, dict[str, object]]:
        try:
            image_inputs, video_inputs, video_kwargs = self._process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            return image_inputs, video_inputs, video_kwargs
        except TypeError:
            image_inputs, video_inputs = self._process_vision_info(messages)
            return image_inputs, video_inputs, {}

    @staticmethod
    def _build_prompt(mode: str) -> str:
        term_count = "每个字段最多 4 个核心短标签" if mode == "粗" else "每个字段尽量给 6 到 12 个可见短标签"
        return f"""Return a concise Chinese field list for visible character visual features only.
Do not infer age. Do not include moral commentary. Do not describe hidden body parts.
{term_count}，用中文顿号分隔。人物和身材优先，背景次要。
Output exactly these fields, one per line:
人物主体：
人物细节：
体型轮廓：
身材细节：
肤色观感：
发型发色：
五官表情：
服装配饰：
动作姿态：
拍摄角度：
场景背景：
画风质感：

Focus on visible body silhouette, body proportions, waist/abdomen, legs, hips, visible chest contour, skin tone, hair, outfit, pose, camera angle, scene, and image style. If a field is uncertain, write 无."""


_extractors: dict[str, object] = {}
_extractor_lock = threading.Lock()


def normalize_feature_backend(feature_backend: str | None) -> str:
    return feature_backend if feature_backend in FEATURE_BACKENDS else QWEN_BACKEND


def cache_speed_for_backend(feature_backend: str | None, speed: str) -> str:
    if normalize_feature_backend(feature_backend) == QWEN_BACKEND:
        return "Qwen精描"
    return "标签"


def get_extractor(feature_backend: str | None = None) -> object:
    backend = "stub" if USE_STUB_MODEL else normalize_feature_backend(feature_backend)
    if backend in _extractors:
        return _extractors[backend]
    with _extractor_lock:
        if backend not in _extractors:
            if USE_STUB_MODEL:
                _extractors[backend] = StubFeatureExtractor()
            elif backend == QWEN_BACKEND:
                _extractors[backend] = QwenVlFeatureExtractor()
            else:
                _extractors[backend] = CamieOnnxFeatureExtractor()
    return _extractors[backend]
