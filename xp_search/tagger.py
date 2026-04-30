from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .config import CAMIE_MODEL_ID, CAMIE_PROMPT_VERSION, FEATURE_FIELDS, MODEL_CACHE_DIR
from .features import normalize_feature_conflicts


class CamieOnnxFeatureExtractor:
    model_id = f"{CAMIE_MODEL_ID}@onnxruntime-gpu"
    prompt_version = CAMIE_PROMPT_VERSION

    def __init__(self, model_id: str = CAMIE_MODEL_ID) -> None:
        self.repo_id = model_id
        self.model_id = f"{model_id}@onnxruntime-gpu"
        self.prompt_version = CAMIE_PROMPT_VERSION
        self._session = None
        self._input_name = ""
        self._idx_to_tag: dict[str, str] = {}
        self._tag_to_category: dict[str, str] = {}
        self._image_size = 512
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._session is not None:
            return

        with self._lock:
            if self._session is not None:
                return

            import onnxruntime as ort
            import torch  # noqa: F401 - importing torch makes CUDA/cuDNN DLLs visible to ONNX Runtime on Windows.
            from huggingface_hub import hf_hub_download

            try:
                ort.preload_dlls()
            except Exception:
                pass

            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            model_path = hf_hub_download(self.repo_id, "camie-tagger-v2.onnx", cache_dir=MODEL_CACHE_DIR)
            metadata_path = hf_hub_download(self.repo_id, "camie-tagger-v2-metadata.json", cache_dir=MODEL_CACHE_DIR)
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            self._idx_to_tag = metadata["dataset_info"]["tag_mapping"]["idx_to_tag"]
            self._tag_to_category = metadata["dataset_info"]["tag_mapping"]["tag_to_category"]
            self._image_size = int(metadata.get("model_info", {}).get("img_size", 512))

            providers: list[Any] = [
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ]
            session_options = ort.SessionOptions()
            session_options.log_severity_level = 3
            self._session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
            self._input_name = self._session.get_inputs()[0].name

    def extract(self, image_path: str | Path, mode: str = "细", speed: str = "极速") -> str:
        self._load()
        assert self._session is not None

        path = Path(image_path)
        threshold = 0.50 if mode == "粗" else 0.45
        top_k = 50 if mode == "粗" else 100
        blob = preprocess_imagenet_square(path, self._image_size)
        outputs = self._session.run(None, {self._input_name: blob})
        logits = outputs[1] if len(outputs) >= 2 else outputs[0]
        prediction = ensure_probability(logits[0])
        tags = pick_camie_tags(
            prediction,
            self._idx_to_tag,
            self._tag_to_category,
            threshold=threshold,
            top_k=top_k,
        )
        skin_terms = estimate_visible_skin_tone(path)
        return normalize_feature_conflicts(tags_to_chinese_features(tags, mode=mode, skin_terms=skin_terms))


def preprocess_imagenet_square(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = image_size / max(width, height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", (image_size, image_size), (124, 116, 104))
    padded.paste(image, ((image_size - new_width) // 2, (image_size - new_height) // 2))
    array = np.asarray(padded).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1))[None, ...].astype("float32")


def ensure_probability(output: np.ndarray) -> np.ndarray:
    output = output.astype("float32")
    if float(output.min()) < 0.0 or float(output.max()) > 1.0:
        output = 1.0 / (1.0 + np.exp(-output))
    return output


def pick_camie_tags(
    probs: np.ndarray,
    idx_to_tag: dict[str, str],
    tag_to_category: dict[str, str],
    threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    indexed: list[tuple[float, str]] = []
    for index, score in enumerate(probs.tolist()):
        tag = idx_to_tag.get(str(index))
        if not tag or tag_to_category.get(tag, "general") != "general" or should_drop_tag(tag):
            continue
        min_score = STRICT_TAG_THRESHOLDS.get(tag, threshold)
        if float(score) < min_score:
            continue
        if score >= threshold:
            indexed.append((float(score), tag))

    indexed.sort(reverse=True, key=lambda item: item[0])
    selected = indexed[:top_k]
    selected = resolve_conflicting_person_tags(selected)
    return [{"tag": tag, "score": round(score, 4)} for score, tag in selected]


FEMALE_PERSON_TAGS = {"1girl", "2girls", "multiple_girls"}
MALE_PERSON_TAGS = {"1boy", "2boys", "multiple_boys"}

STRICT_TAG_THRESHOLDS = {
    "breasts": 0.72,
    "flat_chest": 0.80,
    "small_breasts": 0.80,
    "medium_breasts": 0.80,
    "large_breasts": 0.82,
    "huge_breasts": 0.84,
    "skinny": 0.58,
    "petite": 0.58,
    "slim_legs": 0.58,
    "narrow_waist": 0.62,
    "long_legs": 0.58,
    "curvy": 0.62,
    "hourglass": 0.68,
    "wide_hips": 0.62,
    "thick_thighs": 0.62,
}


def resolve_conflicting_person_tags(selected: list[tuple[float, str]]) -> list[tuple[float, str]]:
    scores = {tag: score for score, tag in selected}
    female_score = max((scores[tag] for tag in FEMALE_PERSON_TAGS if tag in scores), default=0.0)
    male_score = max((scores[tag] for tag in MALE_PERSON_TAGS if tag in scores), default=0.0)
    if not female_score or not male_score or abs(female_score - male_score) < 0.15:
        return selected
    drop_tags = MALE_PERSON_TAGS if female_score > male_score else FEMALE_PERSON_TAGS
    return [(score, tag) for score, tag in selected if tag not in drop_tags]


DROP_TAGS = {
    "artist_name",
    "web_address",
    "patreon_username",
    "signature",
    "watermark",
    "text",
    "censored",
    "uncensored",
    "female_pov",
}

DROP_CONTAINS = (
    "loli",
    "shota",
    "child",
    "pussy",
    "penis",
    "dildo",
    "sex_toy",
    "cum",
)


def should_drop_tag(tag: str) -> bool:
    if tag in DROP_TAGS:
        return True
    return any(part in tag for part in DROP_CONTAINS)


TAG_ZH = {
    "1girl": "女性",
    "1boy": "男性",
    "2girls": "多名女性",
    "multiple_girls": "多名女性",
    "2boys": "多名男性",
    "multiple_boys": "多名男性",
    "solo": "单人",
    "solo_focus": "单人主体",
    "full_body": "全身可见",
    "upper_body": "上半身构图",
    "cowboy_shot": "大腿以上构图",
    "portrait": "肖像构图",
    "close-up": "近景特写",
    "selfie": "自拍",
    "phone": "手机",
    "cellphone": "手机",
    "smartphone": "智能手机",
    "iphone": "手机",
    "mirror": "镜前自拍",
    "full-length_mirror": "全身镜",
    "reflection": "镜面反射",
    "reflection_focus": "镜面构图",
    "looking_at_mirror": "看向镜子",
    "looking_at_viewer": "看向镜头",
    "looking_at_phone": "看向手机",
    "standing": "站立姿态",
    "sitting": "坐姿",
    "lying": "躺姿",
    "kneeling": "跪姿",
    "squatting": "蹲姿",
    "leaning_forward": "身体前倾",
    "leaning_back": "身体后仰",
    "holding": "手持物品",
    "holding_phone": "手持手机",
    "pulling_own_clothes": "拉衣服",
    "clothes_pull": "拉衣服",
    "clothes_lift": "提起衣服",
    "lifting_own_clothes": "提起衣服",
    "dress_lift": "提起裙摆",
    "indoors": "室内环境",
    "outdoors": "室外",
    "bedroom": "卧室场景",
    "bed": "床",
    "on_bed": "床边/床上",
    "pillow": "枕头",
    "curtains": "窗帘",
    "carpet": "地毯",
    "rug": "地毯",
    "poster_(object)": "墙上海报",
    "short_hair": "短发造型",
    "long_hair": "长发造型",
    "very_long_hair": "超长发",
    "medium_hair": "中长发",
    "pink_hair": "粉色头发",
    "white_hair": "白色头发",
    "blonde_hair": "金发",
    "grey_hair": "灰色头发",
    "red_hair": "红色头发",
    "blue_hair": "蓝色头发",
    "black_hair": "黑色头发",
    "brown_hair": "棕色头发",
    "purple_hair": "紫色头发",
    "hair_bun": "发髻",
    "ponytail": "马尾",
    "twintails": "双马尾",
    "ahoge": "呆毛",
    "straight_hair": "直发",
    "wavy_hair": "卷发",
    "hair_ornament": "发饰",
    "blue_eyes": "蓝色眼睛",
    "brown_eyes": "棕色眼睛",
    "green_eyes": "绿色眼睛",
    "red_eyes": "红色眼睛",
    "yellow_eyes": "黄色眼睛",
    "purple_eyes": "紫色眼睛",
    "blush": "脸红",
    "smile": "微笑",
    "light_smile": "浅笑",
    "open_mouth": "张嘴表情",
    "closed_mouth": "闭嘴表情",
    "closed_eyes": "闭眼",
    "parted_lips": "嘴唇微张",
    "underwear": "内衣穿搭",
    "panties": "内裤可见",
    "white_panties": "白色内裤",
    "blue_panties": "蓝色内裤",
    "shirt": "衬衫",
    "white_shirt": "白色衬衫",
    "long_sleeves": "长袖",
    "short_sleeves": "短袖",
    "jacket": "外套",
    "school_uniform": "制服风格",
    "dress": "连衣裙",
    "white_dress": "白色连衣裙",
    "bra": "胸衣",
    "white_bra": "白色胸衣",
    "camisole": "吊带背心",
    "crop_top": "短上衣",
    "skirt": "短裙",
    "miniskirt": "迷你裙",
    "lingerie": "内衣套装",
    "babydoll": "薄纱睡裙",
    "see-through": "透视材质",
    "see-through_clothes": "透视服装",
    "bare_shoulders": "肩部裸露",
    "collarbone": "锁骨可见",
    "bare_arms": "手臂裸露",
    "cleavage": "胸口线条可见",
    "sideboob": "侧胸线条可见",
    "underboob": "下胸线条可见",
    "ribbon": "丝带",
    "white_ribbon": "白色丝带",
    "leg_ribbon": "腿部丝带装饰",
    "thigh_ribbon": "大腿丝带装饰",
    "thigh_strap": "大腿绑带",
    "thighhighs": "长筒袜",
    "hat": "帽子",
    "gloves": "手套",
    "bow": "蝴蝶结",
    "hair_ribbon": "发带",
    "jewelry": "饰品",
    "ring": "戒指",
    "toe_ring": "脚趾环",
    "earrings": "耳饰",
    "barefoot": "赤脚状态",
    "feet": "脚部可见",
    "toes": "脚趾可见",
    "toenails": "脚趾甲可见",
    "toenail_polish": "脚趾甲油",
    "nail_polish": "指甲油",
    "blue_nails": "蓝色指甲",
    "thighs": "大腿可见",
    "thigh_gap": "大腿间隙",
    "legs": "腿部可见",
    "bare_legs": "裸腿可见",
    "ass": "臀部可见",
    "long_legs": "长腿比例",
    "slim_legs": "腿部线条偏细",
    "figure": "身形轮廓清楚",
    "skinny": "偏瘦体态",
    "petite": "小巧体型",
    "tall": "高挑体型",
    "tall_female": "高挑女性",
    "narrow_waist": "腰线纤细",
    "midriff": "腰腹裸露",
    "midriff_peek": "腰腹微露",
    "belly": "腹部可见",
    "stomach": "腹部可见",
    "navel": "腰腹和肚脐可见",
    "breasts": "胸部线条可见",
    "flat_chest": "胸部平坦",
    "small_breasts": "胸部偏小",
    "medium_breasts": "胸部中等",
    "large_breasts": "胸部偏大",
    "huge_breasts": "胸部很大",
    "wide_hips": "臀胯较宽",
    "curvy": "曲线感明显",
    "hourglass": "沙漏型轮廓",
    "thick_thighs": "大腿偏丰满",
    "colored_skin": "特殊肤色",
    "blue_skin": "蓝色肤色",
    "pale_skin": "白皙肤色",
    "simple_background": "简单背景",
    "white_background": "白色背景",
    "3d": "3D质感",
    "comic": "漫画质感",
    "monochrome": "单色画风",
    "greyscale": "灰度画风",
    "cosplay": "角色扮演",
    "realistic": "写实质感",
}

FIELD_RULES = {
    "人物主体": {"1girl", "1boy", "2girls", "multiple_girls", "solo", "solo_focus", "full_body", "upper_body"},
    "体型轮廓": {
        "breasts",
        "curvy",
        "figure",
        "flat_chest",
        "full_body",
        "hourglass",
        "long_legs",
        "narrow_waist",
        "petite",
        "skinny",
        "slim_legs",
        "small_breasts",
        "medium_breasts",
        "large_breasts",
        "huge_breasts",
        "tall",
        "tall_female",
        "wide_hips",
    },
    "身材细节": {
        "bare_shoulders",
        "bare_arms",
        "bare_legs",
        "collarbone",
        "cleavage",
        "legs",
        "thighs",
        "thigh_gap",
        "feet",
        "toes",
        "barefoot",
        "ass",
        "midriff",
        "midriff_peek",
        "belly",
        "stomach",
        "navel",
        "leg_ribbon",
        "thigh_ribbon",
        "thigh_strap",
        "figure",
        "skinny",
        "petite",
        "long_legs",
        "slim_legs",
        "narrow_waist",
        "wide_hips",
        "curvy",
        "hourglass",
        "thick_thighs",
    },
    "肤色观感": {"colored_skin", "blue_skin", "pale_skin"},
    "发型发色": {
        "short_hair",
        "long_hair",
        "very_long_hair",
        "medium_hair",
        "pink_hair",
        "white_hair",
        "blonde_hair",
        "grey_hair",
        "red_hair",
        "blue_hair",
        "black_hair",
        "brown_hair",
        "purple_hair",
        "hair_bun",
        "hair_ornament",
        "ponytail",
        "twintails",
        "ahoge",
        "straight_hair",
        "wavy_hair",
    },
    "五官表情": {
        "looking_at_viewer",
        "looking_at_mirror",
        "looking_at_phone",
        "blue_eyes",
        "brown_eyes",
        "green_eyes",
        "red_eyes",
        "yellow_eyes",
        "purple_eyes",
        "blush",
        "smile",
        "light_smile",
        "open_mouth",
        "closed_mouth",
        "closed_eyes",
        "parted_lips",
    },
    "服装配饰": {
        "underwear",
        "panties",
        "white_panties",
        "blue_panties",
        "shirt",
        "white_shirt",
        "long_sleeves",
        "short_sleeves",
        "jacket",
        "school_uniform",
        "bra",
        "white_bra",
        "camisole",
        "crop_top",
        "dress",
        "white_dress",
        "skirt",
        "miniskirt",
        "lingerie",
        "babydoll",
        "see-through",
        "see-through_clothes",
        "bare_shoulders",
        "ribbon",
        "white_ribbon",
        "leg_ribbon",
        "thigh_ribbon",
        "thigh_strap",
        "thighhighs",
        "hat",
        "gloves",
        "bow",
        "hair_ribbon",
        "jewelry",
        "ring",
        "toe_ring",
        "earrings",
    },
    "动作姿态": {
        "selfie",
        "holding_phone",
        "holding",
        "standing",
        "sitting",
        "lying",
        "kneeling",
        "squatting",
        "leaning_forward",
        "leaning_back",
        "looking_at_mirror",
        "looking_at_phone",
        "pulling_own_clothes",
        "clothes_pull",
        "clothes_lift",
        "lifting_own_clothes",
        "dress_lift",
    },
    "拍摄角度": {
        "full_body",
        "upper_body",
        "cowboy_shot",
        "portrait",
        "close-up",
        "selfie",
        "mirror",
        "full-length_mirror",
        "reflection",
        "reflection_focus",
    },
    "场景背景": {
        "indoors",
        "outdoors",
        "bedroom",
        "bed",
        "on_bed",
        "pillow",
        "curtains",
        "carpet",
        "rug",
        "poster_(object)",
        "mirror",
        "simple_background",
        "white_background",
    },
    "画风质感": {"3d", "comic", "monochrome", "greyscale", "cosplay", "realistic"},
}


def tags_to_chinese_features(
    tags: list[dict[str, Any]],
    mode: str = "细",
    skin_terms: list[str] | None = None,
) -> str:
    tag_names = [item["tag"] for item in tags if not should_drop_tag(item["tag"])]
    details_limit = 8 if mode == "粗" else 14
    field_limits = 5 if mode == "粗" else 12
    lines: list[str] = []

    for field in FEATURE_FIELDS:
        if field == "人物细节":
            detail_tags = [tag for tag in tag_names if tag not in CHEST_TAGS]
            values = translated_values(detail_tags, set(detail_tags), limit=details_limit)
        elif field == "肤色观感":
            values = [term for term in (skin_terms or []) if term]
            values.extend(translated_values(tag_names, FIELD_RULES.get(field, set()), limit=field_limits))
            values = list(dict.fromkeys(values))[:field_limits]
        elif field == "发型发色":
            values = translated_hair_values(tag_names, limit=field_limits)
        elif field in {"体型轮廓", "身材细节"}:
            values = translated_values(tag_names, FIELD_RULES.get(field, set()), limit=field_limits)
        else:
            values = translated_values(tag_names, FIELD_RULES.get(field, set()), limit=field_limits)
        lines.append(f"{field}：{'、'.join(values) if values else '无'}")

    return "\n".join(lines)


HAIR_COLOR_TAGS = {
    "black_hair",
    "blonde_hair",
    "blue_hair",
    "brown_hair",
    "grey_hair",
    "pink_hair",
    "purple_hair",
    "red_hair",
    "white_hair",
}

CHEST_TAGS = {"breasts", "flat_chest", "small_breasts", "medium_breasts", "large_breasts", "huge_breasts"}

HAIR_STYLE_TAGS = {
    "ahoge",
    "hair_bun",
    "hair_ornament",
    "long_hair",
    "medium_hair",
    "ponytail",
    "short_hair",
    "straight_hair",
    "twintails",
    "very_long_hair",
    "wavy_hair",
}


def translated_hair_values(tag_names: list[str], limit: int) -> list[str]:
    colors = translated_values(tag_names, HAIR_COLOR_TAGS, limit=2)
    styles = translated_values(tag_names, HAIR_STYLE_TAGS, limit=max(1, limit - len(colors)))
    return merged_values(styles, colors, limit=limit)


def merged_values(priority_values: list[str], values: list[str], limit: int) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*priority_values, *values]:
        if not value or value in seen:
            continue
        merged.append(value)
        seen.add(value)
        if len(merged) >= limit:
            break
    return merged


def translated_values(tag_names: list[str], allowed: set[str], limit: int) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for tag in tag_names:
        if tag not in allowed:
            continue
        value = TAG_ZH.get(tag)
        if not value or value in seen:
            continue
        values.append(value)
        seen.add(value)
        if len(values) >= limit:
            break
    return values


def estimate_visible_skin_tone(path: Path) -> list[str]:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((512, 512), Image.Resampling.BILINEAR)
    array = np.asarray(image).astype("float32")
    if array.size == 0:
        return []

    red = array[..., 0]
    green = array[..., 1]
    blue = array[..., 2]
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = 128 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    cr = 128 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    max_channel = np.maximum.reduce([red, green, blue])
    min_channel = np.minimum.reduce([red, green, blue])

    skin_mask = (
        (y > 70)
        & (cr > 132)
        & (cr < 188)
        & (cb > 72)
        & (cb < 150)
        & (red > green * 0.86)
        & (red > blue * 0.92)
        & ((max_channel - min_channel) > 8)
    )
    skin_ratio = float(np.count_nonzero(skin_mask)) / float(skin_mask.size)
    if skin_ratio < 0.006:
        return []

    skin_y = y[skin_mask]
    skin_cr = cr[skin_mask]
    skin_cb = cb[skin_mask]
    tone = float(np.percentile(skin_y, 85))
    warmth = float(np.mean(skin_cr - skin_cb))

    terms: list[str] = []
    if tone >= 205:
        terms.append("白皙肤色")
    elif tone >= 178:
        terms.append("浅肤色")
    elif tone >= 125:
        terms.append("自然肤色")
    else:
        terms.append("偏深肤色")

    if warmth >= 38:
        terms.append("偏暖肤色")
    elif warmth <= 18:
        terms.append("偏冷肤色")
    return terms
