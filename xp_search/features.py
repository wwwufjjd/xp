from __future__ import annotations

import math
import re
from collections import Counter, OrderedDict
from typing import Iterable

from .config import DEFAULT_FIELD_WEIGHTS, FEATURE_FIELDS

FIELD_ALIASES = {
    "人物": "人物主体",
    "主体": "人物主体",
    "人物类型": "人物主体",
    "person": "人物主体",
    "subject": "人物主体",
    "外观": "人物主体",
    "人物细节": "人物细节",
    "细节": "人物细节",
    "人物描写": "人物细节",
    "整体细节": "人物细节",
    "detail": "人物细节",
    "details": "人物细节",
    "身材": "身材细节",
    "身材细节": "身材细节",
    "身体细节": "身材细节",
    "身体特征": "身材细节",
    "身体比例": "身材细节",
    "体态细节": "身材细节",
    "比例": "身材细节",
    "线条": "身材细节",
    "bodydetails": "身材细节",
    "bodyproportions": "身材细节",
    "proportions": "身材细节",
    "figure": "身材细节",
    "体型": "体型轮廓",
    "體型": "体型轮廓",
    "身形": "体型轮廓",
    "体型轮廳": "体型轮廓",
    "體型輪廓": "体型轮廓",
    "body": "体型轮廓",
    "bodytype": "体型轮廓",
    "肤色": "肤色观感",
    "皮肤": "肤色观感",
    "肤色观感": "肤色观感",
    "skin": "肤色观感",
    "发型": "发型发色",
    "发色": "发型发色",
    "头发": "发型发色",
    "hair": "发型发色",
    "表情": "五官表情",
    "五官": "五官表情",
    "脸部": "五官表情",
    "face": "五官表情",
    "expression": "五官表情",
    "服装": "服装配饰",
    "穿搭": "服装配饰",
    "衣着": "服装配饰",
    "配饰": "服装配饰",
    "clothing": "服装配饰",
    "outfit": "服装配饰",
    "accessories": "服装配饰",
    "动作": "动作姿态",
    "姿态": "动作姿态",
    "动作/姿态": "动作姿态",
    "动作姿态": "动作姿态",
    "pose": "动作姿态",
    "action": "动作姿态",
    "拍摄": "拍摄角度",
    "角度": "拍摄角度",
    "构图": "拍摄角度",
    "camera": "拍摄角度",
    "angle": "拍摄角度",
    "场景": "场景背景",
    "背景": "场景背景",
    "环境": "场景背景",
    "scene": "场景背景",
    "background": "场景背景",
    "风格": "画风质感",
    "画风": "画风质感",
    "质感": "画风质感",
    "摄影风格": "画风质感",
    "style": "画风质感",
    "quality": "画风质感",
}

EMPTY_TERMS = {"", "无", "没有", "不明显", "无法判断", "未知", "看不清", "none", "unknown", "n/a"}
SPLIT_RE = re.compile(r"[,\n，、；;|/]+")
WORD_RE = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+")

COMMON_TERM_TRANSLATIONS = [
    (re.compile(r"\bslender\b", re.IGNORECASE), "纤细"),
    (re.compile(r"\bslim\b", re.IGNORECASE), "纤细"),
    (re.compile(r"\bthin\b", re.IGNORECASE), "偏瘦"),
    (re.compile(r"\bfair skin\b", re.IGNORECASE), "白皙肤色"),
    (re.compile(r"\bfair\b", re.IGNORECASE), "白皙"),
    (re.compile(r"\bpale skin\b", re.IGNORECASE), "白皙肤色"),
    (re.compile(r"\blight skin\b", re.IGNORECASE), "浅肤色"),
    (re.compile(r"\blong hair\b", re.IGNORECASE), "长发"),
    (re.compile(r"\bshort hair\b", re.IGNORECASE), "短发"),
    (re.compile(r"\bblack hair\b", re.IGNORECASE), "黑发"),
    (re.compile(r"\bbrown hair\b", re.IGNORECASE), "棕发"),
    (re.compile(r"\bblonde hair\b", re.IGNORECASE), "金发"),
    (re.compile(r"\bwhite hair\b", re.IGNORECASE), "白发"),
    (re.compile(r"\bsmile\b", re.IGNORECASE), "微笑"),
    (re.compile(r"\bsmiling\b", re.IGNORECASE), "微笑"),
    (re.compile(r"\bstanding\b", re.IGNORECASE), "站姿"),
    (re.compile(r"\bsitting\b", re.IGNORECASE), "坐姿"),
    (re.compile(r"\bselfie\b", re.IGNORECASE), "自拍"),
    (re.compile(r"\bindoor\b", re.IGNORECASE), "室内"),
    (re.compile(r"\bindoors\b", re.IGNORECASE), "室内"),
    (re.compile(r"\boutdoor\b", re.IGNORECASE), "室外"),
    (re.compile(r"\boutdoors\b", re.IGNORECASE), "室外"),
    (re.compile(r"\banime\b", re.IGNORECASE), "二次元"),
    (re.compile(r"\billustration\b", re.IGNORECASE), "插画"),
]

BREAST_SIZE_TERMS = {"胸部平坦", "胸部偏小", "胸部中等", "胸部偏大", "胸部很大"}
CHEST_TERMS = BREAST_SIZE_TERMS | {"胸部线条可见"}
LEG_SIZE_TERMS = {"腿部线条偏细", "四肢线条偏细", "大腿线条丰满"}
FIELD_DROP_TERMS = {
    "人物细节": CHEST_TERMS,
}
BODY_SHAPE_ALIASES = {
    "曲线感明显": "曲线轮廓明显",
    "身形轮廓清楚": "身形轮廓明显",
}
FIELD_TERM_ALIASES = {
    "体型轮廓": BODY_SHAPE_ALIASES,
}


def empty_feature_map() -> OrderedDict[str, str]:
    return OrderedDict((field, "") for field in FEATURE_FIELDS)


def normalize_field_name(name: str) -> str | None:
    cleaned = re.sub(r"[\s:：\-_*#()\[\]【】]+", "", name).lower()
    if cleaned in FEATURE_FIELDS:
        return cleaned
    return FIELD_ALIASES.get(cleaned)


def parse_feature_text(text: str | None) -> OrderedDict[str, str]:
    fields = empty_feature_map()
    if not text:
        return fields

    current_field: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("-*# ")
        if not line:
            continue
        match = re.match(r"^([^:：]{1,24})[:：]\s*(.*)$", line)
        if match:
            field = normalize_field_name(match.group(1))
            if field:
                current_field = field
                value = clean_value(match.group(2))
                if value and normalize_term(value) not in EMPTY_TERMS:
                    fields[field] = merge_text(fields[field], value)
                continue
            current_field = None
            continue
        if current_field:
            fields[current_field] = merge_text(fields[current_field], clean_value(line))

    return fields


def clean_value(value: str) -> str:
    value = value.strip().strip("-*# ")
    value = value.replace("\ufffd", "")
    value = re.sub(r"^['\"`]+|['\"`]+$", "", value)
    value = re.sub(r"\s+", " ", value)
    for pattern, replacement in COMMON_TERM_TRANSLATIONS:
        value = pattern.sub(replacement, value)
    return value


def format_feature_map(fields: dict[str, str]) -> str:
    return "\n".join(f"{field}：{fields.get(field, '').strip() or '无'}" for field in FEATURE_FIELDS)


def normalize_feature_conflicts(feature_text: str) -> str:
    fields = parse_feature_text(feature_text)
    normalized = empty_feature_map()
    for field in FEATURE_FIELDS:
        terms = split_terms(fields.get(field, ""))
        drop_terms = FIELD_DROP_TERMS.get(field, set())
        terms = [term for term in terms if term not in drop_terms]
        if not terms:
            continue
        counter = Counter(terms)
        first_seen = {term: index for index, term in enumerate(terms)}
        counter, first_seen = normalize_counter_terms(field, counter, first_seen)
        ordered_terms = sorted(counter, key=lambda term: (first_seen[term], term))
        ordered_terms = resolve_field_conflicts(field, ordered_terms, counter)
        normalized[field] = "、".join(ordered_terms)
    return format_feature_map(normalized)


def merge_text(existing: str, value: str) -> str:
    existing = existing.strip()
    value = value.strip()
    if not existing:
        return value
    if not value:
        return existing
    return f"{existing}、{value}"


def split_terms(value: str) -> list[str]:
    terms: list[str] = []
    for chunk in SPLIT_RE.split(value or ""):
        term = normalize_term(chunk)
        if term and term not in EMPTY_TERMS:
            terms.append(term)
    return terms


def normalize_term(term: str) -> str:
    term = term.strip().lower()
    term = re.sub(r"^[\-*#\s]+|[\-*#\s]+$", "", term)
    term = re.sub(r"\s+", "", term)
    return term


def text_tokens(value: str) -> set[str]:
    value = normalize_term(value)
    if not value:
        return set()

    tokens = set(split_terms(value))
    for word in WORD_RE.findall(value):
        word = normalize_term(word)
        if not word or word in EMPTY_TERMS:
            continue
        tokens.add(word)
        if len(word) >= 4:
            tokens.update(word[i : i + 2] for i in range(len(word) - 1))
    return tokens


def display_terms(value: str) -> set[str]:
    direct_terms = set(split_terms(value))
    if direct_terms:
        return direct_terms
    return {normalize_term(word) for word in WORD_RE.findall(value) if normalize_term(word)}


def merge_feature_profiles(
    feature_texts: Iterable[str],
    max_terms_per_field: int = 12,
    mode: str = "细",
) -> str:
    parsed_profiles = [parse_feature_text(text) for text in feature_texts if text and text.strip()]
    merged = empty_feature_map()
    limit = 4 if mode == "粗" else max_terms_per_field

    for field in FEATURE_FIELDS:
        counter: Counter[str] = Counter()
        first_seen: dict[str, int] = {}
        for index, profile in enumerate(parsed_profiles):
            for term in split_terms(profile.get(field, "")):
                counter[term] += 1
                first_seen.setdefault(term, index)

        if not counter:
            continue

        counter, first_seen = normalize_counter_terms(field, counter, first_seen)
        repeated = [term for term, count in counter.items() if count >= 2]
        repeated_sorted = sorted(repeated, key=lambda term: (-counter[term], first_seen[term], term))
        unique_sorted = sorted(
            [term for term in counter if term not in repeated],
            key=lambda term: (first_seen[term], -counter[term], term),
        )
        field_limit = limit
        if mode != "粗" and field in {"人物细节", "身材细节", "服装配饰"}:
            field_limit = max(field_limit, 14)
        selected = [*repeated_sorted, *unique_sorted]
        selected = resolve_field_conflicts(field, selected, counter)
        merged[field] = "、".join(selected[:field_limit])

    return format_feature_map(merged)


def normalize_counter_terms(
    field: str,
    counter: Counter[str],
    first_seen: dict[str, int],
) -> tuple[Counter[str], dict[str, int]]:
    aliases = FIELD_TERM_ALIASES.get(field, {})
    if not aliases:
        return counter, first_seen

    normalized: Counter[str] = Counter()
    normalized_first_seen: dict[str, int] = {}
    drop_terms = FIELD_DROP_TERMS.get(field, set())
    for term, count in counter.items():
        if term in drop_terms:
            continue
        canonical = aliases.get(term, term)
        normalized[canonical] += count
        normalized_first_seen[canonical] = min(first_seen.get(term, 0), normalized_first_seen.get(canonical, first_seen.get(term, 0)))
    return normalized, normalized_first_seen


def resolve_field_conflicts(field: str, terms: list[str], counter: Counter[str]) -> list[str]:
    if field not in {"体型轮廓", "身材细节"}:
        return terms

    resolved = terms[:]
    if field == "体型轮廓":
        breast_terms = [term for term in resolved if term in BREAST_SIZE_TERMS]
        if len(breast_terms) > 1:
            best_breast = strongest_term(breast_terms, resolved, counter)
            resolved = [term for term in resolved if term not in BREAST_SIZE_TERMS or term == best_breast]
    if field == "身材细节" and "大腿线条丰满" in resolved:
        slim_leg_count = max(counter["腿部线条偏细"], counter["四肢线条偏细"])
        full_leg_count = counter["大腿线条丰满"]
        if slim_leg_count >= full_leg_count:
            resolved = [term for term in resolved if term != "大腿线条丰满"]
        else:
            resolved = [term for term in resolved if term not in {"腿部线条偏细", "四肢线条偏细"}]
    return resolved


def strongest_term(terms: list[str], ordered_terms: list[str], counter: Counter[str]) -> str:
    return sorted(
        terms,
        key=lambda term: (-counter[term], ordered_terms.index(term), term),
    )[0]


def weights_for_selected_fields(selected_fields: list[str] | None) -> dict[str, float]:
    selected = set(selected_fields or [])
    return {
        field: DEFAULT_FIELD_WEIGHTS[field] if field in selected else 0.0
        for field in FEATURE_FIELDS
    }


def weighted_text_match(
    profile_text: str,
    candidate_text: str,
    weights: dict[str, float],
) -> tuple[float, dict[str, list[str]], dict[str, float]]:
    profile = parse_feature_text(profile_text)
    candidate = parse_feature_text(candidate_text)
    total_weight = 0.0
    weighted_score = 0.0
    matched: dict[str, list[str]] = {}
    score_parts: dict[str, float] = {}

    for field in FEATURE_FIELDS:
        weight = float(weights.get(field, 0.0))
        if weight <= 0:
            continue

        profile_tokens = text_tokens(profile.get(field, ""))
        candidate_tokens = text_tokens(candidate.get(field, ""))
        if not profile_tokens:
            continue

        total_weight += weight
        if not candidate_tokens:
            score_parts[field] = 0.0
            continue

        overlap = profile_tokens & candidate_tokens
        field_score = len(overlap) / math.sqrt(len(profile_tokens) * len(candidate_tokens))
        score_parts[field] = round(field_score * 100, 2)
        weighted_score += field_score * weight

        display_overlap = display_terms(profile.get(field, "")) & display_terms(candidate.get(field, ""))
        if display_overlap:
            matched[field] = sorted(display_overlap)

    if total_weight == 0:
        return 0.0, matched, score_parts

    return round((weighted_score / total_weight) * 100, 2), matched, score_parts


def format_matched_terms(matched: dict[str, list[str]]) -> str:
    if not matched:
        return ""
    return "；".join(f"{field}: {'、'.join(terms)}" for field, terms in matched.items())
