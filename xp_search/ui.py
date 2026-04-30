from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path
from typing import Any

import gradio as gr

from .config import (
    DATA_DIR,
    DEFAULT_FEATURE_BACKEND,
    DEFAULT_FEATURE_MODE,
    DEFAULT_SELECTED_FIELDS,
    DEFAULT_SPEED_MODE,
    FEATURE_FIELDS,
    FEATURE_BACKENDS,
    FEATURE_MODES,
    MEDIA_EXTENSIONS,
)
from .features import format_matched_terms, merge_feature_profiles, weighted_text_match, weights_for_selected_fields
from .media import collect_reference_media, iter_media_files
from .model import cache_speed_for_backend, get_extractor
from .online_sources import adapter_for_source, download_candidate_cover, split_keywords
from .storage import MediaRecord, MediaStore, get_cached_or_extract_media, get_cached_or_extract_online

CSS = """
.result-note textarea { font-family: ui-monospace, Consolas, monospace; }
.compact-table table { font-size: 13px; }
.step-title { margin-top: 0.5rem; }
.result-card-list {
  display: grid;
  gap: 12px;
}
.result-card {
  display: grid;
  grid-template-columns: minmax(120px, 180px) minmax(0, 1fr);
  gap: 14px;
  padding: 12px;
  border: 1px solid var(--border-color-primary);
  border-radius: 8px;
  background: var(--block-background-fill);
}
.result-card img {
  width: 100%;
  aspect-ratio: 3 / 4;
  object-fit: contain;
  border-radius: 6px;
  background: #111;
}
.result-card-title {
  display: flex;
  gap: 10px;
  align-items: baseline;
  flex-wrap: wrap;
  margin-bottom: 8px;
}
.result-score {
  font-size: 22px;
  font-weight: 700;
  color: var(--button-primary-background-fill);
}
.source-pill {
  font-size: 12px;
  padding: 2px 7px;
  border: 1px solid var(--border-color-primary);
  border-radius: 999px;
  opacity: 0.85;
}
.result-path {
  font-size: 12px;
  opacity: 0.75;
  overflow-wrap: anywhere;
}
.result-label {
  margin-top: 8px;
  font-weight: 700;
}
.result-text {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  line-height: 1.55;
}
@media (max-width: 700px) {
  .result-card { grid-template-columns: 1fr; }
}
"""

ONLINE_SOURCES = ["Pornhub", "通用网页/oEmbed/OpenGraph", "Telegram"]


def scan_library(
    library_path: str,
    recursive: bool,
    scan_limit: int | float,
    feature_backend: str,
    feature_mode: str,
    speed_mode: str,
    force_refresh: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> str:
    if not library_path or not library_path.strip():
        return "请先输入媒体库文件夹路径。"

    root = Path(library_path.strip()).expanduser()
    if not root.exists() or not root.is_dir():
        return f"媒体库文件夹不存在：{root}"

    media_paths = iter_media_files(root, recursive=recursive)
    limit = int(scan_limit or 0)
    if limit > 0:
        media_paths = media_paths[:limit]
    if not media_paths:
        return "没有找到可处理媒体。支持图片 jpg/jpeg/png/webp/bmp 和视频 mp4/mkv/webm/mov/avi/m4v。"

    extractor = get_extractor(feature_backend)
    cache_speed = cache_speed_for_backend(feature_backend, speed_mode or DEFAULT_SPEED_MODE)
    store = MediaStore(
        model_id=extractor.model_id,
        prompt_version=extractor.prompt_version,
        feature_mode=feature_mode or DEFAULT_FEATURE_MODE,
        feature_speed=cache_speed,
    )
    created = 0
    cached = 0
    failed: list[str] = []

    try:
        for index, media_path in enumerate(progress.tqdm(media_paths, desc="生成媒体库封面特征"), start=1):
            try:
                _, was_cached = get_cached_or_extract_media(store, media_path, extractor, force=force_refresh)
                if was_cached:
                    cached += 1
                else:
                    created += 1
            except Exception as exc:  # noqa: BLE001 - UI should continue through bad files.
                failed.append(f"{media_path}: {exc}")
            progress(index / len(media_paths), desc=f"{index}/{len(media_paths)}")
    finally:
        total_records = store.count()
        store.close()

    lines = [
        f"媒体库特征生成完成：本次处理 {len(media_paths)} 个媒体，后端：{feature_backend}，特征：{feature_mode}，速度/缓存档：{cache_speed}。",
        f"新生成/刷新：{created} 个；使用缓存：{cached} 个；失败：{len(failed)} 个。",
        f"当前模式下数据库缓存：{total_records} 个媒体。",
    ]
    if force_refresh:
        lines.append("本次已启用强制重生成，未复用旧特征。")
    if failed:
        lines.append("失败样例：")
        lines.extend(failed[:5])
    return "\n".join(lines)


def build_xp_features(
    uploaded_files: Any,
    reference_folder: str,
    max_references: int | float,
    feature_backend: str,
    feature_mode: str,
    speed_mode: str,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, list[list[str]], str]:
    max_count = int(max_references or 10)
    references = collect_reference_media(uploaded_files, reference_folder, max_count)
    if not references:
        return "", [], "没有找到参考媒体。请上传图片/视频，或输入参考媒体文件夹路径。"

    extractor = get_extractor(feature_backend)
    cache_speed = cache_speed_for_backend(feature_backend, speed_mode or DEFAULT_SPEED_MODE)
    store = MediaStore(
        model_id=extractor.model_id,
        prompt_version=extractor.prompt_version,
        feature_mode=feature_mode or DEFAULT_FEATURE_MODE,
        feature_speed=cache_speed,
    )
    rows: list[list[str]] = []
    feature_texts: list[str] = []
    failed: list[str] = []

    try:
        for index, media_path in enumerate(progress.tqdm(references, desc="生成参考媒体 XP 特征"), start=1):
            try:
                record, _ = get_cached_or_extract_media(store, media_path, extractor)
                feature_texts.append(record.features)
                rows.append([media_type_label(record), record.title, record.path, record.cover_path, record.features])
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{media_path}: {exc}")
            progress(index / len(references), desc=f"{index}/{len(references)}")
    finally:
        store.close()

    xp_features = merge_feature_profiles(feature_texts, mode=feature_mode or DEFAULT_FEATURE_MODE)
    status = f"已使用 {len(feature_texts)} 个参考媒体生成 XP 特征，后端：{feature_backend}，特征：{feature_mode}，速度/缓存档：{cache_speed}。你可以手动删除或改写下面的词再匹配。"
    if failed:
        status += "\n失败样例：\n" + "\n".join(failed[:5])
    return xp_features, rows, status


def collect_online_media(
    xp_features: str,
    feature_backend: str,
    feature_mode: str,
    speed_mode: str,
    selected_fields: list[str] | None,
    top_k: int | float,
    online_min_score: int | float,
    source: str,
    keywords: str,
    page_limit: int | float,
    max_results: int | float,
    request_delay: int | float,
    telegram_api_id: str,
    telegram_api_hash: str,
    telegram_phone: str,
    telegram_session: str,
    telegram_chats: str,
    telegram_message_limit: int | float,
    force_refresh: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> tuple[str, list[list[str]], list[tuple[str, str]], Any, Any, Any, str]:
    keyword_list = split_keywords(keywords)
    if not keyword_list:
        return "请先输入关键词或 URL，每行一个。", [], [], gr.update(), gr.update(), gr.update(), gr.update()

    extractor = get_extractor(feature_backend)
    cache_speed = cache_speed_for_backend(feature_backend, speed_mode or DEFAULT_SPEED_MODE)
    store = MediaStore(
        model_id=extractor.model_id,
        prompt_version=extractor.prompt_version,
        feature_mode=feature_mode or DEFAULT_FEATURE_MODE,
        feature_speed=cache_speed,
    )
    rows: list[list[str]] = []
    preview_gallery: list[tuple[str, str]] = []
    failed: list[str] = []
    created = 0
    cached = 0
    candidate_count = 0

    try:
        adapter = adapter_for_source(
            source,
            request_delay=float(request_delay or 0),
            telegram_api_id=telegram_api_id or "",
            telegram_api_hash=telegram_api_hash or "",
            telegram_phone=telegram_phone or "",
            telegram_session=telegram_session or "",
            telegram_chats=telegram_chats or "",
            telegram_message_limit=int(telegram_message_limit or 100),
        )

        for keyword_index, keyword in enumerate(keyword_list, start=1):
            try:
                candidates = adapter.search(keyword, int(page_limit or 1), int(max_results or 20))
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{keyword}: {exc}")
                continue
            candidate_count += len(candidates)
            for index, candidate in enumerate(progress.tqdm(candidates, desc=f"采集在线封面：{keyword}"), start=1):
                try:
                    candidate = download_candidate_cover(candidate)
                    record, was_cached = get_cached_or_extract_online(store, candidate, extractor, force=force_refresh)
                    if was_cached:
                        cached += 1
                    else:
                        created += 1
                    rows.append([record.source_site, record.title, record.path, record.cover_path, record.features])
                    preview_gallery.append((record.thumb_path if Path(record.thumb_path).exists() else record.cover_path, f"{record.source_site} | {record.title}"))
                except Exception as exc:  # noqa: BLE001
                    failed.append(f"{candidate.page_url}: {exc}")
                total_steps = max(1, len(keyword_list) * max(1, len(candidates)))
                progress(((keyword_index - 1) * max(1, len(candidates)) + index) / total_steps, desc=f"{keyword_index}/{len(keyword_list)}")

        records = store.all_records()
    finally:
        store.close()

    lines = [
        f"在线封面采集完成：来源 {source}，候选 {candidate_count} 个，新生成/刷新 {created} 个，缓存 {cached} 个，失败 {len(failed)} 个。",
        "在线结果已写入同一个媒体库缓存，可直接参与下面的 XP 匹配。",
    ]
    if failed:
        lines.append("失败样例：")
        lines.extend(failed[:5])

    if xp_features and xp_features.strip():
        cards, gallery, table_rows, match_status = match_records(
            records=records,
            xp_features=xp_features,
            selected_fields=selected_fields,
            top_k=top_k,
            min_score=online_min_score,
        )
        return "\n".join(lines), rows, preview_gallery, cards, gallery, table_rows, match_status

    return "\n".join(lines), rows, preview_gallery, gr.update(), gr.update(), gr.update(), "已采集在线封面；填写或生成 XP 特征后再匹配。"


def search_library(
    xp_features: str,
    feature_backend: str,
    feature_mode: str,
    speed_mode: str,
    selected_fields: list[str] | None,
    top_k: int | float,
    min_score: int | float,
) -> tuple[str, list[tuple[str, str]], list[list[str | float]], str]:
    if not xp_features or not xp_features.strip():
        return "", [], [], "请先在第 1 步生成或填写 XP 特征，再进行匹配筛选。"

    extractor = get_extractor(feature_backend)
    cache_speed = cache_speed_for_backend(feature_backend, speed_mode or DEFAULT_SPEED_MODE)
    store = MediaStore(
        model_id=extractor.model_id,
        prompt_version=extractor.prompt_version,
        feature_mode=feature_mode or DEFAULT_FEATURE_MODE,
        feature_speed=cache_speed,
    )
    try:
        records = store.all_records()
    finally:
        store.close()

    if not records:
        return "", [], [], f"当前“{feature_backend}/{feature_mode}/{cache_speed}”模式下媒体库特征为空，请先完成第 2 步或在线封面采集。"

    return match_records(records, xp_features, selected_fields, top_k, min_score)


def match_records(
    records: list[MediaRecord],
    xp_features: str,
    selected_fields: list[str] | None,
    top_k: int | float,
    min_score: int | float,
) -> tuple[str, list[tuple[str, str]], list[list[str | float]], str]:
    weights = weights_for_selected_fields(selected_fields)
    if not any(value > 0 for value in weights.values()):
        return "", [], [], "请至少选择一个参与筛选的特征字段。"

    scored = []
    threshold = float(min_score or 0)
    for record in records:
        score, matched, parts = weighted_text_match(xp_features, record.features, weights)
        if score < threshold:
            continue
        scored.append((score, record, matched, parts))

    scored.sort(key=lambda item: item[0], reverse=True)
    limit = max(1, int(top_k or 30))
    selected = scored[:limit]

    gallery = [
        (
            record.thumb_path if Path(record.thumb_path).exists() else record.cover_path,
            f"{score:.1f} | {media_type_label(record)} | {record.title}",
        )
        for score, record, _, _ in selected
    ]
    rows: list[list[str | float]] = [
        [
            round(score, 2),
            media_type_label(record),
            record.title,
            format_matched_terms(matched),
            record.path,
            record.features,
        ]
        for score, record, matched, _ in selected
    ]
    cards = render_result_cards(selected)
    return cards, gallery, rows, f"匹配完成：展示 {len(selected)} / {len(scored)} 个媒体，当前缓存共 {len(records)} 个媒体。"


def render_result_cards(scored_records: list[tuple[float, MediaRecord, dict[str, list[str]], dict[str, float]]]) -> str:
    if not scored_records:
        return ""

    cards: list[str] = ['<div class="result-card-list">']
    for score, record, matched, _ in scored_records:
        image_path = record.thumb_path if Path(record.thumb_path).exists() else record.cover_path
        image_src = image_data_uri(Path(image_path))
        matched_text = format_matched_terms(matched) or "无"
        source_text = media_type_label(record)
        source_line = source_link(record)
        cards.append(
            "\n".join(
                [
                    '<div class="result-card">',
                    f'<img src="{image_src}" alt="{html.escape(record.title)}">',
                    "<div>",
                    '<div class="result-card-title">',
                    f'<span class="result-score">{score:.1f}</span>',
                    f'<span class="source-pill">{html.escape(source_text)}</span>',
                    f'<span>{html.escape(record.title)}</span>',
                    "</div>",
                    '<div class="result-label">命中特征</div>',
                    f'<div class="result-text">{html.escape(matched_text)}</div>',
                    '<div class="result-label">该封面识别特征</div>',
                    f'<div class="result-text">{html.escape(record.features)}</div>',
                    '<div class="result-label">来源</div>',
                    f'<div class="result-path">{source_line}</div>',
                    "</div>",
                    "</div>",
                ]
            )
        )
    cards.append("</div>")
    return "\n".join(cards)


def media_type_label(record: MediaRecord) -> str:
    if record.media_type == "image":
        return "本地图片"
    if record.media_type == "video":
        return "本地视频封面"
    if record.media_type == "online_video":
        return record.source_site or "在线视频封面"
    return record.media_type


def source_link(record: MediaRecord) -> str:
    value = record.page_url or record.path
    escaped = html.escape(value)
    if value.startswith(("http://", "https://")):
        return f'<a href="{escaped}" target="_blank" rel="noreferrer">{escaped}</a>'
    return escaped


def image_data_uri(path: Path) -> str:
    try:
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
    except OSError:
        return ""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="多参考媒体 XP 特征筛选") as demo:
        gr.Markdown("# 多参考媒体 XP 特征筛选")
        gr.Markdown("流程：先生成参考媒体 XP 特征，再生成/更新本地媒体库特征；在线视频封面可单独采集，最后统一用可编辑 XP 特征匹配。")

        with gr.Row():
            speed_mode = gr.Textbox(value=DEFAULT_SPEED_MODE, visible=False)
            feature_backend = gr.Dropdown(
                label="识别模型",
                choices=FEATURE_BACKENDS,
                value=DEFAULT_FEATURE_BACKEND,
                info="Qwen3-VL v4.5 用 4bit 精描人物细节；Camie v2 用 ONNX 快速批量打标签。",
            )
            feature_mode = gr.Radio(
                label="特征粒度",
                choices=FEATURE_MODES,
                value=DEFAULT_FEATURE_MODE,
                info="细模式保留更多人物细节；粗模式只保留核心特征。",
            )

        gr.Markdown("## 1. 生成 XP 特征", elem_classes=["step-title"])
        with gr.Row():
            with gr.Column(scale=1):
                uploaded_refs = gr.File(
                    label="参考图片/视频文件",
                    file_count="multiple",
                    file_types=sorted(MEDIA_EXTENSIONS),
                )
                reference_folder = gr.Textbox(label="参考图片/视频文件夹路径", placeholder=r"D:\images\favorites")
                max_references = gr.Slider(
                    label="最大参考媒体数量",
                    minimum=1,
                    maximum=50,
                    value=10,
                    step=1,
                )
                xp_button = gr.Button("生成 XP 特征", variant="primary")
            with gr.Column(scale=1):
                xp_status = gr.Textbox(label="XP 特征状态", lines=5, interactive=False)

        xp_features = gr.Textbox(
            label="XP 特征（可自由删除、改写后再筛选）",
            lines=12,
            elem_classes=["result-note"],
        )

        with gr.Accordion("每个参考媒体的封面识别特征", open=False):
            reference_table = gr.Dataframe(
                headers=["类型", "标题", "路径/URL", "封面路径", "识别特征"],
                datatype=["str", "str", "str", "str", "str"],
                wrap=True,
                interactive=False,
                elem_classes=["compact-table"],
            )

        gr.Markdown("## 2. 生成/更新本地媒体库特征", elem_classes=["step-title"])
        with gr.Row():
            with gr.Column(scale=1):
                library_path = gr.Textbox(label="本地图片/视频媒体库文件夹路径", placeholder=r"D:\images\library")
                recursive = gr.Checkbox(label="包含子文件夹", value=True)
                force_refresh = gr.Checkbox(label="强制重生成本地媒体特征", value=False)
            with gr.Column(scale=1):
                scan_limit = gr.Number(label="扫描上限（0 = 全部）", value=0, precision=0)
                scan_button = gr.Button("生成/更新本地媒体特征", variant="primary")
                scan_status = gr.Textbox(label="本地媒体特征状态", lines=6, interactive=False)

        gr.Markdown("## 3. 在线视频封面采集（可选）", elem_classes=["step-title"])
        with gr.Row():
            with gr.Column(scale=1):
                online_source = gr.Dropdown(label="站点", choices=ONLINE_SOURCES, value="通用网页/oEmbed/OpenGraph")
                online_keywords = gr.Textbox(
                    label="关键词或 URL（每行一个）",
                    lines=4,
                    placeholder="Pornhub 填关键词；通用网页填视频页 URL；Telegram 填搜索词。",
                )
                page_limit = gr.Number(label="页数上限", value=1, precision=0)
                max_results = gr.Number(label="每个关键词最大结果数", value=20, precision=0)
                request_delay = gr.Number(label="请求间隔（秒）", value=0.8, precision=2)
                online_min_score = gr.Slider(label="采集后自动匹配最低分", minimum=0, maximum=100, value=0, step=1)
                online_force_refresh = gr.Checkbox(label="强制重生成在线封面特征", value=False)
                online_button = gr.Button("采集在线封面并生成特征", variant="primary")
            with gr.Column(scale=1):
                telegram_api_id = gr.Textbox(label="Telegram api_id", placeholder="只在选择 Telegram 时需要")
                telegram_api_hash = gr.Textbox(label="Telegram api_hash", type="password")
                telegram_phone = gr.Textbox(label="Telegram phone", placeholder="+86...")
                telegram_session = gr.Textbox(label="Telegram session 路径", value=str(DATA_DIR / "telegram_user"))
                telegram_chats = gr.Textbox(label="Telegram 频道/聊天（逗号或换行分隔）", lines=3)
                telegram_message_limit = gr.Number(label="每个聊天搜索消息上限", value=100, precision=0)
                online_status = gr.Textbox(label="在线采集状态", lines=6, interactive=False)

        online_table = gr.Dataframe(
            headers=["来源", "标题", "URL", "封面路径", "识别特征"],
            datatype=["str", "str", "str", "str", "str"],
            wrap=True,
            interactive=False,
            elem_classes=["compact-table"],
        )
        online_gallery = gr.Gallery(
            label="在线采集封面预览",
            columns=[2, 4, 6],
            rows=2,
            object_fit="contain",
            height=360,
        )

        gr.Markdown("## 4. 匹配筛选", elem_classes=["step-title"])
        selected_fields = gr.CheckboxGroup(
            label="参与筛选的特征字段",
            choices=FEATURE_FIELDS,
            value=DEFAULT_SELECTED_FIELDS,
        )
        with gr.Row():
            top_k = gr.Slider(label="Top K", minimum=1, maximum=200, value=30, step=1)
            min_score = gr.Slider(label="最低匹配分", minimum=0, maximum=100, value=0, step=1)
            search_button = gr.Button("用 XP 特征匹配媒体库", variant="primary")

        match_status = gr.Textbox(label="匹配状态", lines=2, interactive=False)
        result_cards = gr.HTML(label="封面与特征对比结果")
        result_gallery = gr.Gallery(
            label="结果网格",
            columns=[2, 4, 6],
            rows=4,
            object_fit="contain",
            height=520,
        )
        result_table = gr.Dataframe(
            headers=["匹配分", "来源", "标题", "命中特征", "路径/URL", "该封面识别特征"],
            datatype=["number", "str", "str", "str", "str", "str"],
            wrap=True,
            interactive=False,
            elem_classes=["compact-table"],
        )

        xp_button.click(
            fn=build_xp_features,
            inputs=[uploaded_refs, reference_folder, max_references, feature_backend, feature_mode, speed_mode],
            outputs=[xp_features, reference_table, xp_status],
        )
        scan_button.click(
            fn=scan_library,
            inputs=[library_path, recursive, scan_limit, feature_backend, feature_mode, speed_mode, force_refresh],
            outputs=scan_status,
        )
        online_button.click(
            fn=collect_online_media,
            inputs=[
                xp_features,
                feature_backend,
                feature_mode,
                speed_mode,
                selected_fields,
                top_k,
                online_min_score,
                online_source,
                online_keywords,
                page_limit,
                max_results,
                request_delay,
                telegram_api_id,
                telegram_api_hash,
                telegram_phone,
                telegram_session,
                telegram_chats,
                telegram_message_limit,
                online_force_refresh,
            ],
            outputs=[online_status, online_table, online_gallery, result_cards, result_gallery, result_table, match_status],
        )
        search_button.click(
            fn=search_library,
            inputs=[xp_features, feature_backend, feature_mode, speed_mode, selected_fields, top_k, min_score],
            outputs=[result_cards, result_gallery, result_table, match_status],
        )

    return demo
