from __future__ import annotations

from pathlib import Path
from typing import Any

from .media import is_image_path, iter_image_files, make_thumbnail, uploaded_path


def collect_reference_images(
    uploaded_files: Any,
    folder_path: str | None,
    max_count: int,
) -> list[Path]:
    # Kept for old tests/imports. The UI now calls collect_reference_media.
    candidates: list[Path] = []
    if uploaded_files:
        files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
        for item in files:
            path = uploaded_path(item)
            if path and is_image_path(path):
                candidates.append(path.resolve())
    if folder_path and folder_path.strip():
        candidates.extend(iter_image_files(folder_path.strip(), recursive=True))

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
