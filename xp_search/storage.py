from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CACHE_MODEL_ID, DATA_DIR, DB_PATH, DEFAULT_SPEED_MODE, PROMPT_VERSION
from .media import CoverInfo, make_thumbnail, resolve_media_cover

CACHE_TABLE = "media_feature_cache"


@dataclass(frozen=True)
class MediaRecord:
    source_key: str
    media_type: str
    path: str
    page_url: str
    cover_url: str
    cover_path: str
    title: str
    source_site: str
    cover_source: str
    mtime: float
    size: int
    width: int
    height: int
    thumb_path: str
    features: str
    model_id: str
    prompt_version: str
    feature_mode: str
    feature_speed: str


class MediaStore:
    def __init__(
        self,
        db_path: str | Path = DB_PATH,
        model_id: str = CACHE_MODEL_ID,
        prompt_version: str = PROMPT_VERSION,
        feature_mode: str = "细",
        feature_speed: str = DEFAULT_SPEED_MODE,
    ) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path)
        self.model_id = model_id
        self.prompt_version = prompt_version
        self.feature_mode = feature_mode
        self.feature_speed = feature_speed
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS media_feature_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_key TEXT NOT NULL,
                media_type TEXT NOT NULL,
                path TEXT NOT NULL,
                page_url TEXT NOT NULL,
                cover_url TEXT NOT NULL,
                cover_path TEXT NOT NULL,
                title TEXT NOT NULL,
                source_site TEXT NOT NULL,
                cover_source TEXT NOT NULL,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                thumb_path TEXT NOT NULL,
                features TEXT NOT NULL,
                model_id TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                feature_mode TEXT NOT NULL,
                feature_speed TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(source_key, model_id, prompt_version, feature_mode, feature_speed)
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_media_feature_cache_source ON media_feature_cache(source_key)")
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_media_feature_cache_context
            ON media_feature_cache(model_id, prompt_version, feature_mode, feature_speed)
            """
        )
        self.conn.commit()

    def get_cached(self, source_key: str, mtime: float, size: int) -> MediaRecord | None:
        row = self.conn.execute(
            """
            SELECT source_key, media_type, path, page_url, cover_url, cover_path, title, source_site, cover_source,
                   mtime, size, width, height, thumb_path, features,
                   model_id, prompt_version, feature_mode, feature_speed
            FROM media_feature_cache
            WHERE source_key = ?
              AND model_id = ?
              AND prompt_version = ?
              AND feature_mode = ?
              AND feature_speed = ?
            """,
            (source_key, self.model_id, self.prompt_version, self.feature_mode, self.feature_speed),
        ).fetchone()
        if not row:
            return None
        if float(row["mtime"]) != float(mtime) or int(row["size"]) != int(size):
            return None
        if not row["features"].strip():
            return None
        if not Path(row["thumb_path"]).exists() or not Path(row["cover_path"]).exists():
            return None
        return self._row_to_record(row)

    def upsert(self, record: MediaRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO media_feature_cache (
                source_key, media_type, path, page_url, cover_url, cover_path, title, source_site, cover_source,
                mtime, size, width, height, thumb_path, features,
                model_id, prompt_version, feature_mode, feature_speed, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_key, model_id, prompt_version, feature_mode, feature_speed) DO UPDATE SET
                media_type = excluded.media_type,
                path = excluded.path,
                page_url = excluded.page_url,
                cover_url = excluded.cover_url,
                cover_path = excluded.cover_path,
                title = excluded.title,
                source_site = excluded.source_site,
                cover_source = excluded.cover_source,
                mtime = excluded.mtime,
                size = excluded.size,
                width = excluded.width,
                height = excluded.height,
                thumb_path = excluded.thumb_path,
                features = excluded.features,
                updated_at = excluded.updated_at
            """,
            (
                record.source_key,
                record.media_type,
                record.path,
                record.page_url,
                record.cover_url,
                record.cover_path,
                record.title,
                record.source_site,
                record.cover_source,
                record.mtime,
                record.size,
                record.width,
                record.height,
                record.thumb_path,
                record.features,
                record.model_id,
                record.prompt_version,
                record.feature_mode,
                record.feature_speed,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def all_records(self) -> list[MediaRecord]:
        rows = self.conn.execute(
            """
            SELECT source_key, media_type, path, page_url, cover_url, cover_path, title, source_site, cover_source,
                   mtime, size, width, height, thumb_path, features,
                   model_id, prompt_version, feature_mode, feature_speed
            FROM media_feature_cache
            WHERE model_id = ?
              AND prompt_version = ?
              AND feature_mode = ?
              AND feature_speed = ?
            ORDER BY source_site COLLATE NOCASE, title COLLATE NOCASE, source_key COLLATE NOCASE
            """,
            (self.model_id, self.prompt_version, self.feature_mode, self.feature_speed),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def count(self) -> int:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM media_feature_cache
            WHERE model_id = ?
              AND prompt_version = ?
              AND feature_mode = ?
              AND feature_speed = ?
            """,
            (self.model_id, self.prompt_version, self.feature_mode, self.feature_speed),
        ).fetchone()
        return int(row["count"])

    def _row_to_record(self, row: sqlite3.Row) -> MediaRecord:
        return MediaRecord(
            source_key=row["source_key"],
            media_type=row["media_type"],
            path=row["path"],
            page_url=row["page_url"],
            cover_url=row["cover_url"],
            cover_path=row["cover_path"],
            title=row["title"],
            source_site=row["source_site"],
            cover_source=row["cover_source"],
            mtime=float(row["mtime"]),
            size=int(row["size"]),
            width=int(row["width"]),
            height=int(row["height"]),
            thumb_path=row["thumb_path"],
            features=row["features"],
            model_id=row["model_id"],
            prompt_version=row["prompt_version"],
            feature_mode=row["feature_mode"],
            feature_speed=row["feature_speed"],
        )


def get_cached_or_extract_media(
    store: MediaStore,
    media_path: str | Path,
    extractor: object,
    force: bool = False,
) -> tuple[MediaRecord, bool]:
    path = Path(media_path).resolve()
    stat = path.stat()
    source_key = str(path)
    if not force:
        cached = store.get_cached(source_key, stat.st_mtime, stat.st_size)
        if cached:
            return cached, True

    cover = resolve_media_cover(path)
    record = extract_features_for_cover(
        store=store,
        extractor=extractor,
        source_key=source_key,
        media_type=cover.media_type,
        path=source_key,
        page_url=cover.page_url,
        cover_url=cover.cover_url,
        cover_path=cover.cover_path,
        title=cover.title,
        source_site=cover.source_site,
        cover_source=cover.cover_source,
        mtime=stat.st_mtime,
        size=stat.st_size,
    )
    store.upsert(record)
    return record, False


def get_cached_or_extract_online(
    store: MediaStore,
    candidate: Any,
    extractor: object,
    force: bool = False,
) -> tuple[MediaRecord, bool]:
    cover_path = Path(candidate.cover_path).resolve()
    stat = cover_path.stat()
    source_key = candidate.page_url or candidate.cover_url or str(cover_path)
    if not force:
        cached = store.get_cached(source_key, stat.st_mtime, stat.st_size)
        if cached:
            return cached, True

    record = extract_features_for_cover(
        store=store,
        extractor=extractor,
        source_key=source_key,
        media_type="online_video",
        path=candidate.page_url or str(cover_path),
        page_url=candidate.page_url,
        cover_url=candidate.cover_url,
        cover_path=cover_path,
        title=candidate.title or source_key,
        source_site=candidate.source,
        cover_source="online_cover",
        mtime=stat.st_mtime,
        size=stat.st_size,
    )
    store.upsert(record)
    return record, False


def extract_features_for_cover(
    store: MediaStore,
    extractor: object,
    source_key: str,
    media_type: str,
    path: str,
    page_url: str,
    cover_url: str,
    cover_path: str | Path,
    title: str,
    source_site: str,
    cover_source: str,
    mtime: float,
    size: int,
) -> MediaRecord:
    cover = Path(cover_path).resolve()
    thumb_path, width, height = make_thumbnail(cover)
    features = extractor.extract(cover, mode=store.feature_mode, speed=store.feature_speed)
    return MediaRecord(
        source_key=source_key,
        media_type=media_type,
        path=path,
        page_url=page_url,
        cover_url=cover_url,
        cover_path=str(cover),
        title=title,
        source_site=source_site,
        cover_source=cover_source,
        mtime=mtime,
        size=size,
        width=width,
        height=height,
        thumb_path=thumb_path,
        features=features,
        model_id=store.model_id,
        prompt_version=store.prompt_version,
        feature_mode=store.feature_mode,
        feature_speed=store.feature_speed,
    )


def extract_features_for_media(media: CoverInfo, extractor: object, store: MediaStore) -> MediaRecord:
    stat = media.source_path.stat()
    return extract_features_for_cover(
        store=store,
        extractor=extractor,
        source_key=str(media.source_path.resolve()),
        media_type=media.media_type,
        path=str(media.source_path.resolve()),
        page_url=media.page_url,
        cover_url=media.cover_url,
        cover_path=media.cover_path,
        title=media.title,
        source_site=media.source_site,
        cover_source=media.cover_source,
        mtime=stat.st_mtime,
        size=stat.st_size,
    )


# Backwards-compatible names for existing tests/imports.
ImageRecord = MediaRecord
ImageStore = MediaStore
get_cached_or_extract = get_cached_or_extract_media
