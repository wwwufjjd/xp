from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from .config import ONLINE_COVER_DIR


@dataclass(frozen=True)
class OnlineVideoCandidate:
    source: str
    title: str
    page_url: str
    cover_url: str = ""
    cover_path: str = ""
    duration: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class SourceAdapter:
    source = "通用网页"

    def search(self, keyword: str, page_limit: int, max_results: int) -> list[OnlineVideoCandidate]:
        raise NotImplementedError


class PornhubAdapter(SourceAdapter):
    source = "Pornhub"
    base_url = "https://www.pornhub.com"

    def __init__(self, request_delay: float = 0.0) -> None:
        self.request_delay = max(0.0, float(request_delay or 0.0))

    def search(self, keyword: str, page_limit: int, max_results: int) -> list[OnlineVideoCandidate]:
        import httpx

        keyword = keyword.strip()
        if not keyword:
            return []

        results: list[OnlineVideoCandidate] = []
        seen: set[str] = set()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/114.0",
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en,en-US;q=0.9",
            "Referer": f"{self.base_url}/",
        }
        timeout = httpx.Timeout(20.0, connect=10.0)
        with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
            for page in range(1, max(1, int(page_limit)) + 1):
                response = client.get(
                    f"{self.base_url}/webmasters/search",
                    params={
                        "search": keyword,
                        "page": page,
                        "thumbsize": "large",
                        "ordering": "featured",
                        "period": "alltime",
                    },
                )
                response.raise_for_status()
                candidates = parse_pornhub_api_json(response, self.base_url)
                if not candidates:
                    raise RuntimeError(
                        "Pornhub Webmasters API 没有返回视频结果；可以换更长的英文关键词，"
                        "或在“通用网页/oEmbed/OpenGraph”里粘贴你浏览器能打开的视频页 URL。"
                    )
                for candidate in candidates:
                    if candidate.page_url in seen:
                        continue
                    seen.add(candidate.page_url)
                    results.append(candidate)
                    if len(results) >= max_results:
                        return results
                if self.request_delay:
                    time.sleep(self.request_delay)
        return results


class GenericWebAdapter(SourceAdapter):
    source = "通用网页/oEmbed/OpenGraph"

    def __init__(self, request_delay: float = 0.0) -> None:
        self.request_delay = max(0.0, float(request_delay or 0.0))

    def search(self, keyword: str, page_limit: int, max_results: int) -> list[OnlineVideoCandidate]:
        import httpx
        from bs4 import BeautifulSoup

        url = keyword.strip()
        if not is_http_url(url):
            return []

        headers = {"User-Agent": "Mozilla/5.0 XP-Media-Cover-Search/1.0"}
        response = httpx.get(url, headers=headers, timeout=20.0, follow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        oembed_candidate = fetch_oembed_candidate(soup, str(response.url), headers)
        if oembed_candidate:
            return [oembed_candidate]

        title = meta_content(soup, "property", "og:title") or meta_content(soup, "name", "twitter:title")
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()
        cover = (
            meta_content(soup, "property", "og:image")
            or meta_content(soup, "name", "twitter:image")
            or meta_content(soup, "property", "og:image:secure_url")
        )
        page_url = meta_content(soup, "property", "og:url") or canonical_url(soup) or str(response.url)
        if not cover:
            return []

        if self.request_delay:
            time.sleep(self.request_delay)
        return [
            OnlineVideoCandidate(
                source=self.source,
                title=(title or page_url).strip(),
                page_url=urljoin(str(response.url), page_url),
                cover_url=urljoin(str(response.url), cover),
            )
        ][:max_results]


class TelegramAdapter(SourceAdapter):
    source = "Telegram"

    def __init__(
        self,
        api_id: str,
        api_hash: str,
        phone: str,
        session_name: str,
        chats: str,
        request_delay: float = 0.0,
        message_limit: int = 100,
    ) -> None:
        self.api_id = api_id.strip()
        self.api_hash = api_hash.strip()
        self.phone = phone.strip()
        self.session_name = session_name.strip() or str(ONLINE_COVER_DIR / "telegram_user")
        self.chats = [item.strip() for item in chats.replace("\n", ",").split(",") if item.strip()]
        self.request_delay = max(0.0, float(request_delay or 0.0))
        self.message_limit = max(1, int(message_limit or 100))

    def search(self, keyword: str, page_limit: int, max_results: int) -> list[OnlineVideoCandidate]:
        return asyncio.run(self._search(keyword.strip(), page_limit, max_results))

    async def _search(self, keyword: str, page_limit: int, max_results: int) -> list[OnlineVideoCandidate]:
        if not self.api_id or not self.api_hash:
            raise RuntimeError("Telegram 需要填写 api_id 和 api_hash。")
        if not self.chats:
            raise RuntimeError("Telegram 需要填写要搜索的频道或聊天。")

        try:
            from telethon import TelegramClient
            from telethon.tl.types import DocumentAttributeVideo
        except ImportError as exc:
            raise RuntimeError("缺少 telethon，请先安装 requirements.txt。") from exc

        ONLINE_COVER_DIR.mkdir(parents=True, exist_ok=True)
        client = TelegramClient(self.session_name, int(self.api_id), self.api_hash)
        await client.connect()
        try:
            if not await client.is_user_authorized():
                raise RuntimeError("Telegram session 未登录。请先用 Telethon 登录生成 session 后重试。")

            results: list[OnlineVideoCandidate] = []
            seen: set[str] = set()
            per_chat_limit = min(self.message_limit, max(1, int(page_limit)) * max(1, int(max_results)))
            for chat in self.chats:
                entity = await client.get_entity(chat)
                async for message in client.iter_messages(entity, search=keyword or None, limit=per_chat_limit):
                    document = getattr(message, "document", None)
                    is_video = bool(getattr(message, "video", None))
                    if document:
                        is_video = is_video or any(isinstance(attr, DocumentAttributeVideo) for attr in getattr(document, "attributes", []))
                    if not is_video:
                        continue

                    page_url = telegram_message_url(entity, message.id)
                    if page_url in seen:
                        continue
                    cover_path = telegram_cover_path(page_url)
                    downloaded = await client.download_media(message, file=str(cover_path), thumb=-1)
                    if not downloaded:
                        continue
                    cover_path = Path(downloaded)
                    title = telegram_message_title(entity, message)
                    results.append(
                        OnlineVideoCandidate(
                            source=self.source,
                            title=title,
                            page_url=page_url,
                            cover_path=str(cover_path),
                            duration=str(video_duration(document)),
                            metadata={"chat": chat, "message_id": message.id},
                        )
                    )
                    seen.add(page_url)
                    if len(results) >= max_results:
                        return results
                    if self.request_delay:
                        await asyncio.sleep(self.request_delay)
            return results
        finally:
            await client.disconnect()


def parse_pornhub_search_html(
    html: str,
    base_url: str,
    require_search_context: bool = False,
) -> list[OnlineVideoCandidate]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    if looks_like_pornhub_home_or_sfw(soup):
        return []

    candidates: list[OnlineVideoCandidate] = []
    seen: set[str] = set()
    roots = search_result_roots(soup)
    if require_search_context and not roots:
        return []
    if not roots:
        roots = [soup]

    cards = []
    for root in roots:
        cards.extend(root.select("li.videoBox, li.pcVideoListItem, div.videoBox, div.pcVideoListItem"))
    if not cards and not require_search_context:
        for root in roots:
            cards.extend(root.select("a[href*='view_video.php']"))

    for card in cards:
        root = card
        link = root if root.name == "a" and root.get("href") else root.select_one("a[href*='view_video.php']")
        if not link:
            continue
        page_url = urljoin(base_url, link.get("href", ""))
        if not page_url or page_url in seen:
            continue

        image = root.select_one("img")
        cover_url = image_attr(image) if image else ""
        if not cover_url:
            cover_url = first_attr(root, ["data-src", "data-thumb_url", "data-mediabook", "data-mediumthumb"])
        if not cover_url:
            continue

        title = first_attr(link, ["title", "aria-label"])
        if not title and image:
            title = first_attr(image, ["alt", "title"])
        if not title:
            title_node = root.select_one(".title, .title a, .videoTitle, span.title")
            title = title_node.get_text(" ", strip=True) if title_node else page_url
        duration_node = root.select_one(".duration, .videoDuration, var.duration")
        duration = duration_node.get_text(" ", strip=True) if duration_node else ""

        candidates.append(
            OnlineVideoCandidate(
                source="Pornhub",
                title=title.strip(),
                page_url=page_url,
                cover_url=urljoin(base_url, cover_url),
                duration=duration,
            )
        )
        seen.add(page_url)
    return candidates


def parse_pornhub_api_json(response: Any, base_url: str) -> list[OnlineVideoCandidate]:
    try:
        data = response.json()
    except Exception:
        return []

    videos = data.get("videos") if isinstance(data, dict) else None
    if not isinstance(videos, list):
        return []

    candidates: list[OnlineVideoCandidate] = []
    seen: set[str] = set()
    for video in videos:
        if not isinstance(video, dict):
            continue
        page_url = str(video.get("url") or "").strip()
        if not page_url:
            video_id = str(video.get("video_id") or "").strip()
            if not video_id:
                continue
            page_url = f"{base_url}/view_video.php?viewkey={video_id}"
        page_url = urljoin(base_url, page_url)
        if page_url in seen:
            continue

        cover_url = (
            str(video.get("thumb") or "").strip()
            or str(video.get("default_thumb") or "").strip()
            or first_pornhub_thumb(video)
        )
        if not cover_url:
            continue

        candidates.append(
            OnlineVideoCandidate(
                source="Pornhub",
                title=str(video.get("title") or page_url).strip(),
                page_url=page_url,
                cover_url=urljoin(page_url, cover_url),
                duration=str(video.get("duration") or ""),
                metadata={
                    "video_id": video.get("video_id", ""),
                    "views": video.get("views", ""),
                    "rating": video.get("rating", ""),
                    "publish_date": video.get("publish_date", ""),
                    "source": "webmasters_api",
                },
            )
        )
        seen.add(page_url)
    return candidates


def first_pornhub_thumb(video: dict[str, Any]) -> str:
    thumbs = video.get("thumbs")
    if not isinstance(thumbs, list):
        return ""
    for thumb in reversed(thumbs):
        if isinstance(thumb, dict) and thumb.get("src"):
            return str(thumb["src"]).strip()
    return ""


def is_pornhub_search_response(response: Any, keyword: str = "") -> bool:
    final_path = urlparse(str(response.url)).path.rstrip("/")
    if final_path != "/video/search":
        return False
    if "sfw_homepage" in response.text or 'id="search_form_sfw"' in response.text:
        return False
    return bool(search_result_roots_from_html(response.text) or keyword.lower() in response.text.lower())


def search_result_roots_from_html(html: str) -> list[Any]:
    from bs4 import BeautifulSoup

    return search_result_roots(BeautifulSoup(html, "lxml"))


def search_result_roots(soup: Any) -> list[Any]:
    selectors = [
        "#videoSearchResult",
        "ul#videoSearchResult",
        "ul.videos.search-video-thumbs",
        "ul.search-video-thumbs",
        ".search-video-thumbs",
        "[data-testid='search-results']",
        ".search-results",
    ]
    roots: list[Any] = []
    seen: set[int] = set()
    for selector in selectors:
        for node in soup.select(selector):
            node_id = id(node)
            if node_id in seen:
                continue
            roots.append(node)
            seen.add(node_id)
    return roots


def looks_like_pornhub_home_or_sfw(soup: Any) -> bool:
    html_text = str(soup)
    if "sfw_homepage" in html_text or 'id="search_form_sfw"' in html_text:
        return True
    title = soup.title.string.strip().lower() if soup.title and soup.title.string else ""
    return title == "pornhub"


def fetch_oembed_candidate(soup: Any, page_url: str, headers: dict[str, str]) -> OnlineVideoCandidate | None:
    import httpx

    link = soup.select_one("link[type='application/json+oembed'], link[type='text/json+oembed']")
    if not link or not link.get("href"):
        return None
    endpoint = urljoin(page_url, link.get("href"))
    response = httpx.get(endpoint, headers=headers, timeout=20.0, follow_redirects=True)
    response.raise_for_status()
    data = response.json()
    cover = data.get("thumbnail_url") or data.get("thumbnail")
    if not cover:
        return None
    return OnlineVideoCandidate(
        source="通用网页/oEmbed/OpenGraph",
        title=(data.get("title") or page_url).strip(),
        page_url=data.get("web_page") or data.get("url") or page_url,
        cover_url=urljoin(page_url, cover),
        metadata={"provider_name": data.get("provider_name", ""), "type": data.get("type", "")},
    )


def download_candidate_cover(candidate: OnlineVideoCandidate) -> OnlineVideoCandidate:
    if candidate.cover_path:
        path = Path(candidate.cover_path)
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"封面文件不存在：{path}")
        return candidate

    if not candidate.cover_url:
        raise RuntimeError(f"没有封面 URL：{candidate.page_url}")

    import httpx

    ONLINE_COVER_DIR.mkdir(parents=True, exist_ok=True)
    cover_url = normalize_cover_url(candidate.cover_url, candidate.page_url)
    digest = hashlib.sha1(f"{candidate.page_url}|{cover_url}".encode("utf-8")).hexdigest()
    suffix = Path(urlparse(cover_url).path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    output_path = ONLINE_COVER_DIR / f"{digest}{suffix}"
    if not output_path.exists() or output_path.stat().st_size == 0:
        headers = {"User-Agent": "Mozilla/5.0 XP-Media-Cover-Search/1.0"}
        response = httpx.get(cover_url, headers=headers, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        output_path.write_bytes(response.content)

    return OnlineVideoCandidate(
        source=candidate.source,
        title=candidate.title,
        page_url=candidate.page_url,
        cover_url=cover_url,
        cover_path=str(output_path),
        duration=candidate.duration,
        metadata=candidate.metadata,
    )


def adapter_for_source(
    source: str,
    request_delay: float = 0.0,
    telegram_api_id: str = "",
    telegram_api_hash: str = "",
    telegram_phone: str = "",
    telegram_session: str = "",
    telegram_chats: str = "",
    telegram_message_limit: int = 100,
) -> SourceAdapter:
    if source == "Pornhub":
        return PornhubAdapter(request_delay=request_delay)
    if source == "Telegram":
        return TelegramAdapter(
            api_id=telegram_api_id,
            api_hash=telegram_api_hash,
            phone=telegram_phone,
            session_name=telegram_session,
            chats=telegram_chats,
            request_delay=request_delay,
            message_limit=telegram_message_limit,
        )
    return GenericWebAdapter(request_delay=request_delay)


def split_keywords(value: str) -> list[str]:
    return [line.strip() for line in (value or "").splitlines() if line.strip()]


def is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def normalize_cover_url(cover_url: str, page_url: str = "") -> str:
    if cover_url.startswith("//"):
        return "https:" + cover_url
    if page_url:
        return urljoin(page_url, cover_url)
    return cover_url


def image_attr(image: Any) -> str:
    return first_attr(image, ["data-src", "data-thumb_url", "data-mediumthumb", "data-image", "src"])


def first_attr(node: Any, names: list[str]) -> str:
    if not node:
        return ""
    for name in names:
        value = node.get(name)
        if value:
            return str(value).strip()
    return ""


def meta_content(soup: Any, attr_name: str, attr_value: str) -> str:
    node = soup.find("meta", attrs={attr_name: attr_value})
    if not node:
        return ""
    return str(node.get("content") or "").strip()


def canonical_url(soup: Any) -> str:
    node = soup.find("link", rel=lambda value: value and "canonical" in value)
    return str(node.get("href") or "").strip() if node else ""


def telegram_cover_path(page_url: str) -> Path:
    ONLINE_COVER_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(page_url.encode("utf-8")).hexdigest()
    return ONLINE_COVER_DIR / f"telegram_{digest}.jpg"


def telegram_message_url(entity: Any, message_id: int) -> str:
    username = getattr(entity, "username", None)
    if username:
        return f"https://t.me/{username}/{message_id}"
    chat_id = getattr(entity, "id", "")
    return f"telegram://chat/{chat_id}/{message_id}"


def telegram_message_title(entity: Any, message: Any) -> str:
    text = (getattr(message, "message", "") or "").strip().splitlines()
    if text:
        return text[0][:120]
    title = getattr(entity, "title", "") or getattr(entity, "username", "") or "Telegram"
    return f"{title} #{message.id}"


def video_duration(document: Any) -> int:
    if not document:
        return 0
    for attr in getattr(document, "attributes", []):
        duration = getattr(attr, "duration", 0)
        if duration:
            return int(duration)
    return 0
