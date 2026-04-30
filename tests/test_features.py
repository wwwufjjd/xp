from __future__ import annotations

import tempfile
import unittest
import subprocess
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image

from xp_search.features import (
    merge_feature_profiles,
    normalize_feature_conflicts,
    parse_feature_text,
    weighted_text_match,
    weights_for_selected_fields,
)
from xp_search.images import collect_reference_images
from xp_search.media import collect_reference_media, ffmpeg_path, iter_media_files, resolve_media_cover
from xp_search.config import QWEN_BACKEND
from xp_search.model import StubFeatureExtractor, is_sparse_feature_text, normalize_feature_backend, normalize_model_output
from xp_search.storage import ImageStore, get_cached_or_extract
from xp_search.tagger import estimate_visible_skin_tone, pick_camie_tags, tags_to_chinese_features
from xp_search.online_sources import (
    is_pornhub_search_response,
    parse_pornhub_api_json,
    parse_pornhub_search_html,
    telegram_message_title,
    telegram_message_url,
)


class FeatureTests(unittest.TestCase):
    def test_parse_feature_text_handles_aliases(self) -> None:
        parsed = parse_feature_text("穿搭：白色上衣、短裙\n身材：腰线清晰、腿部比例长\n动作：站姿、正面\n风格：自拍")
        self.assertEqual(parsed["服装配饰"], "白色上衣、短裙")
        self.assertEqual(parsed["身材细节"], "腰线清晰、腿部比例长")
        self.assertEqual(parsed["动作姿态"], "站姿、正面")
        self.assertEqual(parsed["画风质感"], "自拍")

    def test_parse_feature_text_handles_typos_without_leaking_unknown_fields(self) -> None:
        parsed = parse_feature_text("人物主体：女性\n体型轮廳：纤细\n其他：忽略\n肤色观感：白肤")
        self.assertEqual(parsed["人物主体"], "女性")
        self.assertEqual(parsed["体型轮廓"], "纤细")
        self.assertEqual(parsed["肤色观感"], "白肤")
        self.assertNotIn("其他", parsed["人物主体"])

    def test_merge_profiles_prefers_repeated_terms(self) -> None:
        profile = merge_feature_profiles(
            [
                "人物主体：单人、清晰人像\n体型轮廓：纤细轮廓\n肤色观感：白皙肤色\n发型发色：长发\n五官表情：微笑\n服装配饰：白色上衣\n动作姿态：站姿\n拍摄角度：正面\n场景背景：室内\n画风质感：自拍",
                "人物主体：单人、半身像\n体型轮廓：纤细轮廓\n肤色观感：白皙肤色\n发型发色：长发\n五官表情：自然表情\n服装配饰：浅色上衣\n动作姿态：站姿\n拍摄角度：正面\n场景背景：室内\n画风质感：自拍",
            ]
        )
        self.assertIn("人物主体：单人", profile)
        self.assertIn("体型轮廓：纤细轮廓", profile)
        self.assertIn("肤色观感：白皙肤色", profile)
        self.assertIn("发型发色：长发", profile)

    def test_merge_profiles_keeps_useful_unique_terms_after_repeated_terms(self) -> None:
        profile = merge_feature_profiles(
            [
                "身材细节：腰腹、肚脐、腿部可见",
                "身材细节：腰腹、肚脐、肩部裸露",
            ],
            max_terms_per_field=4,
        )
        self.assertIn("腰腹", profile)
        self.assertIn("肚脐", profile)
        self.assertIn("腿部可见", profile)
        self.assertIn("肩部裸露", profile)

    def test_merge_profiles_resolves_conflicting_body_terms(self) -> None:
        profile = merge_feature_profiles(
            [
                "体型轮廓：偏瘦体态、胸部偏小、曲线感明显、身形轮廓清楚\n身材细节：腿部线条偏细、四肢线条偏细",
                "体型轮廓：偏瘦体态、胸部偏小、胸部中等、曲线轮廓明显、身形轮廓明显\n身材细节：腿部线条偏细、大腿线条丰满",
                "体型轮廓：偏瘦体态、胸部中等、胸部偏小、曲线轮廓明显\n身材细节：四肢线条偏细、大腿线条丰满",
            ]
        )
        body_line = parse_feature_text(profile)["体型轮廓"]
        details_line = parse_feature_text(profile)["身材细节"]
        self.assertIn("胸部偏小", body_line)
        self.assertNotIn("胸部中等", body_line)
        self.assertIn("曲线轮廓明显", body_line)
        self.assertNotIn("曲线感明显", body_line)
        self.assertIn("身形轮廓明显", body_line)
        self.assertNotIn("身形轮廓清楚", body_line)
        self.assertIn("腿部线条偏细", details_line)
        self.assertNotIn("大腿线条丰满", details_line)

    def test_single_feature_text_resolves_conflicting_body_terms(self) -> None:
        normalized = normalize_feature_conflicts(
            "人物细节：女性、胸部中等、胸部偏小\n"
            "体型轮廓：纤细身形、胸部中等、胸部偏小、胸部平坦、曲线感明显、曲线轮廓明显\n"
            "身材细节：腿部线条偏细、四肢线条偏细、大腿线条丰满"
        )
        parsed = parse_feature_text(normalized)
        self.assertNotIn("胸部中等", parsed["人物细节"])
        self.assertIn("胸部中等", parsed["体型轮廓"])
        self.assertNotIn("胸部偏小", parsed["体型轮廓"])
        self.assertNotIn("胸部平坦", parsed["体型轮廓"])
        self.assertIn("曲线轮廓明显", parsed["体型轮廓"])
        self.assertNotIn("曲线感明显", parsed["体型轮廓"])
        self.assertNotIn("大腿线条丰满", parsed["身材细节"])

    def test_coarse_merge_keeps_fewer_terms(self) -> None:
        profile = merge_feature_profiles(
            [
                "人物主体：单人、清晰人像、正脸、半身像、自拍主体\n体型轮廓：纤细轮廓\n肤色观感：白皙肤色\n发型发色：长发\n五官表情：微笑\n服装配饰：白色上衣\n动作姿态：站姿\n拍摄角度：正面\n场景背景：室内\n画风质感：自拍",
            ],
            mode="粗",
        )
        first_line = profile.splitlines()[0]
        self.assertLessEqual(len(first_line.split("：", 1)[1].split("、")), 4)

    def test_weighted_match_changes_with_selected_fields(self) -> None:
        target = "人物主体：单人\n体型轮廓：纤细轮廓\n肤色观感：白皙肤色\n发型发色：长发\n五官表情：自然表情\n服装配饰：白色上衣\n动作姿态：站姿\n拍摄角度：正面\n场景背景：室内\n画风质感：自拍"
        candidate = "人物主体：单人\n体型轮廓：普通体型\n肤色观感：白皙肤色\n发型发色：短发\n五官表情：自然表情\n服装配饰：白色上衣\n动作姿态：坐姿\n拍摄角度：正面\n场景背景：室外\n画风质感：自拍"
        no_body, _, _ = weighted_text_match(target, candidate, weights_for_selected_fields(["人物主体", "服装配饰", "画风质感"]))
        with_body, _, _ = weighted_text_match(target, candidate, weights_for_selected_fields(["人物主体", "体型轮廓", "服装配饰", "画风质感"]))
        self.assertGreater(no_body, with_body)

    def test_normalize_model_output_falls_back_to_raw_caption(self) -> None:
        normalized = normalize_model_output("a portrait with long hair and light clothes", mode="粗")
        self.assertIn("人物主体：a portrait", normalized)

    def test_normalize_model_output_translates_common_english_terms(self) -> None:
        normalized = normalize_model_output("人物主体：女性\n体型轮廓：slender\n肤色观感：fair skin")
        self.assertIn("体型轮廓：纤细", normalized)
        self.assertIn("肤色观感：白皙肤色", normalized)
        self.assertTrue(is_sparse_feature_text(normalized))

    def test_normalize_feature_backend_accepts_qwen(self) -> None:
        self.assertEqual(normalize_feature_backend(QWEN_BACKEND), QWEN_BACKEND)

    def test_camie_tags_convert_to_chinese_feature_fields_without_body_inference(self) -> None:
        features = tags_to_chinese_features(
            [
                {"tag": "1girl", "score": 0.9},
                {"tag": "full_body", "score": 0.8},
                {"tag": "figure", "score": 0.7},
                {"tag": "small_breasts", "score": 0.7},
                {"tag": "legs", "score": 0.7},
                {"tag": "bare_shoulders", "score": 0.7},
                {"tag": "navel", "score": 0.7},
                {"tag": "short_hair", "score": 0.7},
                {"tag": "holding_phone", "score": 0.7},
                {"tag": "artist_name", "score": 0.9},
            ]
        )
        self.assertIn("人物主体：女性、全身可见", features)
        self.assertIn("体型轮廓：全身可见、身形轮廓清楚、胸部偏小", features)
        self.assertNotIn("纤细身形", features)
        self.assertNotIn("人物细节：女性、全身可见、身形轮廓清楚、胸部偏小", features)
        self.assertIn("身材细节：身形轮廓清楚、腿部可见、肩部裸露、腰腹和肚脐可见", features)
        self.assertIn("发型发色：短发造型", features)
        self.assertIn("动作姿态：手持手机", features)
        self.assertNotIn("artist", features)

    def test_camie_tags_do_not_infer_slender_from_full_body_and_visible_waist(self) -> None:
        features = tags_to_chinese_features(
            [
                {"tag": "1girl", "score": 0.9},
                {"tag": "full_body", "score": 0.8},
                {"tag": "medium_breasts", "score": 0.7},
                {"tag": "breasts", "score": 0.7},
                {"tag": "navel", "score": 0.7},
            ]
        )
        self.assertIn("人物主体：女性、全身可见", features)
        self.assertIn("身材细节：腰腹和肚脐可见", features)
        self.assertNotIn("纤细身形", features)
        self.assertNotIn("偏瘦体态", features)
        self.assertNotIn("腰线纤细", features)
        self.assertIn("体型轮廓：全身可见、胸部中等、胸部线条可见", features)
        self.assertNotIn("人物细节：女性、全身可见、胸部中等", features)

    def test_pick_camie_tags_requires_high_confidence_for_chest_terms(self) -> None:
        idx_to_tag = {"0": "1girl", "1": "medium_breasts", "2": "narrow_waist"}
        tag_to_category = {"1girl": "general", "medium_breasts": "general", "narrow_waist": "general"}
        low_confidence = pick_camie_tags(
            np.array([0.9, 0.70, 0.55], dtype="float32"),
            idx_to_tag,
            tag_to_category,
            threshold=0.45,
            top_k=10,
        )
        high_confidence = pick_camie_tags(
            np.array([0.9, 0.83, 0.64], dtype="float32"),
            idx_to_tag,
            tag_to_category,
            threshold=0.45,
            top_k=10,
        )

        self.assertEqual([item["tag"] for item in low_confidence], ["1girl"])
        self.assertEqual([item["tag"] for item in high_confidence], ["1girl", "medium_breasts", "narrow_waist"])

    def test_skin_tone_estimator_returns_visible_skin_tone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "skin.png"
            Image.new("RGB", (32, 32), color=(238, 190, 164)).save(image_path)
            terms = estimate_visible_skin_tone(image_path)
        self.assertTrue(any("肤色" in term for term in terms))


class ImageCollectionTests(unittest.TestCase):
    def test_collect_reference_images_merges_files_and_folder_with_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.jpg"
            second = root / "second.png"
            third = root / "third.jpg"
            for path in [first, second, third]:
                Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path)

            refs = collect_reference_images([str(first)], str(root), max_count=2)
            self.assertEqual(len(refs), 2)
            self.assertEqual(refs[0], first.resolve())

    def test_storage_cache_skips_unchanged_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "white_long_indoor.jpg"
            db_path = root / "library.db"
            Image.new("RGB", (16, 16), color=(255, 255, 255)).save(image_path)

            store = ImageStore(db_path, model_id="stub", prompt_version="test", feature_mode="细")
            try:
                first, first_cached = get_cached_or_extract(store, image_path, StubFeatureExtractor())
                second, second_cached = get_cached_or_extract(store, image_path, StubFeatureExtractor())
            finally:
                store.close()

            self.assertFalse(first_cached)
            self.assertTrue(second_cached)
            self.assertEqual(first.features, second.features)

    def test_storage_cache_separates_feature_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "white_long_indoor.jpg"
            db_path = root / "library.db"
            Image.new("RGB", (16, 16), color=(255, 255, 255)).save(image_path)

            fine = ImageStore(db_path, model_id="stub", prompt_version="test", feature_mode="细")
            coarse = ImageStore(db_path, model_id="stub", prompt_version="test", feature_mode="粗")
            try:
                _, fine_cached = get_cached_or_extract(fine, image_path, StubFeatureExtractor())
                _, coarse_cached = get_cached_or_extract(coarse, image_path, StubFeatureExtractor())
                self.assertEqual(fine.count(), 1)
                self.assertEqual(coarse.count(), 1)
            finally:
                fine.close()
                coarse.close()

            self.assertFalse(fine_cached)
            self.assertFalse(coarse_cached)


class MediaCollectionTests(unittest.TestCase):
    def test_iter_media_files_includes_images_and_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "a.jpg"
            video_path = root / "b.mp4"
            text_path = root / "c.txt"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            video_path.write_bytes(b"not a real video")
            text_path.write_text("skip", encoding="utf-8")

            media = iter_media_files(root)

            self.assertEqual([path.name for path in media], ["a.jpg", "b.mp4"])

    def test_collect_reference_media_dedupes_uploaded_and_folder_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "a.jpg"
            video_path = root / "b.mp4"
            Image.new("RGB", (8, 8), color=(255, 255, 255)).save(image_path)
            video_path.write_bytes(b"video")

            refs = collect_reference_media([str(video_path)], str(root), max_count=2)

            self.assertEqual(len(refs), 2)
            self.assertEqual(refs[0], video_path.resolve())

    def test_resolve_video_cover_falls_back_to_extracted_frame(self) -> None:
        try:
            ffmpeg = ffmpeg_path()
        except Exception:
            self.skipTest("ffmpeg not available")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "sample.mp4"
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=32x32:d=1",
                "-frames:v",
                "5",
                str(video_path),
            ]
            subprocess.run(command, check=True)

            cover = resolve_media_cover(video_path)

            self.assertEqual(cover.media_type, "video")
            self.assertEqual(cover.cover_source, "frame")
            self.assertTrue(cover.cover_path.exists())


class OnlineSourceTests(unittest.TestCase):
    def test_parse_pornhub_search_html_extracts_cover_title_and_link(self) -> None:
        html = """
        <ul>
          <li class="videoBox">
            <a href="/view_video.php?viewkey=abc123" title="Demo Video">
              <img data-src="//img.example.test/thumb.jpg" alt="Alt Title">
            </a>
            <span class="duration">1:23</span>
          </li>
        </ul>
        """

        results = parse_pornhub_search_html(html, "https://www.pornhub.com")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Demo Video")
        self.assertEqual(results[0].page_url, "https://www.pornhub.com/view_video.php?viewkey=abc123")
        self.assertEqual(results[0].cover_url, "https://img.example.test/thumb.jpg")
        self.assertEqual(results[0].duration, "1:23")

    def test_parse_pornhub_api_json_extracts_video_cover(self) -> None:
        response = SimpleNamespace(
            json=lambda: {
                "videos": [
                    {
                        "video_id": "abc123",
                        "title": "API Demo",
                        "url": "https://www.pornhub.com/view_video.php?viewkey=abc123",
                        "thumb": "https://img.example.test/thumb.jpg",
                        "duration": "3:21",
                    }
                ]
            }
        )

        results = parse_pornhub_api_json(response, "https://www.pornhub.com")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "API Demo")
        self.assertEqual(results[0].cover_url, "https://img.example.test/thumb.jpg")
        self.assertEqual(results[0].duration, "3:21")

    def test_pornhub_parser_ignores_sfw_homepage_recommendations(self) -> None:
        html = """
        <html><head><title>Pornhub</title></head>
        <body>
          <script>var pageKeyStat = "sfw_homepage";</script>
          <form id="search_form_sfw"></form>
          <li class="videoBox">
            <a href="/view_video.php?viewkey=wrong" title="Homepage Recommendation">
              <img data-src="//img.example.test/wrong.jpg">
            </a>
          </li>
        </body></html>
        """

        self.assertEqual(parse_pornhub_search_html(html, "https://www.pornhub.com"), [])

    def test_pornhub_search_response_rejects_home_redirect(self) -> None:
        response = SimpleNamespace(url="https://www.pornhub.com/", text='<html><title>Pornhub</title><form id="search_form_sfw"></form></html>')

        self.assertFalse(is_pornhub_search_response(response, "jk"))

    def test_pornhub_search_context_requires_result_container(self) -> None:
        html = """
        <html><head><title>Search</title></head>
        <body>
          <li class="videoBox">
            <a href="/view_video.php?viewkey=abc123" title="Loose Card">
              <img data-src="//img.example.test/thumb.jpg">
            </a>
          </li>
        </body></html>
        """

        self.assertEqual(parse_pornhub_search_html(html, "https://www.pornhub.com", require_search_context=True), [])

    def test_telegram_message_helpers_work_with_mock_objects(self) -> None:
        entity = SimpleNamespace(username="channel_name", title="Channel")
        message = SimpleNamespace(id=42, message="First line\nsecond line")

        self.assertEqual(telegram_message_url(entity, 42), "https://t.me/channel_name/42")
        self.assertEqual(telegram_message_title(entity, message), "First line")


if __name__ == "__main__":
    unittest.main()
