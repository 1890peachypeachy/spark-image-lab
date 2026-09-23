"""PE-T2I rewriter parsing, sizing, and staging checks (CPU-only, no inference)."""

import json
from pathlib import Path
import tempfile
import unittest

from rewriter import (MODEL_REVISION, RewriteError, install_error,
                      missing_model_files, parse_rewrite, size_for_ratio)


class ParseRewriteTests(unittest.TestCase):
    def test_plain_json_answer(self):
        text = '{"rewritten_prompt": "A detailed cityscape", "wh_ratio": "16:9"}'
        self.assertEqual(parse_rewrite(text), ("A detailed cityscape", "16:9"))

    def test_thinking_then_json(self):
        text = "Let me think about framing and lighting.\nThe user wants a city.\n</think>\n" \
               '{"rewritten_prompt": "A wide cinematic cityscape at dusk", "wh_ratio": "16:9"}'
        self.assertEqual(parse_rewrite(text), ("A wide cinematic cityscape at dusk", "16:9"))

    def test_chatter_without_close_tag_uses_last_json_fallback(self):
        text = 'Reasoning without a close tag... {"note": "intermediate"}\n' \
               '{"rewritten_prompt": "A calm lakeside morning", "wh_ratio": "3:2"} trailing note'
        self.assertEqual(parse_rewrite(text), ("A calm lakeside morning", "3:2"))

    def test_fenced_json_after_thinking(self):
        text = 'Some reasoning...\n```json\n{"rewritten_prompt": "A portrait", "wh_ratio": "2:3"}\n```'
        self.assertEqual(parse_rewrite(text), ("A portrait", "2:3"))

    def test_unknown_ratio_is_kept_empty(self):
        text = '{"rewritten_prompt": "A dog", "wh_ratio": "21:9"}'
        self.assertEqual(parse_rewrite(text), ("A dog", ""))

    def test_missing_ratio_is_kept_empty(self):
        text = '{"rewritten_prompt": "A cat on a windowsill"}'
        self.assertEqual(parse_rewrite(text), ("A cat on a windowsill", ""))

    def test_unusable_output_raises(self):
        with self.assertRaises(RewriteError):
            parse_rewrite("The rewriter rambled without producing JSON.")
        with self.assertRaises(RewriteError):
            parse_rewrite('{"rewritten_prompt": "", "wh_ratio": "1:1"}')

    def test_truncated_thinking_raises(self):
        with self.assertRaises(RewriteError):
            parse_rewrite("Reached the token budget mid-thought, no answer block")


class SizeForRatioTests(unittest.TestCase):
    def test_known_ratio_maps_to_card_sizes(self):
        self.assertEqual(size_for_ratio("16:9", (1024, 1024)), (2752, 1536))
        self.assertEqual(size_for_ratio("1:1", (1024, 1024)), (2048, 2048))
        self.assertEqual(size_for_ratio("9:16", (1024, 1024)), (1536, 2752))

    def test_unknown_ratio_returns_fallback(self):
        self.assertEqual(size_for_ratio("", (1234, 567)), (1234, 567))
        self.assertEqual(size_for_ratio("21:9", None), None)


class InstallCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)

    def populate(self, single_file=False):
        index = {"weight_map": {"a.weight": "model-00001.safetensors"}}
        (self.dir / "config.json").write_text("{}")
        (self.dir / "tokenizer.json").write_text("{}")
        (self.dir / "tokenizer_config.json").write_text("{}")
        (self.dir / "system_prompt.txt").write_text("system\n")
        if single_file:
            (self.dir / "model.safetensors").write_bytes(b"x")
        else:
            (self.dir / "model.safetensors.index.json").write_text(json.dumps(index))
            (self.dir / "model-00001.safetensors").write_bytes(b"x")
        (self.dir / ".spark-rewriter.json").write_text(
            json.dumps({"model": "Qwen/Qwen-Image-2.1-PE-T2I", "revision": MODEL_REVISION}))

    def test_complete_staged_dir_passes(self):
        self.populate()
        self.assertIsNone(install_error(self.dir))

    def test_single_file_variant_passes(self):
        self.populate(single_file=True)
        self.assertIsNone(install_error(self.dir))

    def test_empty_dir_reports_missing(self):
        error = install_error(self.dir)
        self.assertIsNotNone(error)
        self.assertIn("Incomplete rewriter", error or "")
        self.assertIn("config.json", missing_model_files(self.dir))

    def test_missing_shard_is_reported(self):
        self.populate()
        (self.dir / "model-00001.safetensors").unlink()
        self.assertIn("model-00001.safetensors", missing_model_files(self.dir))

    def test_revision_mismatch(self):
        self.populate()
        (self.dir / ".spark-rewriter.json").write_text(
            json.dumps({"model": "Qwen/Qwen-Image-2.1-PE-T2I", "revision": "0" * 40}))
        error = install_error(self.dir)
        self.assertIsNotNone(error)
        self.assertIn("revision mismatch", error or "")


if __name__ == "__main__":
    unittest.main()
