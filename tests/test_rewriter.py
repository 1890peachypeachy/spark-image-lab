"""PE-T2I rewriter parsing, sizing, and staging checks (CPU-only, no inference)."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lab
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


class GenerateRewritePathTests(unittest.TestCase):
    """lab.generate(rewrite_prompt=True) provenance + sizing, inference stubbed."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)

    def fake_pipeline(self):
        class FakePipe:
            def __call__(self, **kwargs):
                from PIL import Image

                class R:
                    images = [Image.new("RGBA", (kwargs["width"], kwargs["height"]), "red")]

                return R()

        return FakePipe()

    def run_generate(self, info, apply_ratio=False):
        import sys
        import types

        if "torch" not in sys.modules:
            fake_torch = types.ModuleType("torch")
            fake_torch.__version__ = "0.0"
            fake_torch.inference_mode = lambda: __import__("contextlib").nullcontext()
            fake_cuda = types.SimpleNamespace(max_memory_allocated=lambda: 0,
                                              synchronize=lambda: None,
                                              reset_peak_memory_stats=lambda: None,
                                              empty_cache=lambda: None)
            fake_torch.cuda = fake_cuda
            fake_torch.Generator = lambda *a, **k: types.SimpleNamespace(manual_seed=lambda s: None)
            sys.modules["torch"] = fake_torch
            import importlib.metadata as _md

            _orig_version = _md.version

            def fake_version(name):
                return {"torch": "0.0", "diffusers": "0.0", "transformers": "0.0"}.get(name) or _orig_version(name)

            _md.version = fake_version
        with patch.object(lab, "OUTPUTS", self.out), \
                patch.object(lab.rewriter, "rewrite", return_value=info), \
                patch.object(lab, "PIPE", self.fake_pipeline()), \
                patch("rewriter._STATE", {"model": ("t", "m", "s")}):
            image, files, metadata = lab.generate(
                "a corgi", None, 1024, 1024, 40, 7,
                rewrite_prompt=True, apply_ratio=apply_ratio)
        return metadata

    def test_rewritten_generation_records_provenance_and_ratio_size(self):
        info = {"rewritten_prompt": "A rich rewritten prompt", "wh_ratio": "2:3",
                "elapsed_seconds": 12.5, "new_tokens": 900}
        metadata = self.run_generate(info, apply_ratio=True)
        self.assertEqual(metadata["user_prompt"], "a corgi")
        self.assertEqual(metadata["rewritten_prompt"], "A rich rewritten prompt")
        self.assertEqual(metadata["rewriter"]["wh_ratio"], "2:3")
        self.assertEqual(metadata["rewriter"]["model"], "Qwen/Qwen-Image-2.1-PE-T2I")
        self.assertEqual(metadata["rewriter"]["revision"], MODEL_REVISION)
        # 2:3 must map through WH_RATIO_TO_SIZE, overriding 1024x1024.
        self.assertEqual((metadata["width"], metadata["height"]), (1696, 2528))
        self.assertEqual(metadata["prompt"], "A rich rewritten prompt")

    def test_rewrite_without_ratio_keeps_requested_size(self):
        info = {"rewritten_prompt": "Rewritten", "wh_ratio": "",
                "elapsed_seconds": 3.0, "new_tokens": 100}
        metadata = self.run_generate(info)
        self.assertEqual((metadata["width"], metadata["height"]), (1024, 1024))
        self.assertEqual(metadata["rewriter"]["wh_ratio"], "")


if __name__ == "__main__":
    unittest.main()
