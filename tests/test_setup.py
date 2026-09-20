import json
from pathlib import Path
import tempfile
import unittest

from manage import download, missing_model_files


class SetupTests(unittest.TestCase):
    def test_download_requires_explicit_license_acceptance(self):
        self.assertEqual(download(False), 2)

    def test_missing_model_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = missing_model_files(Path(directory))
            self.assertIn("model_index.json", missing)
            self.assertIn("LICENSE", missing)

    def test_shard_presence_and_path_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = root / "text_encoder"
            encoder.mkdir()
            index = encoder / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
            self.assertIn("text_encoder/missing.safetensors", missing_model_files(root))
            index.write_text("{broken")
            self.assertIn("text_encoder/model.safetensors.index.json (invalid)", missing_model_files(root))


if __name__ == "__main__":
    unittest.main()
