import json
from pathlib import Path
import tempfile
import unittest

from manage import download, installed_model_revision, missing_model_files
from settings import MODEL_REVISION, ROOT, prepare_output_directory


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

    def test_zero_byte_required_file_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model_index.json").touch()
            self.assertIn("model_index.json", missing_model_files(root))

    def test_hugging_face_metadata_identifies_legacy_download(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / ".cache/huggingface/download/model_index.json.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(MODEL_REVISION + "\netag\ntimestamp\n")
            self.assertEqual(installed_model_revision(Path(directory)), MODEL_REVISION)

    def test_custom_output_directory_must_be_dedicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unrelated.txt").write_text("private")
            with self.assertRaises(RuntimeError):
                prepare_output_directory(root)

    def test_output_directory_cannot_contain_application(self):
        with self.assertRaises(RuntimeError):
            prepare_output_directory(ROOT.parent)


if __name__ == "__main__":
    unittest.main()
