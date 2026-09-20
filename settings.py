"""Paths and pinned model identity shared by setup, inference, and benchmarks."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP_VERSION = (ROOT / "VERSION").read_text().strip()
MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
MODEL_DIR = Path(os.environ.get("SPARK_MODEL_DIR", ROOT / "model")).resolve()
OUTPUTS = Path(os.environ.get("SPARK_OUTPUT_DIR", ROOT / "outputs")).resolve()
