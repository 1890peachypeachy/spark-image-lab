"""Preflight and explicit, revision-pinned model download."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys

from settings import MODEL_DIR, MODEL_ID, MODEL_REVISION, OUTPUTS, prepare_output_directory

REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


def missing_model_files(directory):
    required = ["model_index.json", "LICENSE", "scheduler/scheduler_config.json",
                "processor/tokenizer.json", "processor/tokenizer_config.json",
                "processor/preprocessor_config.json", "processor/chat_template.jinja",
                "text_encoder/config.json", "text_encoder/model.safetensors.index.json",
                "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors.index.json",
                "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
    # Heretic NVFP4 variant: single-file text encoder (model.safetensors, no shard index).
    if ((directory / "text_encoder" / "model.safetensors").is_file()
            and not any(directory.joinpath("text_encoder").glob("model-0000*.safetensors"))):
        required = [n for n in required if n != "text_encoder/model.safetensors.index.json"]
    missing = [name for name in required
               if not (directory / name).is_file() or (directory / name).stat().st_size == 0]
    for name in required:
        if not name.endswith(".index.json") or name in missing:
            continue
        try:
            index = json.loads((directory / name).read_text())
            weight_map = index["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("invalid weight map")
            shards = set(weight_map.values())
            for shard in shards:
                relative = Path(name).parent / shard
                path = (directory / relative).resolve()
                if not path.is_relative_to(directory.resolve()) or not path.is_file() or path.stat().st_size == 0:
                    missing.append(str(relative))
        except (KeyError, TypeError, ValueError, OSError):
            missing.append(f"{name} (invalid)")
    return missing


def installed_model_revision(directory):
    directory = Path(directory)
    marker = directory / ".spark-model.json"
    if marker.is_file():
        try:
            identity = json.loads(marker.read_text())
            if identity.get("model") == MODEL_ID and REVISION_PATTERN.fullmatch(identity.get("revision", "")):
                return identity["revision"]
        except (AttributeError, json.JSONDecodeError, OSError):
            return None
        return None
    metadata = directory / ".cache/huggingface/download/model_index.json.metadata"
    try:
        revision = metadata.read_text().splitlines()[0].strip()
    except (IndexError, OSError):
        return None
    return revision if REVISION_PATTERN.fullmatch(revision) else None


def model_install_error(directory):
    missing = missing_model_files(directory)
    if missing:
        return f"Incomplete model. Missing or invalid: {', '.join(missing)}"
    revision = installed_model_revision(directory)
    if revision != MODEL_REVISION:
        found = revision or "unverified"
        return f"Model revision mismatch: expected {MODEL_REVISION}, found {found}. Re-run the download command."
    return None


def doctor(require_model=False):
    import torch

    failures = []
    print(f"Architecture: {platform.machine()}")
    if platform.machine() not in ("aarch64", "arm64"):
        failures.append("This build targets ARM64 DGX Spark / GB10 systems.")
    if not torch.cuda.is_available():
        failures.append("CUDA unavailable. Check the NVIDIA driver, Container Toolkit, and GPU access.")
    else:
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name}; memory: {gpu.total_memory / 2**30:.1f} GiB")
        if "GB10" not in gpu.name:
            failures.append("This release is validated for GB10, not this GPU.")
    try:
        prepare_output_directory()
        if not os.access(OUTPUTS, os.W_OK):
            failures.append("Output directory is not writable; check LOCAL_UID and LOCAL_GID.")
        print(f"Output disk free: {shutil.disk_usage(OUTPUTS).free / 2**30:.1f} GiB")
    except (OSError, RuntimeError) as error:
        failures.append(str(error))
    model_error = model_install_error(MODEL_DIR)
    print("Model: " + (model_error or "required files and revision verified"))
    if require_model and model_error:
        failures.append(model_error + " Run ./spark download --accept-model-license.")
    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    return 1 if failures else 0


def download(accepted):
    if not accepted:
        print("Read Qwen's model license first:\n"
              f"https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/LICENSE\n"
              "It limits use to research/evaluation; commercial use needs a separate license.\n"
              "Re-run with --accept-model-license only if you accept those terms.", file=sys.stderr)
        return 2
    from huggingface_hub import snapshot_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if missing_model_files(MODEL_DIR) and shutil.disk_usage(MODEL_DIR).free < 40 * 2**30:
        print("Allow at least 40 GiB free for the model download and temporary files.", file=sys.stderr)
        return 1
    snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(MODEL_DIR))
    missing = missing_model_files(MODEL_DIR)
    if missing:
        print(f"Incomplete download: {', '.join(missing)}", file=sys.stderr)
        return 1
    (MODEL_DIR / ".spark-model.json").write_text(json.dumps({"model": MODEL_ID, "revision": MODEL_REVISION}, indent=2) + "\n")
    print(f"Model ready: {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("doctor")
    check.add_argument("--require-model", action="store_true")
    fetch = sub.add_parser("download")
    fetch.add_argument("--accept-model-license", action="store_true")
    args = parser.parse_args()
    sys.exit(doctor(args.require_model) if args.command == "doctor" else download(args.accept_model_license))
