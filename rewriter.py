"""Optional PE-T2I prompt rewriter: staged weights, JSON parsing, and inference.

The official Qwen/Qwen-Image-2.1-PE-T2I checkpoint (9B, BF16) turns a brief
image request into a detailed English prompt plus a recommended aspect ratio.
Weights are large enough that they must not stay resident alongside the image
pipeline on a co-tenanted GB10 box, so the default cycle is load -> rewrite ->
unload. Set SPARK_REWRITER_KEEP_LOADED=1 to keep the model resident instead.

Top-level imports stay stdlib-only so CPU-only tests can exercise parsing,
validation, and download gating without torch or transformers.
"""

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen-Image-2.1-PE-T2I"
MODEL_REVISION = "f3ed7985c788ad75b3ab7223e0c4c51e2a43545b"
MARKER_NAME = ".spark-rewriter.json"
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")

WH_RATIO_TO_SIZE = {
    "1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400),
    "3:2": (2528, 1696), "2:3": (1696, 2528), "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

REWRITE_LOCK = threading.Lock()
_STATE = {}
_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class RewriteError(RuntimeError):
    """User-reportable rewriter failure."""


def max_new_tokens():
    return int(os.environ.get("SPARK_REWRITER_MAX_TOKENS", "4096"))


def keep_loaded():
    return os.environ.get("SPARK_REWRITER_KEEP_LOADED", "") == "1"


def required_files(directory):
    directory = Path(directory)
    required = ["config.json", "model.safetensors.index.json", "tokenizer.json",
                "tokenizer_config.json", "system_prompt.txt"]
    # Single-file checkpoint (no shard index) is also accepted.
    if ((directory / "model.safetensors").is_file()
            and not any(directory.glob("model-0000*.safetensors"))):
        required = [name for name in required if name != "model.safetensors.index.json"]
    return required


def missing_model_files(directory):
    directory = Path(directory)
    missing = [name for name in required_files(directory)
               if not (directory / name).is_file() or (directory / name).stat().st_size == 0]
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file() and not missing:
        try:
            weight_map = json.loads(index_path.read_text())["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("invalid weight map")
            for shard in set(weight_map.values()):
                path = (directory / shard).resolve()
                if not path.is_relative_to(directory.resolve()) or not path.is_file() or path.stat().st_size == 0:
                    missing.append(str(shard))
        except (KeyError, TypeError, ValueError, OSError):
            missing.append("model.safetensors.index.json (invalid)")
    return missing


def installed_revision(directory):
    marker = Path(directory) / MARKER_NAME
    if not marker.is_file():
        return None
    try:
        identity = json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    revision = identity.get("revision")
    return revision if isinstance(revision, str) and REVISION_PATTERN.fullmatch(revision) else None


def install_error(directory):
    directory = Path(directory)
    missing = missing_model_files(directory)
    if missing:
        return f"Incomplete rewriter. Missing or invalid: {', '.join(missing)}"
    revision = installed_revision(directory)
    if revision != MODEL_REVISION:
        found = revision or "unverified"
        return f"Rewriter revision mismatch: expected {MODEL_REVISION}, found {found}. Re-run the download command."
    return None


def download(directory, accepted):
    if not accepted:
        print("Read Qwen's model license first:\n"
              f"https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/LICENSE\n"
              "It limits use to research/evaluation; commercial use needs a separate license.\n"
              "Re-run with --accept-model-license only if you accept those terms.", file=sys.stderr)
        return 2
    from huggingface_hub import snapshot_download

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if missing_model_files(directory) and shutil.disk_usage(directory).free < 25 * 2**30:
        print("Allow at least 25 GiB free for the rewriter download and temporary files.", file=sys.stderr)
        return 1
    snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(directory))
    missing = missing_model_files(directory)
    if missing:
        print(f"Incomplete download: {', '.join(missing)}", file=sys.stderr)
        return 1
    (directory / MARKER_NAME).write_text(json.dumps({"model": MODEL_ID, "revision": MODEL_REVISION}, indent=2) + "\n")
    print(f"Prompt rewriter ready: {directory}")
    return 0


def parse_rewrite(text):
    """Extract {"rewritten_prompt", "wh_ratio"} from the model answer."""
    answer = text.partition("</think>")[2] if "</think>" in text else text
    answer = answer.strip()
    candidates = [answer]
    fence = _FENCE_PATTERN.search(answer)
    if fence:
        candidates.insert(0, fence.group(1))
    # Fallback: the answer may carry prefix/trailing chatter or a missing
    # </think>; scan for the last parseable JSON object and prefer it.
    decoder = json.JSONDecoder()
    for start in [match.start() for match in re.finditer(r"\{", answer)][-20:][::-1]:
        try:
            obj, _ = decoder.raw_decode(answer[start:])
        except json.JSONDecodeError:
            continue
        candidates.append(obj)
    for candidate in candidates:
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except json.JSONDecodeError:
                continue
        if isinstance(candidate, dict) and isinstance(candidate.get("rewritten_prompt"), str) \
                and candidate["rewritten_prompt"].strip():
            ratio = candidate.get("wh_ratio")
            if not isinstance(ratio, str) or ratio not in WH_RATIO_TO_SIZE:
                ratio = ""
            return candidate["rewritten_prompt"].strip(), ratio
    raise RewriteError("Rewriter returned an unusable answer (no rewritten_prompt JSON). Try again.")


def size_for_ratio(ratio, fallback):
    if ratio in WH_RATIO_TO_SIZE:
        return WH_RATIO_TO_SIZE[ratio]
    return fallback


def _load(directory):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cached = _STATE.get("model")
    if cached is not None:
        return cached
    model_error = install_error(directory)
    if model_error:
        raise RewriteError(model_error)
    if not torch.cuda.is_available():
        raise RewriteError("CUDA is unavailable. Start the container with GPU access; see docs/troubleshooting.md.")
    started = time.perf_counter()
    print("Loading PE-T2I prompt rewriter in BF16 on CUDA", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(directory), dtype=torch.bfloat16, local_files_only=True,
    ).to("cuda").eval()
    system_prompt = (Path(directory) / "system_prompt.txt").read_text().strip()
    state = (tokenizer, model, system_prompt)
    print(f"Prompt rewriter loaded in {time.perf_counter() - started:.1f}s", flush=True)
    if keep_loaded():
        _STATE["model"] = state
    return state


def _unload():
    import torch

    _STATE.pop("model", None)
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_generation(prompt, directory, budget):
    """Run one rewrite in an inner frame so model/tokenizer locals are
    destroyed before _unload() calls empty_cache()."""
    tokenizer, model, system_prompt = _load(directory)
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    import torch
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=budget,
            do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
        )
    generated = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), int(generated.shape[0])


def rewrite(prompt, directory, token_budget=None):
    """Rewrite one prompt; returns {"rewritten_prompt", "wh_ratio", "elapsed_seconds", "new_tokens"}."""
    budget = token_budget or max_new_tokens()
    started = time.perf_counter()
    with REWRITE_LOCK:
        try:
            answer, new_tokens = _run_generation(prompt, directory, budget)
        finally:
            if not keep_loaded():
                _unload()
    rewritten, ratio = parse_rewrite(answer)
    return {
        "rewritten_prompt": rewritten,
        "wh_ratio": ratio,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "new_tokens": new_tokens,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or smoke-check the PE-T2I prompt rewriter.")
    parser.add_argument("command", choices=["download"])
    parser.add_argument("--accept-model-license", action="store_true")
    args = parser.parse_args()
    from settings import REWRITER_DIR

    sys.exit(download(REWRITER_DIR, args.accept_model_license))
