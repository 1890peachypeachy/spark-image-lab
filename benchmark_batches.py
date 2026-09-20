import gc
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch

import lab


def host_status():
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in ("MemTotal", "MemAvailable", "SwapFree"):
            memory[name + "_gib"] = round(int(value.split()[0]) / 2**20, 3)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=10,
    )
    return {"memory": memory, "gpu_temp_power_clock_util": gpu.stdout.strip()}


def main():
    folder = lab.OUTPUTS / ("batch-benchmark-" + time.strftime("%Y%m%d-%H%M%S"))
    folder.mkdir(parents=True)
    report_path = folder / "results.json"
    report = {
        "model": "Qwen/Qwen-Image-2.1", "revision": lab.MODEL_REVISION,
        "gpu": torch.cuda.get_device_name(0),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "diffusers", "transformers")},
        "settings": {"width": 1024, "height": 1024, "steps": 40,
                     "dtype": "bfloat16", "true_cfg_scale": 1.0,
                     "use_kv_cache": True, "prompt": lab.DEMOS[0]["prompt"],
                     "mode": "same prompt, independent seeds, num_images_per_prompt",
                     "seeds": [42, 43, 44, 45], "order": [1, 2, 4, 4, 2, 1],
                     "allocator_limit_fraction": 0.8},
        "timing": "CUDA-synchronized pipeline call including text encoding, denoising, decoding and PIL conversion; excludes model load and PNG writes",
        "runs": [], "status": "loading",
    }

    def persist():
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(report_path)

    persist()
    print(f"REPORT {report_path}", flush=True)
    torch.cuda.set_per_process_memory_fraction(0.8)
    load_start = time.perf_counter()
    pipe = lab.pipeline()
    report["load_seconds"] = round(time.perf_counter() - load_start, 3)
    pipe.set_progress_bar_config(disable=True)

    def run_case(batch_size, label, seeds, save_images):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        before = host_status()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        def progress(pipeline, step, timestep, values):
            if (step + 1) % 10 == 0:
                print(f"PROGRESS {label} batch={batch_size} step={step + 1}/40 elapsed={time.perf_counter() - started:.1f}s", flush=True)
            return values

        with torch.inference_mode():
            images = pipe(
                prompt=report["settings"]["prompt"], width=1024, height=1024,
                num_inference_steps=40, num_images_per_prompt=batch_size,
                true_cfg_scale=1.0, use_kv_cache=True,
                generator=[torch.Generator("cuda").manual_seed(seed) for seed in seeds],
                callback_on_step_end=progress,
            ).images
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30
        if len(images) != batch_size:
            raise RuntimeError(f"Expected {batch_size} outputs, received {len(images)}")
        record = {
            "label": label, "batch_size": batch_size, "seeds": seeds,
            "seconds": round(elapsed, 3), "seconds_per_image": round(elapsed / batch_size, 3),
            "images_per_minute": round(60 * batch_size / elapsed, 4),
            "peak_allocated_gib": round(peak_allocated, 3),
            "peak_reserved_gib": round(peak_reserved, 3),
            "before": before, "after": host_status(), "outputs": [],
        }
        save_start = time.perf_counter()
        for index, image in enumerate(images):
            pixels = np.asarray(image.convert("RGB"))
            deviation = float(pixels.std())
            if image.size != (1024, 1024) or deviation < 5:
                raise RuntimeError(f"Invalid image: size={image.size}, pixel std={deviation}")
            name = f"{label}-seed-{seeds[index]}.png"
            if save_images:
                image.save(folder / name)
            record["outputs"].append({"file": name if save_images else None,
                                      "mode": image.mode, "pixel_std": round(deviation, 3)})
        record["validation_and_save_seconds"] = round(time.perf_counter() - save_start, 3)
        return record

    try:
        report["status"] = "warmup"
        persist()
        report["warmup"] = run_case(1, "warmup", [123], False)
        print("WARMUP " + json.dumps(report["warmup"]), flush=True)
        report["status"] = "benchmarking"
        persist()
        for index, size in enumerate(report["settings"]["order"]):
            record = run_case(size, f"run-{index + 1:02d}-batch-{size}", list(range(42, 42 + size)), True)
            report["runs"].append(record)
            persist()
            print("RESULT " + json.dumps(record), flush=True)
        report["summary"] = []
        baseline_seconds = statistics.mean(r["seconds"] for r in report["runs"] if r["batch_size"] == 1)
        for size in (1, 2, 4):
            cases = [r for r in report["runs"] if r["batch_size"] == size]
            seconds = statistics.mean(r["seconds"] for r in cases)
            report["summary"].append({
                "batch_size": size, "trials": len(cases), "mean_batch_seconds": round(seconds, 3),
                "min_batch_seconds": min(r["seconds"] for r in cases),
                "max_batch_seconds": max(r["seconds"] for r in cases),
                "seconds_per_image": round(seconds / size, 3),
                "images_per_minute": round(60 * size / seconds, 4),
                "throughput_vs_single": round(size * baseline_seconds / seconds, 4),
                "max_peak_allocated_gib": max(r["peak_allocated_gib"] for r in cases),
                "max_peak_reserved_gib": max(r["peak_reserved_gib"] for r in cases),
            })
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        persist()
        print("FINAL " + json.dumps({"status": report["status"], "summary": report.get("summary"), "report": str(report_path)}), flush=True)


if __name__ == "__main__":
    main()
