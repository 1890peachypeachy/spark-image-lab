# Benchmarks

Baseline measured September 20, 2026 on ASUS GX10, NVIDIA GB10, approximately
121.6 GiB usable unified memory, driver 580.173.02, NVIDIA PyTorch 26.08 container.
Model revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`;
Diffusers revision `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`.

BF16, 1024 x 1024, 40 steps, true CFG 1.0, KV cache enabled. One full warmup,
then batch order 1, 2, 4, 4, 2, 1, using the same prompt and seeds 42 onward.
CUDA synchronized around inference. No quantization, compilation, or step reduction.

| Batch | Trial times (s) | Mean s/image | Images/min | Peak allocated GiB | Peak reserved GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 52.716, 52.408 | 52.562 | 1.1415 | 36.915 | 38.322 |
| 2 | 101.997, 101.505 | 50.876 | 1.1793 | 42.976 | 47.332 |
| 4 | 201.069, 201.808 | 50.360 | 1.1914 | 55.687 | 61.072 |

All 14 measured images completed. Batch 4 saves about 8.8 seconds compared with
four sequential baseline images, a 4.4% throughput improvement. This establishes
that four fits for this workload, not the maximum batch size. The app keeps
single-image inference as its interactive default.

Times include prompt encoding, denoising, decode and PIL conversion, excluding
model loading and disk writes. Allocated and reserved PyTorch memory differ;
neither is total device/system usage. Warmup was 54.09 seconds and model loading
197.6 seconds. Different prompts, reference edits, larger resolutions, and other
machines require separate measurements. Do not advertise this as a universal
DGX Spark performance guarantee.

## Reproduce

Stop the app first to avoid loading two model instances. Run on the Spark:

```bash
./spark stop
docker compose run --rm --no-deps lab python benchmark_batches.py
./spark start
```

Export `LOCAL_UID=$(id -u)` and `LOCAL_GID=$(id -g)` before the direct Compose
command when your UID/GID differ from 1000. Results and images are written under
`outputs/batch-benchmark-<timestamp>/`, excluded from Git. The script uses
`demos.json` only as a reproducible benchmark fixture; demos are not UI presets.
Review a benchmark's metadata for private paths before sharing it.
