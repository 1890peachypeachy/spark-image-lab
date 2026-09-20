# Troubleshooting

| Symptom | Check |
| --- | --- |
| Browser cannot connect just after start | Model loading takes several minutes. Check `./spark logs` and `./spark status`. |
| Docker reports an occupied port | Stop the old container or choose `SPARK_HTTP_PORT` in `.env`. Update the SSH tunnel too. |
| CUDA unavailable | Run `nvidia-smi` on the host, then `./spark doctor`. Check Container Toolkit and Docker GPU access. |
| Unsupported GPU or architecture | This release targets ARM64 GB10 systems. It is not a gaming-PC install. |
| Model missing/incomplete | Read the model license, then rerun `./spark download --accept-model-license`. Downloads are resumable. |
| Download/build network failure | Check connectivity and available disk. Retry the same pinned version; don't install unpinned replacements. |
| NVIDIA registry authentication required | Authenticate with NVIDIA's documented NGC credentials; never commit tokens. |
| GPU memory exhausted | Reduce dimensions/reference count. Stop other GPU workloads or duplicate Lab instances. Retry; restart if the CUDA context has failed. |
| Cannot save output | Check disk space and ownership of `outputs/`. The launcher uses your UID/GID. |
| History row has missing references | Older prototype records may lack retained uploads. Re-upload the originals. |
| Recent images absent in another browser | Reload that browser; history is shared on disk but not broadcast between sessions. |
| Old image disappears from history | Both its PNG and valid JSON must be present. Check server logs for skipped records. |
| Transparent request still has a background | Prompting does not guarantee alpha. Inspect the PNG and `alpha_extrema`; this is model behavior. |

For a bug report, include hardware model, app tag/commit, host driver, container
version, dimensions/steps/reference count, and a sanitized error. Do not upload
private prompts, reference images, full outputs directories, or credentials.
