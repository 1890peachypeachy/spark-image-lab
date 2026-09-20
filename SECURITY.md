# Security

This private alpha is a local, trusted, single-owner application. It has no
account system or per-user history isolation. Anyone who can reach it can read
generation history, reference copies, and downloadable outputs and submit GPU
work. Do not expose it publicly or use it as an untrusted multi-tenant service.

The supported Compose configuration binds the host port to loopback. Use SSH
forwarding for remote access. Keep Gradio sharing disabled. The inference server
serves files from `outputs/`, including reference copies; that directory is not
a security boundary between local users. Uploads are limited to 25 MB per file,
ten references per generation, and Pillow's normal image safety checks.

Model loading uses local files and does not enable remote-code trust. Setup
downloads a pinned model revision. Gradio and Hugging Face telemetry are disabled.
Build/download steps still contact NVIDIA, package registries, GitHub, and Hugging
Face. Updating dependencies and reviewing their advisories is a release duty.

Report vulnerabilities privately through the repository's **Security** tab by
selecting **Report a vulnerability**. Do not open a public issue for a suspected
vulnerability, and do not attach secrets or real reference images to reports.

No production-security support guarantee is offered for alpha versions.
