# Releases, Tags, and Public Launch

Keep this repository **private** until the owner explicitly approves publication.
Pushing code, adding topics, and creating private tags must not change visibility.
No automation publishes containers, model weights, or releases.

## Versioning

Use semantic versions in `VERSION` and matching annotated Git tags prefixed with
`v`. The initial private milestone is `v0.1.0-alpha.1`. Additional private testing
milestones increment `alpha.2`, `alpha.3`, and so on. Tag only a reviewed commit
with passing checks, and do not move or overwrite an existing tag.

Before tagging, update `CHANGELOG.md`, run CPU/container tests, perform the GPU
smoke test, and record the tested hardware and software revisions. GitHub topics
describe the project; they are not version tags. Suggested topics: `dgx-spark`,
`nvidia`, `gb10`, `qwen-image`, `gradio`, `local-ai`, `image-generation`,
`image-editing`, `python`.

## Before Public Release

- [ ] Owner explicitly approves changing repository visibility.
- [ ] Fresh-clone setup verified end to end by a second tester.
- [ ] Validate on an NVIDIA-branded DGX Spark; retain GX10 results separately.
- [ ] Verify pinned model terms, third-party notices, and intended use with the owner.
- [ ] Review Git history for secrets, personal images, local addresses, and private paths.
- [ ] Capture approved screenshots with non-sensitive prompts/images.
- [ ] CPU CI, container checks, generation, reference edit, and restart-history tests pass.
- [ ] Smoke-test a bad upload and memory/disk failure handling.
- [ ] Review dependency/security advisories; enable private vulnerability reporting.
- [ ] Test documented update and rollback on a separate checkout.
- [ ] Confirm issue templates, contribution guide, topics, description, and license.
- [ ] Write release notes with known limits and measured, scoped performance claims.

Do not claim universal hardware coverage, production hardening, commercial model
rights, or a fully offline installation. Current inference is local after setup.
