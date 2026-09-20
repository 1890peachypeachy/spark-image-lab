# Contributing

Spark Image Lab is a private alpha focused on DGX Spark / GB10 systems. Preserve
the straightforward Gradio interface. Discuss support for other hardware or
models before adding dependencies or alternate execution paths.

## Development

CPU-only checks require Python 3.12+ and no model download:

```bash
python -B -m unittest discover -s tests -v
bash -n spark
```

Tests requiring the real Gradio/PyTorch environment run in the container through
`./spark test`. GPU generation must also be checked on a GB10 machine; GitHub's
CPU CI is not a substitute. Keep test data synthetic and use temporary output
directories. Never put real prompts, reference uploads, weights, tokens, or
generated output folders in a pull request.

Use feature branches and focused commits such as `feat(history): ...`,
`fix(setup): ...`, or `docs: ...`. Run checks, explain observable behavior changes,
include sanitized desktop/mobile screenshots for UI work, and record GPU checks
or explicitly state they were not run. Preserve backward readability of stored
history. Changes to pinned dependencies or inference settings need a new benchmark.

## Review Checklist

- Inputs validated and failure cases covered.
- Existing history survives and missing references are not silently fabricated.
- Secrets, model files, personal image data, and machine-specific paths absent.
- Loopback-only host binding and disabled public sharing preserved.
- README, changelog, and tests updated when behavior changes.
- Model and third-party licenses respected; dependency notices retained.

By contributing, you agree to license your original contributions under this
repository's MIT license. Do not contribute materials you lack permission to share.
