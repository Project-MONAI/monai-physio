## What and why

<!-- One or two sentences. Link the issue: Fixes #123 -->

## Release note

<!-- One line for users, or "None" for internal changes. -->

## Checklist

- [ ] Title starts with ENH:, FIX:, REF:, TST:, DOC: or MNT:
- [ ] `pytest tests/` passes locally (list any opt-in buckets you also ran:
      `--run-gpu`, `--run-slow`, `--run-tutorials`)
- [ ] Tests added or updated (real data via the session fixtures where practical)
- [ ] Docstrings and docs updated
- [ ] Public API or CLI break? Entry added to `docs/developer/migration_next.md`
      (what, why, before/after, conversion script or "None needed"), or N/A
- [ ] New data, weights or third-party code? License and citation recorded

## Not tested here

<!-- e.g. "no GPU available - requires_gpu tests not run; please add the
run-gpu-tests label" -->
