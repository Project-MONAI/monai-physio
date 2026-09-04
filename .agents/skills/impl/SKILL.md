---
description: Read relevant MONAI Physio source files, summarize current behavior, propose a brief plan, then implement the requested feature or refactor in small diffs. Calls out breaking changes and logs them in the migration guide.
---

Implement the following in the MONAI Physio repository:

$ARGUMENTS

Instructions:
1. Use `graphify query "<question>"` to locate relevant files, then read them in full.
2. Summarize current behavior in 2–4 sentences.
3. State the implementation plan in numbered steps. For non-trivial changes, pause and confirm before proceeding.
4. Implement in the smallest reviewable diff possible.
5. Update docstrings and type hints for every changed public method.
6. Run `ruff check . --fix && ruff format .` after editing Python files.
7. Prefer compatibility. Break a public API only when the change is generally
   beneficial to future users, and never via a deprecation shim, removed-symbol
   re-export, or removed-symbol stub — provide a conversion script when the
   change is substantial.
8. Explicitly note any breaking changes introduced, and append an entry for
   each to `docs/developer/migration_next.md` using the template at the bottom
   of that file.
9. Do not add features beyond what was requested.
