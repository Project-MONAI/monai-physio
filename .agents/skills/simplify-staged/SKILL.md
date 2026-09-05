---
description: Review and simplify code in all tracked files modified since HEAD, applying open-source clarity and quality standards before committing.
---

Review and simplify the code in all files that would be staged by `git commit -a`.

$ARGUMENTS

## Goals

This is an open-source project that welcomes contributors with a range of backgrounds
and skill levels. Every change should meet the following bar before it is committed:

- **Readable** - logic is immediately clear to a developer unfamiliar with the file.
- **Focused** - each function or method does one thing; no hidden side effects.
- **Consistent** - style, naming, and patterns match the surrounding codebase.
- **Well-typed** - full type hints; no `Any` unless unavoidable and documented.
- **Documented** - every public method has a docstring that states what it does,
  its arguments, and return value.

## Instructions

1. Run `git diff HEAD` to get the full diff of tracked modified files.
   Also run `git diff HEAD --name-only` to list just the changed files.

2. For each modified Python file, read the **entire file** - not just the diff.
   Understanding the unchanged context is necessary to judge whether the new code
   fits naturally. If a path from `git diff HEAD --name-only` no longer exists on
   disk (deleted or renamed), skip it - do not attempt to read or edit it.

3. For each changed section, apply the checks below. Fix every issue found directly
   in the file using small, targeted edits. Do not refactor code that was not changed.

### Clarity checks
- [ ] Are variable and parameter names self-describing? Rename single-letter or
      cryptic names (except conventional loop indices and math variables).
- [ ] Are boolean flags or ternary chains better expressed as an early return,
      a guard clause, or a named predicate?
- [ ] Are there any magic numbers or string literals that should be named constants?
- [ ] Is any comment explaining *what* the code does (redundant) rather than *why*?
      Remove what-comments; keep why-comments.

### Reuse and abstraction checks
- [ ] Is logic duplicated across two or more changed sites? Extract a helper only
      if it will be called from at least two places *within the current change set*.
      Do not introduce speculative abstractions.
- [ ] Does the new code re-implement something already available in the existing
      codebase (check with `graphify query`) or in the standard library / dependencies
      already listed in `pyproject.toml`?

### Quality checks
- [ ] Are there missing or incorrect type annotations? Fix them.
- [ ] Does every new public method have a complete docstring (summary, Args,
      Returns)? Do not restate the fixed ITK shape / axis-order / LPS conventions.
- [ ] Are error messages descriptive enough for a contributor who has never seen
      this code to understand what went wrong and where?
- [ ] Is there dead code (unreachable branches, unused imports, stale comments)?
      Remove it.

4. After editing, run ruff only on Python files that are modified and still exist
   on disk. Use `git diff --diff-filter=d --name-only HEAD -- '*.py'` to list
   changed, non-deleted `.py` files, then pass only that list to
   `ruff check --fix` and `ruff format`. Do not run ruff project-wide - it
   may reformat files outside the current change set.

5. Report a concise summary of every change made, grouped by file. If a file needed
   no changes, say so explicitly. Do not describe changes you did not make.
