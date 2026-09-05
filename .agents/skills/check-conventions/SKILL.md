---
description: Audit changed files (or a given path) against MONAI Physio's hard project rules - base-class inheritance, logging, coordinate conventions, USD entry point, Windows multiprocessing guard, quoting, type-hint style, line length, and emoji ban. Reports violations without auto-fixing.
---

Audit MONAI Physio source for hard-rule violations.

$ARGUMENTS

By default, audit every Python file modified since `HEAD`. If $ARGUMENTS names
specific files or directories, audit those instead.

## Determine the file set

```powershell
# Default: changed, non-deleted .py files since HEAD
git diff --diff-filter=d --name-only HEAD -- '*.py'
```

If a path was passed in $ARGUMENTS, expand it to all `.py` files under that
path (recursively). Skip files that no longer exist on disk.

## Rules to check

For each file, read the **entire file** (rules below depend on surrounding
context such as class inheritance), then flag every occurrence of:

### Base class and logging
- [ ] A class that orchestrates workflow / segmentation / registration / USD
      conversion but does **not** inherit from `MONAIPhysioBase`.
- [ ] A `print(` call inside the body of a class that inherits from
      `MONAIPhysioBase` (it must use `self.log_info()` / `self.log_debug()`).
      Standalone scripts and helper / data-container classes may use `print()`.

### USD / coordinate conventions
- [ ] An `import` of `monai_physio.vtk_to_usd` (or `from ... vtk_to_usd ...`)
      from a file that is **not** `src/monai_physio/convert_vtk_to_usd.py`
      and is **not** itself inside `src/monai_physio/vtk_to_usd/`.
      Experiments, CLIs, tests, and tutorials must use `ConvertVTKToUSD`.
- [ ] A docstring or comment claiming PyVista surfaces are in **RAS** - they
      are in **LPS** internally; convert to USD Y-up only at export.

### Windows multiprocessing
- [ ] A module-level instantiation of `SegmentChestTotalSegmentator` (or a
      module-level call into it) that is not guarded by
      `if __name__ == "__main__":`. Required on Windows because
      `torch.multiprocessing` re-imports the module in child workers.

### Code style
- [ ] `X | None` in a type hint (use `Optional[X]`; ruff `UP007` is suppressed).
- [ ] `Any` in a public signature without a comment explaining why.
- [ ] A docstring delimited with `'''` (use `"""`).
- [ ] A string literal using `'...'` (use `"..."`; ruff `flake8-quotes` sets
      `inline-quotes = "double"`).
- [ ] A line longer than 88 characters.
- [ ] An emoji or other non-ASCII glyph inside a `.py` file (Windows cp1252
      encoding has broken builds; keep emojis out of source).

### Public API hygiene
- [ ] A public method (no leading underscore) without a NumPy-style docstring.
- [ ] A docstring or comment that restates the fixed ITK shape, axis order, or
      LPS world space - flag it even when it also documents what the parameter
      or return value means. Only genuine deviations from the conventions in
      `CLAUDE.md` may be documented.
- [ ] A comment that narrates a bug fix's history instead of the code itself -
      what was wrong, what changed, or how it was diagnosed. That belongs in
      the commit message. Flag it unless the motivation is truly exceptional
      (a non-obvious constraint a future editor would otherwise reintroduce).

### Migration guide
- [ ] A deprecation shim, removed-symbol re-export, or removed-symbol stub
      kept solely for backward compatibility. Break the API instead, and ship
      a conversion script when the break is substantial.
- [ ] A public class, method, CLI flag, or signature that the diff renames,
      removes, or changes, with no matching entry in
      `docs/developer/migration_next.md`. The entry may record `None needed`
      for the conversion script when the break is not substantial. Report the
      missing entry against the changed line.

## Output

Group findings by file. For each finding, print:

```text
<path>:<line>  <rule short name>  <one-line excerpt>
```

End with a one-line summary: total findings per rule category.

Do **not** auto-fix. The point is to surface violations the user can decide
how to address. If `$ARGUMENTS` includes `--fix`, ask before mutating anything
and limit fixes to the trivially mechanical rules (line length is not one of
them - `ruff format` covers that).
