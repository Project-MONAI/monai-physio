# CLAUDE.md

Project guidance for Claude Code in this repository.

## Role

We are developing open-source code for scientific AI libraries.

**A GPU is assumed.** Supporting CPU-only machines is not a requirement.
Design for the GPU first and use it wherever it is faster — do not add CPU
fallbacks, dtype compromises, or size limits to keep a CPU-only path viable,
and do not weaken an algorithm because a CPU could not run it.

It follows that tests and tutorials may require a GPU. Mark those that do with
`@pytest.mark.requires_gpu` so the bucket stays honest, but do not contort a
test to avoid the marker.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria
("make it work") require constant clarification.

## 5. Project-Specific Rules

- Prefer compatibility. Break a public API only when the change is generally
  beneficial to future users.
- Never add deprecation shims, removed-symbol re-exports, or removed-symbol
  stubs. When a break is substantial, ship code that automates the conversion
  instead.
- Every commit that breaks a public API must add an entry to
  `docs/developer/migration_next.md`: what changed, why it benefits future
  users, before/after code, and the conversion script (or `None needed`).
- At release time, `git mv docs/developer/migration_next.md
  docs/developer/migration_<version>.md`, retitle it, and start a fresh
  `migration_next.md` from its entry template.

---

**These guidelines are working if:** fewer unnecessary changes in diffs,
fewer rewrites due to overcomplication, and clarifying questions come
before implementation rather than after mistakes.

## Commands

**Python launcher:** Use `py` on this Windows system (not `python`).

```bash
# Install in editable mode (preferred)
uv pip install -e .[dev]

# Lint and format
ruff check . --fix && ruff format .

# Type checking
mypy src/ tests/

# All pre-commit hooks
pre-commit run --all-files

**Version bumping:** `bumpver update --patch` (or `--minor`, `--major`)

## Architecture

All classes inherit from `MONAIPhysioBase` (`monai_physio_base.py`),
which provides a shared logger. Use `self.log_info()`, `self.log_debug()`
— never `print()`.

Use graphify (see section below) to locate classes, methods, and signatures.

**Key data conventions:**

- Images: `itk.Image`, axes X, Y, Z [, T] in LPS world space (ITK's native
  frame; `itk.imread` normalizes DICOM, NIfTI, MHA, and NRRD inputs to LPS)
  stored using itk.imwrite with compression=True
— never silently squeeze or permute axes
- Surfaces: `pv.PolyData` in LPS (inherited from the source `itk.Image` via
  `itk.vtk_image_from_image`); converted to USD right-handed Y-up only at USD
  export by `vtk_to_usd.lps_points_to_usd` (USD +X=Left, +Y=Superior, +Z=Anterior)
- Labelmaps: ITK images with integer labels defined by anatomy segmenter used.
- Masks: ITK binary images
- Transforms: ITK transforms stored in `.hdf` files with compression

These conventions are fixed and hold everywhere, so this list is their single
source of truth — do not restate shape, axis order, or world space in
docstrings, comments, or test docstrings. Document only genuine deviations,
such as a raw NumPy array whose axes are reversed relative to the ITK image it
came from.

## Testing

- Fast tests (recommended for development — slow/GPU/Simpleware/tutorial
  tests are auto-skipped unless their opt-in flag is passed)
  py -m pytest tests/ -v
- Baselines in `tests/baselines/` via Git LFS — run `git lfs pull` after cloning
- `tests/conftest.py`: session-scoped fixtures chaining
  download → convert → segment → register
- `src/monai_physio/test_tools.py`: baseline comparison utilities (`TestTools`, etc.)
- Markers (all opt-in via `--run-<bucket>`): `slow`, `requires_gpu`,
  `requires_simpleware`, `tutorial`. Data-dependent tests no
  longer use a marker — they pull data through fixtures and run by default.
- `experiments/` scripts are exploratory and are not run as tests; the
  `tutorials/` scripts are the optional end-to-end suite
- **Avoid `pytest --run-slow` (or `--run-all`) unless the user explicitly
  requests it.** Those runs take far too long for an interactive session.
  Default to the fast suite and let the user invoke the slow buckets.
- Prefer images from `ROOT/data/test/slicer_heart_small` for tests
- Prefer storing results in subdirs `./results/<test_name>`


## Agents and Skills

Role-specific subagents live in `.agents/agents/`; slash-command skills in
`.agents/skills/`. See `AGENTS.md` for role-based guidance that applies across
Claude, Codex, and other AI tooling.

- `/plan` — inspect files, summarize design, produce a numbered plan (no code changes)
- `/impl` — read → summarize → plan → implement in small diffs
- `/test-feature` — propose test plan, write real-data-driven pytest tests
  with baselines
- `/doc-feature` — update docstrings
- `/check-conventions` — audit changed files against project hard rules
  (base-class, logging, coordinate frame, USD entry point, Windows mp guard,
  quoting, type hints, line length, emoji ban)
- `/simplify-staged` — readability / quality pass over `git diff HEAD`
- `/commit` — stage tracked changes, draft `<TAG>: …` message, loop until hooks pass
- `/review-pr <NUMBER>` — drive `utils/ai_agent_github_reviews.py` to triage
  a PR's review comments and apply accepted edits as pending changes

## File Operations

Use `git mv` / `git rm` — not `mv` / `rm` — to preserve history.

## Documentation Policy

Do **not** create new `.md` files unless explicitly requested.
Document via docstrings and inline comments.

Exception: the migration guide. Append to `docs/developer/migration_next.md`
whenever a commit breaks a public API, and create
`docs/developer/migration_<version>.md` plus a fresh `migration_next.md` at
release time.

## Code Style

- Double quotes for strings and docstrings
- Full type hints (`mypy` strict; `disallow_untyped_defs = true`)
- `Optional[X]` not `X | None` (ruff `UP007` suppressed)
- Max line length: 88 characters
- Follow behavior guidelines.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:

- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
