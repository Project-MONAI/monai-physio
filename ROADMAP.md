# MONAI Physio roadmap

This roadmap guides MONAI Physio development for researchers, contributors and
engineers building physiological modeling workflows. **Now** is the immediate
focus; **Next** and **Later** are non-exhaustive future directions, with no implied
delivery dates. Each item summarizes its benefit and links to a tracking issue for
acceptance criteria, dependencies, status, ownership and supporting references.

## Now

### MP-001: Publish a Slicer extension

- **Benefit:** Slicer users can export anatomy to Omniverse and apply pretrained
  motion models.
  See [#155](https://github.com/Project-MONAI/monai-physio/issues/155).

### MP-002: Complete progress reporting and optional callbacks

- **Benefit:** GUI applications can keep users informed and remain interactive during
  long processes.
  See [#156](https://github.com/Project-MONAI/monai-physio/issues/156).

### MP-003: Export inferred deformation vectors and fields

- **Benefit:** Downstream workflows can reuse inferred motion without repeating
  inference.
  See [#157](https://github.com/Project-MONAI/monai-physio/issues/157).

### MP-004: Cross-platform first use and contributor onboarding

- **Benefit:** New users and contributors get a reliable first result and a clear path
  to a small contribution.
  See [#158](https://github.com/Project-MONAI/monai-physio/issues/158).

### MP-005: Expand tutorial baselines and regression conditions

- **Benefit:** Researchers can separate regressions from data, environment and
  protocol changes.
  See [#159](https://github.com/Project-MONAI/monai-physio/issues/159).

### MP-006: AI-assisted sustainable support and maintenance

- **Benefit:** Maintainers spend less time on repetitive support and more on
  reproducible scientific issues.
  See [#160](https://github.com/Project-MONAI/monai-physio/issues/160).

### MP-007: Update ICON for upstream finetuning without patching

- **Benefit:** Users can install and finetune registration reproducibly without local
  dependency workarounds.
  See [#161](https://github.com/Project-MONAI/monai-physio/issues/161).

## Next

### MP-008: Skills-based access to key MONAI Physio functions

- **Benefit:** Researchers can invoke repeatable workflows through an AI-assistant
  skills interface.
  See [#162](https://github.com/Project-MONAI/monai-physio/issues/162).

### MP-009: Simpleware meshes for Warp integration

- **Benefit:** Simulation researchers can transfer high-quality anatomical meshes into
  Warp.
  See [#163](https://github.com/Project-MONAI/monai-physio/issues/163).

### MP-010: Warp-generated physics data and an AI surrogate

- **Benefit:** Researchers can reproduce simulation-to-training experiments and
  quantify surrogate error.
  See [#164](https://github.com/Project-MONAI/monai-physio/issues/164).

### MP-011: Characterize existing physiological motion models

- **Benefit:** Researchers understand current cardiac and respiratory model failure
  modes.
  See [#165](https://github.com/Project-MONAI/monai-physio/issues/165).

## Later

### MP-012: Vascular modeling using iterative AI methods

- **Benefit:** Researchers can explore iterative vascular reconstruction or refinement
  with measurable stopping criteria.
  See [#166](https://github.com/Project-MONAI/monai-physio/issues/166).

### MP-013: Explore brain CT perfusion for stroke research

- **Benefit:** Researchers can assess whether perfusion modeling can support useful
  stroke research questions.
  See [#167](https://github.com/Project-MONAI/monai-physio/issues/167).

### MP-014: Evaluate alternative shape representations

- **Benefit:** Researchers can compare current shape models with representations
  suited to more complex anatomy.
  See [#168](https://github.com/Project-MONAI/monai-physio/issues/168).

## Proposing work and deciding priorities

Use the existing [issue forms][new-issue] and [contributor guide][contributing].
Search existing issues/PRs first; cite an `MP-` identifier when applicable.
Describe the user problem, repository evidence, smallest useful outcome,
acceptance criteria, dependencies, data/license constraints and expected upkeep.
Questions and early ideas can use [Discussions][discussions] where available.
Do not include private patient data in public reports.

**Proposed maintainer process:** prioritize scientific correctness, reproducible
user blockers, dependency risk, breadth of benefit and available review/compute
capacity. Prefer finishing a small item over expanding the list. Record acceptance,
deferral or scope changes in a linked issue/PR; contributors opt into ownership.
Update this file when decisions change, and mark work complete only with linked
acceptance evidence. Review priorities alongside release planning; no fixed
cadence or governance policy is implied until maintainers agree.

`ROADMAP.md` is the authoritative source. The README links here and Sphinx includes
this file; edit this source rather than maintaining parallel priority lists.

## Help wanted

- **MP-004:** try the selected first tutorial on one GPU environment and submit
  exact commands, versions, expected/actual output and one actionable setup fix.
- **MP-004:** check the four documentation links at the top of `tests/README.md`
  and propose replacements that point to maintained guidance.
- **MP-005:** audit one evaluation output table for units, split provenance and
  missing run metadata; propose a bounded documentation improvement.
- **MP-001:** reproduce one exported USD animation and report viewer version,
  timing and geometry checks. Coordinate with existing PRs before starting code.

[contributing]: https://github.com/Project-MONAI/monai-physio/blob/main/CONTRIBUTING.md
[new-issue]: https://github.com/Project-MONAI/monai-physio/issues/new/choose
[discussions]: https://github.com/Project-MONAI/monai-physio/discussions
