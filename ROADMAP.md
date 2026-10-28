# MONAI Physio roadmap

This roadmap is for researchers developing physiological AI models, scientific
software contributors, and engineers integrating medical-image-to-simulation
workflows. The research scope is subject-specific anatomy and physiological
motion, currently centered on cardiac and respiratory applications.

MONAI Physio is not validated for clinical use. It is a research and
visualization toolkit, not a medical device, and must not be used for diagnosis,
treatment planning, or clinical decision-making. Reproducible research metrics
and plausible animations do not establish clinical validity.

## Evidence and assessment

Assessed **2026-10-28**, against
[`main` at 8d875cb](https://github.com/Project-MONAI/monai-physio/tree/8d875cb9d0a24b9d6a6f8e78a62842a678ee8aee).
Accessible sources included repository guidance, source code, tutorials, Sphinx
documentation, test and workflow definitions, nine published releases, and the
available issue and PR listings (open and closed, both pages), with selected
relevant descriptions. No existing roadmap was found in the inspected files or
issue/PR listings. Discussions were unavailable through the approved connection;
this assessment does not claim to capture decisions recorded there. The latest
listed release was
[2026.09.1](https://github.com/Project-MONAI/monai-physio/releases/tag/2026.09.1);
`main` also contains later changes. Inspection is not a fresh execution or
scientific validation of these workflows.

- **Implemented in the inspected code:** segmentation and registration pipelines,
  statistical shape models, cardiac/respiratory motion surrogate training and
  inference, and OpenUSD export. Evaluation includes per-structure metrics and
  leave-one-out tutorials. Physics-informed cardiac tutorials add a neo-Hookean
  loss, ablation comparisons, and stress visualization. See the
  [tutorial source](https://github.com/Project-MONAI/monai-physio/tree/main/tutorials),
  [evaluation implementation][evaluation], [physics-informed implementation][physics],
  and merged PRs [#123][pr123], [#126][pr126], and [#152][pr152]. These are
  research capabilities, not claims of generalization or clinical efficacy.
- **Documented aspirations:** the [README][readme] describes electrophysiology,
  blood flow, organ perfusion, and Omniverse as a hub connecting other solvers,
  interactive simulation, and physical systems. Those statements do not establish
  completed integrations or validated physiological solvers.
- **Proposals below:** the ordering, acceptance criteria, and prioritization
  process are recommendations awaiting maintainer review. Supporting PRs show
  context, not approval of these priorities. Containerization [PR #129][pr129]
  is open and is not part of the inspected `main` functionality.

## Reading the priorities

**Now** proposes foundational work; **Next** depends on that foundation;
**Later** requires a separate scope decision. These are sequencing groups, not
delivery dates or commitments to start everything at once. Every item has status
**Proposed — awaiting maintainer review** and ownership **Unassigned**. An
existing PR author is not automatically the owner of a roadmap item. Keep the
`MP-` identifiers stable when moving or completing work.

## Now

### MP-001: A reliable installation-to-first-result path

- **Benefit:** a new researcher can obtain a recognizable result without guessing
  package, repository, data, or GPU requirements.
- **Acceptance:** maintainers select one first tutorial (proposed: Tutorial 00
  lung demo); record a successful clean-environment run with package/commit,
  Python, OS, GPU/CUDA and dependency versions, data/weight provenance and access
  requirements, commands, expected outputs, runtime and memory observations.
  Align README, installation and quickstart instructions, including the clone
  directory name. Explain GPU requirements per workflow and known unsupported
  configurations rather than promising universal CPU support.
- **Dependencies:** accessible sample data/weights and a supported GPU environment;
  maintainer choice of reference tutorial and platform.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [installation][install], [Tutorial 00][demo],
  [release 2026.09.1][release], and the contrasting README Tutorial 01 entry path.

### MP-002: Reproducible evaluation and regression evidence

- **Benefit:** researchers can distinguish numerical regressions from changes in
  data, environment, or evaluation protocol.
- **Acceptance:** publish one reproducible cardiac or lung evaluation recipe
  using existing metrics and held-out cases; record data provenance, subject
  splits, preprocessing, seeds, model/configuration identifiers, environment,
  units, limitations and result tables. Document how shape-model fitting avoids
  train/test leakage. Define justified numerical tolerances and baseline-update
  review criteria for a bounded regression case; report skipped tests separately
  from passes and map expensive evaluation to existing opt-in test buckets.
- **Dependencies:** MP-001; accessible evaluation data, GPU time and maintainer
  agreement on the reference case, metrics and tolerances.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [evaluation implementation][evaluation],
  [leave-one-out tutorial][loo], [testing guide][testing], [PR #123][pr123].

### MP-003: Consistent documentation and contributor onboarding

- **Benefit:** contributors can find an extension point and submit a small,
  testable change without reconciling conflicting instructions.
- **Acceptance:** walk through setup, a first result, and one existing cookbook
  extension recipe; reconcile Python/dependency and CPU/GPU statements with the
  supported path. Repair stale test-document links and distinguish tutorials from
  exploratory experiments. Verify navigation and the Sphinx build; document
  required checks and environment-dependent checks in contributor guidance.
- **Dependencies:** MP-001's supported path and existing contributor conventions.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [contributor guide][contributing], [docs build guide][docs-guide],
  [test README][test-readme], [cookbook][cookbook], [AGENTS.md][agents].

### MP-004: Sustainable support and maintenance

- **Benefit:** users get actionable support and maintainers can spend limited
  capacity on reproducible, high-impact problems.
- **Acceptance:** document a triage checklist covering reproduction, supported
  environment, severity, duplicates and next action; distinguish questions,
  defects and research proposals. Document review of existing CI/nightly failures,
  dependency changes and migration notes before releases, with a cadence agreed
  by maintainers and no promised response time. Link a resolved support example.
- **Dependencies:** maintainer capacity; MP-001's environment record and MP-002's
  distinction between test passes, skips and scientific evidence.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [issue templates][templates], [existing workflows][workflows],
  [contributor/release guide][contributing], and resolved [issue #147][issue147].

## Next

### MP-005: Characterize existing physiological motion models

- **Benefit:** researchers understand where current cardiac and respiratory
  predictions are useful and where they fail.
- **Acceptance:** document a held-out comparison using the existing motion and
  physics-informed tutorials. Report motion error, volume behavior, mesh quality,
  minimum Jacobian/inversion counts and the physics-loss ablation; expose collapse
  or failure cases, material assumptions and parameter sensitivity. Separate
  stress derived from an assumed constitutive law from measured tissue stress.
  Record limitations without claiming validation of new physiological processes.
- **Dependencies:** MP-002's evaluation protocol; valid volumetric meshes and
  trained checkpoints for the physics-informed comparison.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [physics-informed implementation][physics],
  [Tutorial 18][mechanics-demo], [PR #126][pr126], [PR #140][pr140].

### MP-006: One reproducible integration demonstration

- **Benefit:** an integrator can reproduce a complete image-to-motion-to-USD
  example and inspect the resulting animation in a documented viewer.
- **Acceptance:** package instructions for one existing workflow, its input and
  checkpoint provenance, dependency versions, commands, expected artifacts and
  viewer settings. Verify geometry, units and animation timing at the export
  boundary against the existing conventions. Document licensing/access limits
  and a repeat run by a second contributor. Evaluate PR #129 with its author
  before duplicating container work; container delivery is not a prerequisite.
- **Dependencies:** MP-001 and MP-002; compatible viewer and shareable artifacts.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [USD viewing guide][viewing], [Tutorial 00][demo],
  [open PR #129][pr129], [tetrahedral USD export PR #152][pr152].

## Later

### MP-007: Scope one additional physiology or solver connection

- **Benefit:** future expansion answers a concrete research need with an
  evaluation plan and a sustainable maintenance cost.
- **Acceptance:** maintainers review a short proposal for one process or solver
  connection, stating users, scientific assumptions, input/output contract,
  reference data or solver, licensing, evaluation criteria, compute cost and
  maintenance needs. Record a proceed/defer decision and a bounded prototype
  scope before treating it as committed implementation work.
- **Dependencies:** MP-002 and MP-005 evidence; an interested contributor and
  maintainer capacity. No ownership is inferred from partner names in the README.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [README research directions and simulation-hub aspiration][readme].

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

- **MP-001:** try the selected first tutorial on one GPU environment and submit
  exact commands, versions, expected/actual output and one actionable setup fix.
- **MP-003:** check the four documentation links at the top of `tests/README.md`
  and propose replacements that point to maintained guidance.
- **MP-002:** audit one evaluation output table for units, split provenance and
  missing run metadata; propose a bounded documentation improvement.
- **MP-006:** reproduce one exported USD animation and report viewer version,
  timing and geometry checks. Coordinate with existing PRs before starting code.

[readme]: https://github.com/Project-MONAI/monai-physio/blob/main/README.md
[agents]: https://github.com/Project-MONAI/monai-physio/blob/main/AGENTS.md
[contributing]: https://github.com/Project-MONAI/monai-physio/blob/main/CONTRIBUTING.md
[docs-guide]: https://github.com/Project-MONAI/monai-physio/blob/main/docs/README.md
[install]: https://github.com/Project-MONAI/monai-physio/blob/main/docs/installation.rst
[testing]: https://github.com/Project-MONAI/monai-physio/blob/main/docs/testing.rst
[test-readme]: https://github.com/Project-MONAI/monai-physio/blob/main/tests/README.md
[cookbook]: https://github.com/Project-MONAI/monai-physio/tree/main/docs/cookbook
[templates]: https://github.com/Project-MONAI/monai-physio/tree/main/.github/ISSUE_TEMPLATE
[workflows]: https://github.com/Project-MONAI/monai-physio/tree/main/.github/workflows
[viewing]: https://github.com/Project-MONAI/monai-physio/blob/main/docs/viewing_usd.rst
[demo]: https://github.com/Project-MONAI/monai-physio/blob/main/tutorials/tutorial_00_lung_demo.ipynb
[evaluation]: https://github.com/Project-MONAI/monai-physio/blob/main/src/monai_physio/workflow_evaluate_movement.py
[loo]: https://github.com/Project-MONAI/monai-physio/blob/main/tutorials/tutorial_15_lung_leave_one_out.py
[physics]: https://github.com/Project-MONAI/monai-physio/blob/main/src/monai_physio/train_physicsnemo_physics_informed_motion.py
[mechanics-demo]: https://github.com/Project-MONAI/monai-physio/blob/main/tutorials/tutorial_18_duke_heart_physics_informed_motion_infer.py
[release]: https://github.com/Project-MONAI/monai-physio/releases/tag/2026.09.1
[pr123]: https://github.com/Project-MONAI/monai-physio/pull/123
[pr126]: https://github.com/Project-MONAI/monai-physio/pull/126
[pr129]: https://github.com/Project-MONAI/monai-physio/pull/129
[pr140]: https://github.com/Project-MONAI/monai-physio/pull/140
[pr152]: https://github.com/Project-MONAI/monai-physio/pull/152
[issue147]: https://github.com/Project-MONAI/monai-physio/issues/147
[new-issue]: https://github.com/Project-MONAI/monai-physio/issues/new/choose
[discussions]: https://github.com/Project-MONAI/monai-physio/discussions
