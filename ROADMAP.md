# MONAI Physio roadmap

This roadmap guides MONAI Physio development for researchers, contributors and
engineers building physiological modeling workflows. **Now** is the immediate
focus; **Next** and **Later** are non-exhaustive future directions, with no implied
delivery dates. Items are proposals awaiting maintainer review, with ownership
unassigned unless stated otherwise; each records its benefit, acceptance criteria,
dependencies and status.

## Now

### MP-001: Publish a Slicer extension

- **Benefit:** Slicer users can export anatomy to Omniverse and apply pretrained
  motion models.
- **Acceptance:** Publish an installable extension with two demonstrated paths:
  convert Slicer models to OpenUSD for Omniverse, optionally applying tissue
  visualization properties; and use pretrained models to induce respiratory or cardiac
  motion in 3D medical images and models. Document installation, compatible versions,
  checkpoints, licenses and expected outputs. Verify geometry and animation timing
  against repository conventions and reproduce both paths from clean installations.
- **Dependencies:** MP-004, MP-002 and MP-003; distributable checkpoints, Slicer
  packaging and compatible Omniverse viewer.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [USD viewing guide][viewing], [Tutorial 00][demo], [USD export PR
  #152][pr152].

### MP-002: Complete progress reporting and optional callbacks

- **Benefit:** GUI applications can keep users informed and remain interactive during
  long processes.
- **Acceptance:** Audit long-running entry points and extend the existing optional
  callback mechanism where needed. Test bounded, monotonic progress, completion and
  no-callback behavior. Document callback threading, error behavior and concurrent-run
  limitations; demonstrate worker execution with GUI-thread updates so callbacks do
  not themselves block the event loop.
- **Dependencies:** Existing shared callback API; a representative Slicer/GUI
  integration and agreement on concurrency behavior.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [MONAIPhysioBase][progress] already implements callbacks; coverage and
  GUI responsiveness still need verification.

### MP-003: Export inferred deformation vectors and fields

- **Benefit:** Downstream workflows can reuse inferred motion without repeating
  inference.
- **Acceptance:** Agree on a small format matrix (proposed: NIfTI/NRRD vector fields
  and VTK point vectors), then document and test exports from cardiac and respiratory
  inference. Preserve units, geometry, vector basis, transform direction and time
  indexing under existing conventions. Round-trip each selected format and reproduce a
  landmark or image warp in a downstream consumer.
- **Dependencies:** Existing inference and transform utilities; maintainer selection
  of formats and downstream consumers.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [deformation utilities][transforms], [motion evaluation][evaluation],
  [data conventions][agents].

### MP-004: Cross-platform first use and contributor onboarding

- **Benefit:** New users and contributors get a reliable first result and a clear path
  to a small contribution.
- **Acceptance:** Select a first tutorial (proposed: Tutorial 00) and supported OS/GPU
  matrix. Record clean runs across supported platforms with commands,
  Python/dependency/CUDA versions, data and checkpoint provenance, expected outputs
  and resource needs. Align README, installation, quickstart and contributor
  instructions; repair stale links, verify Sphinx navigation and walk through one
  cookbook extension. Clearly identify unsupported configurations; CPU-only support is
  not required.
- **Dependencies:** Accessible sample data/weights, platform testers and maintainer
  agreement on the support matrix.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [installation][install], [Tutorial 00][demo], [contributor
  guide][contributing], [docs guide][docs-guide], [cookbook][cookbook].

### MP-005: Expand tutorial baselines and regression conditions

- **Benefit:** Researchers can separate regressions from data, environment and
  protocol changes.
- **Acceptance:** Expand reviewed cardiac and respiratory tutorial baselines and a
  bounded matrix of supported environments, inputs and optional settings. Record
  held-out splits, leakage prevention, seeds, versions, units and result tables;
  justify tolerances and baseline-update criteria. Exercise failure cases, report
  skips separately and map expensive cases to existing opt-in buckets rather than
  treating regenerated baselines as passes.
- **Dependencies:** MP-004; accessible evaluation data, GPU budget and agreed
  reference cases.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [evaluation][evaluation], [leave-one-out tutorial][loo], [testing
  guide][testing], [PR #123][pr123].

### MP-006: AI-assisted sustainable support and maintenance

- **Benefit:** Maintainers spend less time on repetitive support and more on
  reproducible scientific issues.
- **Acceptance:** Pilot AI assistance for issue triage, reproducer guidance,
  documentation answers and dependency/CI review using existing tooling. Define human
  review, source citations, access boundaries and escalation; assess a bounded sample
  for correctness and maintainer time saved. Document a triage checklist and
  release-review cadence without promising response times or autonomous merges.
- **Dependencies:** Maintainer capacity, approved AI tooling and MP-004/MP-005
  evidence.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [AI assistant guide][ai-guide], [issue templates][templates],
  [workflows][workflows], [resolved issue #147][issue147].

### MP-007: Update ICON for upstream finetuning without patching

- **Benefit:** Users can install and finetune registration reproducibly without local
  dependency workarounds.
- **Acceptance:** Identify and test the latest compatible upstream ICON/uniGradICON
  version with finetuning support; record the exact version or revision. Inventory and
  remove obsolete patches or branch overrides only after clean-install, finetuning,
  checkpoint-loading and registration regression checks pass. Update dependency and
  migration guidance. If upstream support is not yet available, record the blocker
  rather than claim completion.
- **Dependencies:** Upstream finetuning API availability and compatibility; MP-005
  regression cases.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [dependency configuration][dependencies], [finetuning workflow][icon].
  The inspected configuration selects a finetuning branch; this is not evidence that a
  newer release resolves it.

## Next

### MP-008: Skills-based access to key MONAI Physio functions

- **Benefit:** Researchers can invoke repeatable workflows through an AI-assistant
  skills interface.
- **Acceptance:** Select a small function set and provide versioned skills with
  explicit inputs, prerequisites, outputs, failure handling and reproducible example
  invocations. Compare results with direct CLI/Python calls; document approval and
  data-access boundaries and avoid duplicating runtime logic.
- **Dependencies:** MP-004 and MP-002; agreed skill host/interface and supported
  function set.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [AI assistant guide][ai-guide], [contributor guide][contributing].
  Existing assistant guidance is not a delivered runtime skills interface.

### MP-009: Simpleware meshes for Warp integration

- **Benefit:** Simulation researchers can transfer high-quality anatomical meshes into
  Warp.
- **Acceptance:** Demonstrate one licensed Simpleware meshing-to-Warp import path with
  documented element types, boundaries, units and material labels. Report mesh-quality
  criteria, invalid-element handling and a reproducible import check; provide
  shareable sample artifacts where licensing permits.
- **Dependencies:** Simpleware access, compatible Warp representation and a selected
  anatomy/mesh-quality specification.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [Simpleware integration][simpleware], [physics-informed
  motion][physics]. Existing segmentation does not establish a meshing-to-Warp
  pipeline.

### MP-010: Warp-generated physics data and an AI surrogate

- **Benefit:** Researchers can reproduce simulation-to-training experiments and
  quantify surrogate error.
- **Acceptance:** Choose one bounded application, with blood flow and tissue-solid
  interaction as candidates. Generate training data in Warp with recorded physics,
  boundary conditions, parameters and seeds; train an AI approximation and evaluate
  held-out parameters/cases against solver outputs and a reference check. Publish
  runtime, error, conservation/stability checks and failure limits with reproducible
  commands.
- **Dependencies:** MP-005; selected physical formulation and reference solution,
  compute budget, and MP-009 when anatomical meshes are required.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [README solver aspirations][readme], [existing physics-informed
  training][physics]. A constitutive loss is not a Warp-generated training-data
  demonstration.

### MP-011: Characterize existing physiological motion models

- **Benefit:** Researchers understand current cardiac and respiratory model failure
  modes.
- **Acceptance:** Report held-out motion error, volume behavior, mesh quality, minimum
  Jacobian/inversions and physics-loss ablations. Include material assumptions,
  sensitivity and failure cases; distinguish constitutive-law-derived stress from
  measured stress.
- **Dependencies:** MP-005, suitable volumetric meshes and trained checkpoints.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [physics-informed implementation][physics], [Tutorial
  18][mechanics-demo], [PR #140][pr140].

## Later

### MP-012: Vascular modeling using iterative AI methods

- **Benefit:** Researchers can explore iterative vascular reconstruction or refinement
  with measurable stopping criteria.
- **Acceptance:** Scope one vascular task, define what each AI iteration updates and
  compare with a non-iterative baseline on held-out data. Report geometric/topological
  errors, convergence, failure cases and compute cost; review a bounded prototype
  proposal before implementation.
- **Dependencies:** MP-005; licensed vascular data, reference geometry and an agreed
  iterative method.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [README research directions][readme], [existing shape-model
  fitting][shape]. Iterative vascular AI is a research proposal, not an established
  capability.

### MP-013: Explore brain CT perfusion for stroke research

- **Benefit:** Researchers can assess whether perfusion modeling can support useful
  stroke research questions.
- **Acceptance:** Review one defined research question, data access and acquisition
  assumptions; propose a reproducible reference comparison with perfusion metrics,
  uncertainty and failure analysis. Record a proceed/defer decision based on technical feasibility.
- **Dependencies:** Appropriately governed CT perfusion data, domain expertise and
  MP-005 evaluation practices.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [README organ-perfusion aspiration][readme]. Brain CT perfusion and
  stroke modeling are exploratory directions.

### MP-014: Evaluate alternative shape representations

- **Benefit:** Researchers can compare current shape models with representations
  suited to more complex anatomy.
- **Acceptance:** Review applicable scikit-learn methods and recent MICCAI
  deep-learning methods, including implicit neural representations; cite selected
  papers and versions in the study proposal. Benchmark a bounded shortlist against
  current PCA-based fitting using the same held-out cases, reconstruction accuracy,
  topology, correspondence needs, compute and downstream motion compatibility.
- **Dependencies:** MP-005, representative anatomy and a literature/method shortlist
  with usable implementations and licenses.
- **Status / owner:** Proposed — awaiting maintainer review / Unassigned.
- **Evidence:** [current PCA-based fitting][shape], [evaluation][evaluation]. No
  specific recent paper or alternative method is claimed verified by this roadmap.

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

[readme]: https://github.com/Project-MONAI/monai-physio/blob/main/README.md
[agents]: https://github.com/Project-MONAI/monai-physio/blob/main/AGENTS.md
[contributing]: https://github.com/Project-MONAI/monai-physio/blob/main/CONTRIBUTING.md
[docs-guide]: https://github.com/Project-MONAI/monai-physio/blob/main/docs/README.md
[install]: https://github.com/Project-MONAI/monai-physio/blob/main/docs/installation.rst
[testing]: https://github.com/Project-MONAI/monai-physio/blob/main/docs/testing.rst
[cookbook]: https://github.com/Project-MONAI/monai-physio/tree/main/docs/cookbook
[templates]: https://github.com/Project-MONAI/monai-physio/tree/main/.github/ISSUE_TEMPLATE
[workflows]: https://github.com/Project-MONAI/monai-physio/tree/main/.github/workflows
[viewing]: https://github.com/Project-MONAI/monai-physio/blob/main/docs/viewing_usd.rst
[demo]: https://github.com/Project-MONAI/monai-physio/blob/main/tutorials/tutorial_00_lung_demo.ipynb
[evaluation]: https://github.com/Project-MONAI/monai-physio/blob/main/src/monai_physio/workflow_evaluate_movement.py
[loo]: https://github.com/Project-MONAI/monai-physio/blob/main/tutorials/tutorial_15_lung_leave_one_out.py
[physics]: https://github.com/Project-MONAI/monai-physio/blob/main/src/monai_physio/train_physicsnemo_physics_informed_motion.py
[mechanics-demo]: https://github.com/Project-MONAI/monai-physio/blob/main/tutorials/tutorial_18_duke_heart_physics_informed_motion_infer.py
[pr123]: https://github.com/Project-MONAI/monai-physio/pull/123
[pr140]: https://github.com/Project-MONAI/monai-physio/pull/140
[pr152]: https://github.com/Project-MONAI/monai-physio/pull/152
[issue147]: https://github.com/Project-MONAI/monai-physio/issues/147
[new-issue]: https://github.com/Project-MONAI/monai-physio/issues/new/choose
[discussions]: https://github.com/Project-MONAI/monai-physio/discussions

[progress]: https://github.com/Project-MONAI/monai-physio/blob/main/src/monai_physio/monai_physio_base.py
[transforms]: https://github.com/Project-MONAI/monai-physio/blob/main/src/monai_physio/process_transforms.py
[icon]: https://github.com/Project-MONAI/monai-physio/blob/main/src/monai_physio/workflow_finetune_icon_registration.py
[dependencies]: https://github.com/Project-MONAI/monai-physio/blob/main/pyproject.toml
[simpleware]: https://github.com/Project-MONAI/monai-physio/blob/main/src/monai_physio/simpleware_medical/README.md
[ai-guide]: https://github.com/Project-MONAI/monai-physio/blob/main/docs/developer/ai_assistants.rst
[shape]: https://github.com/Project-MONAI/monai-physio/blob/main/src/monai_physio/register_models_pca.py
