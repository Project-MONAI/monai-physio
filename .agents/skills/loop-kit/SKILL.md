---
name: loop-kit
description: Generate a bounded, verified implement/verify/review loop (task contract, prompts, subagents, skills, verifier) for a given task. Use when asked to set up a loop, agent loop, or iterative refinement for a task. Does not implement the task.
---

# Loop Kit: Generate a Verified Task Loop for Codex or Claude Code

## Choose the host

Use the current coding host unless the user names another; record it in the kit. The task contract, evidence, repair limits, and review gate are host-independent. Generate only the selected host's wrappers.

- **Codex:** keep this skill in `.agents/skills/loop-kit/SKILL.md`. Invoke `$loop-kit <task description>` in Codex CLI/IDE, select the skill in the app, or reference this exact file. Generate project agents in `.codex/agents/` and the task supervisor skill in `.agents/skills/<slug>-loop/SKILL.md`.
- **Claude Code:** keep the mirrored skill in `.claude/skills/loop-kit/SKILL.md` and use `/loop-kit <task description>`. Generate its agents and supervisor skill under `.claude/`.
- Run from the intended checkout/worktree so its project guidance is loaded. A skill path does not change the workspace, permissions, or granted task scope. Never overwrite an existing task's kit or another skill with a colliding slug.
- Keep the two Loop Kit entrypoints aligned; `.agents/agents/*.md` and Claude agent YAML are not Codex custom-agent definitions.

The design follows the earlier note, *MONAI Physio: verified initialization and code-review loops*, which adapts Youssef Hosni's [How to Create Loops with Claude Code](https://levelup.gitconnected.com/how-to-create-loops-with-claude-code-a-practical-guide-to-agentic-automation-6f422390a143).

---

## Role

You are a **loop architect**. Your job is to turn the task below into a small, file-based kit that lets the selected host implement it in a **bounded, verified loop**.

**You must not implement the task itself.** Generate and check the scaffolding, then stop unless execution was explicitly requested. Phase 0 and implementation require both an approved contract and recorded authorization to execute it; neither kit generation nor contract approval alone authorizes a run.

<task>
{{Describe the task here, or reference a file. Include the goal, the repository or working area, constraints, and any known acceptance criteria.}}
</task>

## Autonomy: minimize interrupts (applies to the kit and to the run)

The loop is **automated**. A human halt is a cost; every halt must be justified.

1. **Ask everything once, up front.** Before generating the kit, anticipate every question the run could raise and put them in ONE batch (use the host's user-input tool when available, or a single list): decisions and tolerances; data/mode choices; dependency installs and environment quirks (version pins, Windows/platform gaps, plugin conflicts); approval for expensive jobs; whether the supervisor may run the verifier itself; and **standing authorizations** for predictable amendments (verifier/environment fixes, baseline-relative lint rules, schema/path naming mismatches, API differences discovered in Phase 0). Do not trickle questions between steps.
2. **Standing authorization replaces routine per-step gates, not evidence.** Record the answers in `TASK.md` and the supervisor skill. Auto-advance only under an explicit approved run scope and the review gate below. An [H] acceptance criterion requires recorded human confirmation of that criterion; permission to run is not confirmation that it passed. Hold only the affected slice for missing human evidence, a pending amendment, or material unresolved risk; do not invent approval.
3. **Self-resolve within recorded authorization, then log.** Fix verifier/environment defects, schema/name mismatches, and baseline issues only within the approved scope and paths. Record amendments and re-run checks and independent review under a new run id. Any changed control file must have a new approved or explicitly pre-authorized hash; a newly observed failure cannot silently become an accepted baseline.
4. **Stop or hold on a concrete gate:** use BLOCKED for repair-limit exhaustion, an unauthorized protected-path change, an unavailable independent reviewer or required prerequisite, or an action outside standing authorization (commit/push/publish/messages, licensed downloads, new spend). Use HOLD_FOR_HUMAN for required human evidence, material unresolved uncertainty, or an unapproved amendment. Give one consolidated decision request for the affected work.
5. **Run to completion when execution is authorized.** Continue through all slices and the final report within the approved scope; stop or hold only on the gates above.
6. **Make the generated kit say so:** `README.md`, `LOOP_INSTRUCTIONS.md`, and the supervisor skill must state these rules, include the auto-advance decision rule, and let the supervisor run the verifier when the human configured that.

## Principles every generated file must follow

1. **Roles are separate.**
   - A **worker** proposes and implements.
   - A **trusted verifier**, which the human controls, runs the checks and records evidence.
   - A **fresh-context reviewer** judges the exact output in a separate agent invocation or session without seeing the worker's reasoning. The same model is acceptable; the implementing conversation reviewing itself is not independent.
   - The **human** decides up front (decisions and standing authorizations); during the run the supervisor applies them and auto-advances.

   The worker never certifies its own success.
2. **Authority order:**
   1. the human-approved `TASK.md`, frozen by a hash;
   2. the verifier's `verification.json` for the current output hash;
   3. the reviewer's validated findings for the current output hash;
   4. `PROGRESS.md`, which is memory only.

   A line saying "done" or "approved" in a model-written file confers nothing.
3. **Small slices.** Split the work into ordered slices. Each slice gets its own acceptance criteria (ACs) and one worker run per attempt. Start the next slice only when the current one meets the review gate below; missing [H] evidence or material uncertainty holds the affected slice.
4. **Every AC has an evidence type:**
   - **[V]:** a deterministic check that the verifier executes;
   - **[R]:** reviewer-confirmed from the output;
   - **[H]:** human-confirmed.

   Prefer [V] wherever a check is possible. A required check that is missing, skipped, or not run counts as a failure, not a pass.
5. **Bounded repair.**
   - Allow at most 3 repair cycles per slice, counted from the evidence folders rather than from conversation memory.
   - Stop early on a repeated identical failure, alternating edits, a missing prerequisite, an unauthorized protected-path change, or a decision that needs the human.
   - An exhausted limit means `BLOCKED`, never success.
6. **Protected references.**
   - The worker may never edit the contract, the verifier, the reference baselines, golden files, or tolerances.
   - It may never add skips or delete assertions to make checks pass.
   - Contract changes are proposed in a `contract_amendment.md`; the supervisor may apply only explicitly pre-authorized amendment classes. All other amendments require a human decision.
7. **No external side effects by default.** Agents never:
   - commit, push, open PRs, resolve review threads, or publish anything;
   - send messages, make purchases, or download licensed data;
   - start expensive jobs without explicit approval.
8. **Evidence is immutable.** Each run writes `outputs/<run-id>/` (an output snapshot, raw logs, `verification.json`, `review.json`, `decision.md`). Closed runs are never edited. Any new edit invalidates earlier evidence.
9. **Reuse before invention.** Use the project's existing guidance (`CLAUDE.md`, `AGENTS.md`, existing skills, subagents, commands, CI, test markers). Reference them rather than duplicating them. Do not build a general-purpose orchestration framework.
10. **Prove value.** Keep a plain single-prompt version of the task as a baseline, and describe a paired comparison against it.

## Review gate (shared by both skill copies and every generated kit)

- **Independent inputs:** the reviewer gets the approved contract, exact diff/output snapshot (including untracked deliverables), verifier evidence, and relevant source context. Do not provide the worker's reasoning, summary, conversation history, or earlier reviews. For a repair, review the new snapshot independently; the supervisor separately checks closure of prior findings.
- **Current identity:** both verification and review must match the task/control hashes, base revision, slice, run id, and output hash. Require the reviewer session/invocation identity to differ from the worker's. Missing fields, malformed output, stale hashes, a failed invocation, or no independent reviewer mean BLOCKED; never silently fall back to self-review.
- **Complete assessment:** require an assessment for every applicable AC. Each [R] criterion needs reviewer evidence; each [H] criterion needs recorded human evidence. Required checks that fail, are skipped, or are missing cannot pass.
- **Findings and closure:** concrete blocking findings need a location, violated AC/invariant, failure scenario, and verification step. Record uncertain concerns as hypotheses and investigate them. A material unresolved hypothesis or unverified requirement holds the slice; a nonblocking concern may be closed with evidence and a recorded rationale. The supervisor cannot discard a concrete blocker without evidence or the applicable human decision.
- **Decision:** ADVANCE requires valid current evidence, verifier PASS for every required check, all [R]/[H] evidence present, no scope violation, no open blocking finding, no material unresolved uncertainty, no pending amendment, and approved execution scope. A required check that actually runs and FAILS is a blocking verification finding even when the reviewer has no actionable finding: record its check ID/command, violated AC, failure log, and reproduction step without inventing a code location. It routes to NEEDS_REVISION while repair budget remains and to BLOCKED when exhausted. Missing/skipped checks and other BLOCKED conditions retain precedence. NEEDS_REVISION also covers actionable reviewer findings with remaining repair budget. HOLD_FOR_HUMAN covers the human-evidence/amendment/risk gates; missing prerequisites or exhausted repairs are BLOCKED. Zero findings alone is never sufficient.
- **Invalidate after edits:** any output, base, contract, verifier, baseline, or tolerance change requires new verification and independent review; preserve prior evidence. Hashes identify evidence, not enforce operating-system isolation or reviewer independence.
- **Combined changes:** when parallel tasks affect the same codebase, individually passing reviews do not approve their combined result. Review and test the exact integrated output before claiming integration complete; keep integration within its separately approved scope and preserve the no-commit/no-push rule.

## Step 1: Investigate (read-only)

Before writing anything:
- **Find the current conventions.** Read `CLAUDE.md`, `AGENTS.md`, `README`, and the contributing docs. Inspect only relevant project agent/skill instructions under `.codex/agents/`, `.agents/skills/`, and the selected host's `.claude/` paths; do not enumerate personal configuration or read credentials/settings for this purpose. Look at `.github/workflows/`, the test layout and markers, linters, type checkers, the docs build, and how the environment is set up (venvs, worktrees, GPU, data).
- **Find what the task touches.** Identify the files, APIs, and data, the existing tests, and the riskiest invariants. Examples: units, coordinate systems, numerical tolerances, security boundaries, public APIs, migrations.
- **Classify the task.** Is it code, documentation, data analysis, research or writing, infrastructure, or mixed? The classification decides what [V] checks are possible. For non-code tasks, still prefer deterministic checks: schema validation, link and citation checks, word or section limits, recomputing numbers from source data, spellcheck, build success. Where only judgment is possible, use [R] or [H] and say so explicitly in `TASK.md`.
- **Check the installed tools.** Record the host/client version (Codex app or `codex --version`, Claude Code `claude --version`), Python, and test tools. Check that host's help and official skills/subagent documentation before relying on formats or commands. Codex app execution does not require a separately installed Claude CLI. Use native delegation when available; do not install a CLI, change global config, enable permissions, or invent headless flags to make a run work.

Ask the human **only** what can't be found in the repo: the business intent, priority trade-offs, or approvals. Anticipate the full question set and ask it **once, in one batch** (see Autonomy), recording answers as named decisions D1…Dn in the kit. Don't trickle questions later.

## Step 2: Generate the kit

Create a kit directory at `.agent-loops/<task-slug>/`, plus the agent and skill files listed in Step 3. Use the exact file names below.

| File | Contents |
|---|---|
| `README.md` | What the loop does and the one-time setup. Setup covers the isolated worktree or branch, git-ignoring `PROGRESS.md` and `outputs/`, freezing `TASK.md` with a platform-appropriate hash command (`Get-FileHash -Algorithm SHA256` on Windows), and placing the trusted hashes/verifier in an explicitly approved control location outside worker write scope. Do not assume Unix permission commands, create unapproved profile/temp files, or claim a read-only attribute provides isolation. Then give the exact commands for each phase, the optional `/goal` text, the limits, the smoke tests, the rollout order, and the baseline comparison. |
| `TASK.md` | **Status:** DRAFT, plus a contract version. Then these numbered sections: (1) the requested behavior and public interface; (2) the slices table (ID, content, dependencies); (3) the ACs: global ones (G1 no regressions against the baseline, G2 ordinary output changes stay within allowed paths; named, explicitly authorized supervisor amendments are checked separately against the trusted amendment record, G3 no weakened checks, G4 lint and format on changed files only, plus task-specific rules) and per-slice ones, each tagged [V], [R], or [H] and concrete enough to test; (4) allowed paths; (5) protected paths for the worker, where any worker change or unauthorized supervisor change means BLOCKED; explicitly pre-authorized supervisor amendments must be named and re-verified under the amendment rule; (6) approved tolerances and thresholds; (7) human decisions D1…Dn, never guessed; (8) out of scope. |
| `REFERENCE.md` | The facts the agents may rely on: APIs and signatures, data formats, domain rules, and links to sources. Every item is marked "verify against the installed code." |
| `LOOP_INSTRUCTIONS.md` | The authority order, the phases, and the per-slice cycle diagram: worker, then verifier, then reviewer, then a decision (`ADVANCE`, `NEEDS_REVISION`, `HOLD_FOR_HUMAN`, or `BLOCKED`), then the review gate below. Then come the exact decision rules, the limits (repairs, wall-clock time, one run lock, interrupted runs marked `UNVERIFIED`, one bounded retry for transient infrastructure errors only), the evidence layout, the update rule for `PROGRESS.md`, the optional `/goal` text, five workflow smoke tests, and the human handoff checklist. The smoke tests are: a planted known defect is caught; no source change gives a quiet no-op; a restart keeps findings and attempt counts; a missing prerequisite gives BLOCKED; a forbidden edit is refused. |
| `INIT_PROMPT.md` | Phase 0, which writes no task output. It records identity and isolation (repo, branch, base commit, clean tree, interpreter, tool versions) and runs an environment preflight (stop with named blockers). It maps the scoped context (exact files and lines), runs the baseline checks (record counts and pre-existing failures without fixing them), and gives a per-slice plan (files, tests per [V] AC, risks). It lists AC problems as a `contract_amendment.md` and makes recommendations for D1…Dn. It ends with `INIT_STATUS: READY \| BLOCKED`, then stops. |
| `WORKER_PROMPT.md` | Takes `SLICE`, `RUN`, and an optional `MODE=REPAIR` plus a findings path. The worker reads the slice ACs, `PROGRESS.md` (to avoid repeating failed approaches), the init report, and the installed sources. It writes tests from the ACs (named `test_<slice>_<ac>_*` or an equivalent per check), keeps changes minimal and in scope, and runs the slice's [V] checks itself, pasting the real output. The "Never" list repeats the protected-path and side-effect rules. When stuck, it writes an amendment or names a blocker and stops. In repair mode it fixes accepted findings only. It writes `worker_summary.md` in a fixed format and appends to `PROGRESS.md` without ever writing pass status. |
| `REVIEWER_PROMPT.md` | Fresh context with read-only tools. The reviewer must not read the worker's summary or earlier reviews. Inputs are the contract, the exact diff or output snapshot, `verification.json`, and the surrounding context. It works through a prioritized checklist: correctness against the ACs, the task-specific invariants from Step 1, default-path safety, contract integrity, and evidence gaps. A finding is blocking only with a location, the violated AC or invariant, a concrete failure scenario, and a verification step. Anything else is a `hypothesis`. Output is JSON: `slice`, `run_id`, `task_hash`, `control_hashes`, `base_revision`, `output_hash`, `worker_invocation_id`, `reviewer_invocation_id`, `ac_assessment[]`, `findings[]`, `unverified_requirements[]`, `scope_violations[]`, and an advisory `recommendation`. |
| `VERIFY.md` | The baseline commands, the global checks, and a table mapping each slice to its required [V] checks, plus prior slices for regression. It defines the `verification.json` schema (run, slice, task hash, control hashes, TASK hash ok, base, output hash, worker invocation id, environment, a per-check exit code and counts and status, `human_pending`, overall status, blocked reason). It includes a **reference verifier script** that the human saves outside the repo. The script checks the TASK hash; snapshots and hashes the output; checks allowed and protected paths, including untracked files; excludes only exact supervisor amendment paths validated against a trusted, worker-protected amendment record (authorization reference, supervisor identity, exact path, old/new hashes, and amendment class), and verifies their new content/control hashes separately; rejects missing/mismatched amendment evidence, worker changes to protected paths, or other out-of-scope changes; guards against diffs that add skips or remove assertions or tolerances; runs lint on changed files only and the baseline suite, tolerating only the pre-existing failures the human listed; and runs the slice checks, treating missing or skipped as BLOCKED. It writes the JSON and exits non-zero unless everything passes. It also lists what the reference script doesn't yet do. |
| `PROGRESS.md` | A template with a current-state table (slice, status, attempts n/3, last run, output hash, next action), human decisions, open findings, rejected findings, failed approaches, API differences, pre-existing baseline failures, and a compact newest-first log. A header says it is memory, not evidence. |
| `BASELINE_single_prompt.md` | A plain single-prompt version of the same task, used as the comparison arm. |

## Step 3: Generate wrappers for the selected host

All wrappers point to the shared kit; keep task logic in the kit, not copied into each wrapper. The worker must not edit the active Loop Kit, task contract, verifier, wrappers, baselines, or tolerances. The supervisor stores returned review evidence and decisions; the reviewer does not need write access to save its own report.

### Codex

Use standalone TOML files in `.codex/agents/`, each with `name`, `description`, and `developer_instructions`. Inherit the current model and reasoning settings unless the user explicitly selected others. Do not generate `sandbox_mode`, `approval_policy`, or MCP overrides in these files; Codex subagents inherit the parent permissions. "Read-only reviewer" is a role instruction, not a separate security boundary.

Generate these three task-specific wrappers, replacing `<slug>` consistently:

`.codex/agents/<slug>-worker.toml`:
```toml
name = "<slug>-worker"
description = "Implement the approved task slice and repair accepted findings."
developer_instructions = """
Read .agent-loops/<slug>/WORKER_PROMPT.md and follow the approved TASK.md.
Edit only worker-allowed output paths. Do not change the contract, verifier,
wrappers, baselines, tolerances, or active Loop Kit. Do not weaken checks.
Do not certify success, commit, push, publish, send messages, install tools,
or start expensive jobs without the required separate authorization.
"""
```

`.codex/agents/<slug>-reviewer.toml`:
```toml
name = "<slug>-reviewer"
description = "Independently review the exact output and verification evidence."
developer_instructions = """
Read .agent-loops/<slug>/REVIEWER_PROMPT.md and the approved TASK.md.
Review only the supplied output snapshot, verification evidence, and relevant
source context. Do not read worker reasoning, summaries, or earlier reviews.
Do not modify files, run mutating commands, change controls, or take external
actions. Return the required review JSON to the supervisor. Missing evidence
is not approval. You cannot replace the independent verifier.
"""
```

`.codex/agents/<slug>-initializer.toml`:
```toml
name = "<slug>-initializer"
description = "Inspect task scope and run the authorized baseline checks."
developer_instructions = """
Read .agent-loops/<slug>/INIT_PROMPT.md and the approved TASK.md.
Inspect the checkout and run only authorized baseline checks. Write no task
implementation or control changes. Return the init report to the supervisor.
Report missing prerequisites; do not install tools, commit, push, publish,
or take external actions without separate authorization.
"""
```

Generate `.agents/skills/<slug>-loop/SKILL.md` with YAML frontmatter `name: <slug>-loop` and a task-specific description. Its supervisor instructions must:
1. Verify the checkout/worktree, approved contract/control hashes, execution authorization, slice, and run ID before dispatch.
2. Use native Codex delegation to run the initializer, then each worker attempt; wait for each result. Dispatch completion is not task completion.
3. Stop worker edits, snapshot the output, and run the trusted verifier when authorized. Persist its actual result under the run ID.
4. Invoke the reviewer with fresh context and only the approved review inputs. Do not fork/inherit the worker's transcript or feed it the worker summary. If named custom roles are unavailable in that runtime, use a fresh generic subagent with the corresponding role prompt and constrained inputs; if independent context cannot be obtained, report BLOCKED.
5. Capture actual worker/reviewer invocation identities, validate returned evidence against the exact snapshot, persist it without overwriting closed runs, and apply the shared review gate. Record `decision.md`; advance only when permitted. Do not let the worker write verification/review/decision evidence. A verifier failure remains actionable even when the reviewer finds nothing.

Document invocation through `$<slug>-loop` or the app skill selector, plus a direct file-reference fallback when discovery has not refreshed. Verify the generated roles are recognized before relying on their names; otherwise use the explicit fresh-subagent fallback above. Do not enable global flags, change sandbox settings, or create persistent chats/automations as a workaround.

Current official formats: [Codex subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents) and [skill discovery](https://learn.chatgpt.com/docs/build-skills). Verify against the installed client; parsing TOML alone does not prove runtime discovery.

### Claude Code

Retain the existing Claude format when that host is selected:
- `.claude/agents/<slug>-worker.md`: Markdown/YAML with scoped edit and command tools; follows `WORKER_PROMPT.md` and the same protected-path/side-effect rules.
- `.claude/agents/<slug>-reviewer.md`: read-only tools only; follows `REVIEWER_PROMPT.md` in a fresh invocation without worker reasoning. Return the review to the supervisor.
- `.claude/agents/<slug>-initializer.md`: read-only inspection plus authorized baseline checks; follows `INIT_PROMPT.md`.
- `.claude/skills/<slug>-loop/SKILL.md`: follows the same supervisor cycle and review gate; uses verified Claude-specific agent invocation. Never substitute Codex TOML for Claude's Markdown/YAML.

Do not install hooks or auto-approve operations. Any optional hook proposal is feedback only, not enforcement.

### Parallel tasks

Use a unique slug, isolated checkout/worktree, output directories, and run lock per independent task. Name shared files/dependencies before starting concurrent work; sequence dependent changes, use distinct writable result/cache paths, and schedule GPU-heavy checks to avoid contention. Shared source data may be read-only. Each task's worker -> verifier -> reviewer cycle stays ordered; only independent tasks run concurrently. Passing feature reviews do not approve integration: test and independently review the exact combined output under its approved scope. Do not merge, commit, or push merely because the loops pass.

## Step 4: Self-check the kit, then stop

Run these checks over the generated kit and write the results to `.agent-loops/<slug>/KIT_CHECK.md`:
- Every AC has exactly one evidence tag. Every [V] AC appears in `VERIFY.md` with a concrete command or test pattern. Every slice has at least one [V] AC, or a stated reason why it can't.
- Worker-allowed and protected paths do not overlap. The kit's control files, verifier, and baselines remain protected. Exercise the verifier against a named authorized supervisor amendment with matching hashes, an unnamed protected-path change, and an amendment with mismatched hashes: only the first is eligible, after fresh verification and review; the others are BLOCKED.
- The worker and reviewer prompts both repeat the protected-path rules and the side-effect rules. The reviewer prompt forbids reading the worker's summary. Check that the supervisor cannot substitute self-review or a missing/malformed/stale review for independent evidence.
- The reference verifier script parses (`python -m py_compile` on the extracted block) and contains no network calls.
- Every command in the README exists in the installed environment, or is marked "verify."
- Decisions D1…Dn are listed in both `TASK.md` and `PROGRESS.md`.
- Exercise the decision rules with: tests PASS but no review; a review of a stale output; a required test skipped; a required test FAIL with no reviewer finding (both remaining and exhausted repair budgets); a concrete blocker despite an advisory approval; missing [H] evidence despite run permission; a fully verified low-risk slice; and individually passing parallel changes without a combined review. Record the expected decision and rationale. Use the generated gate logic when executable; otherwise label this a manual walkthrough, not an execution test.
- Validate selected-host wrappers: parse TOML for Codex and verify the required agent fields and skill frontmatter; validate Claude YAML only for that host. Confirm path/slug consistency and no overwritten existing kit. Record discovery/runtime checks separately from syntax checks. Walk through Codex with no Claude CLI, unavailable named roles with fresh native subagents, unavailable independent context, and overlapping parallel tasks.
- Keep the two Loop Kit skill copies aligned; do not rewrite existing task kits or reinterpret their approvals without a separate request.

Then reply with (stopping only if the Autonomy batch is still unanswered; otherwise continue straight into Phase 0 and the slices when the user has asked for a full run):
1. the files created;
2. the slice list, with the AC count per slice and the split between [V], [R], and [H];
3. the human decisions needed before Phase 0;
4. the selected host, client/version differences, and which discovery/delegation checks actually ran;
5. the exact next commands the human should run (freeze `TASK.md`, install the verifier, run Phase 0).

Do not run Phase 0 or start any slice until `TASK.md` is approved and execution is explicitly authorized, including when both are granted together. Contract approval alone is not execution permission. Once both are recorded, proceed autonomously within that scope under the Autonomy section.

## Rules for this generation run
- During kit generation, write only the new task kit `.agent-loops/<slug>/` and the selected host's task-specific wrappers: Codex `.codex/agents/<slug>-{worker,reviewer,initializer}.toml` plus `.agents/skills/<slug>-loop/SKILL.md`, or the corresponding Claude paths above. Do not modify source, tests, CI, or existing guidance during generation. During separately authorized execution, the worker may edit only output paths approved in `TASK.md`; the active Loop Kit, wrappers, contract, verifier, baselines, tolerances, and global configuration remain protected throughout.
- Keep each generated file short and specific to the task. If something is generic, reference this design rather than padding the file.
- If the task is too vague to write testable ACs, still produce the kit. Mark the vague ACs `[NEEDS DECISION]`, list the questions as decisions, and say so in your reply.
