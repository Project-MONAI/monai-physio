---
description: Screen a GitHub PR's review comments by invoking utils/ai_agent_github_reviews.py. Fetches inline threads and PR-level reviews via gh CLI, applies accepted edits as pending working-tree changes, resolves processed threads, and writes a Markdown summary tagged by PR number.
---

Process review comments on a GitHub PR using the project's PR-review driver.

$ARGUMENTS

## Usage shape

`$ARGUMENTS` is typically just a PR number, optionally followed by flags
forwarded to `utils/ai_agent_github_reviews.py`. Examples:

- `/review-pr 42`
- `/review-pr 42 --agent claude`
- `/review-pr 42 --since-last-push`
- `/review-pr 42 --prompt-only`
- `/review-pr 42 --no-resolve`

If `$ARGUMENTS` is empty, ask the user for the PR number before doing anything.

## Preconditions

1. Confirm `gh` is installed and authenticated:
   ```powershell
   gh auth status
   ```
   If not authenticated, stop and tell the user to run `gh auth login`.

2. Confirm the current branch is **not** `main` and not detached. If it is,
   warn the user - applied edits will land on whichever branch is checked
   out. Ask before proceeding.

3. Confirm the working tree has no unstaged Python edits the user would not
   want mixed with applied review suggestions:
   ```powershell
   git status --short
   ```
   If there are pending changes, surface them and ask whether to continue.

## Run

Invoke the driver from the active `.\venv` (activate first if needed):

```powershell
python utils/ai_agent_github_reviews.py --pr <NUMBER> [flags...]
```

Defaults worth knowing:
- `--agent codex` is the script default. Pass `--agent claude` to drive
  Claude Code instead.
- Without `--no-resolve`, every processed inline-comment thread is marked
  resolved on GitHub after the agent finishes.
- With `--since-last-push`, only comments posted after this clone last saw
  the remote PR head branch move are included.
- `--prompt-only` prints the prompt without invoking an agent or resolving
  anything.

## After the run

1. Read `pr_<NUMBER>_review_summary.md` written to the repo root. Report:
   - Total comments processed.
   - APPLY / REVISE / REJECT counts.
   - Any rejection that cited a hard rule from AGENTS.md.

2. Show the resulting working-tree diff so the user can decide what to keep:
   ```powershell
   git diff --stat
   git diff
   ```

3. Do **not** stage or commit. The script applies edits as pending
   working-tree changes only; the user controls staging
   (`git add -p`) and commit.

4. Delete `pr_<NUMBER>_review_summary.md` after reporting - the working-tree
   diff is the durable record; the summary is intermediate state.

5. If the run failed:
   - "gh CLI not found" → tell the user to install via
     `winget install GitHub.cli` then `gh auth login`.
   - "claude CLI not found" → tell the user to install via
     `winget install Anthropic.ClaudeCode`, or retry with `--agent codex`.
   - Any other error: surface the script's stderr verbatim and stop.
