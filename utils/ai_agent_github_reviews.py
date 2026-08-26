"""
ai_agent_github_reviews.py — Screen GitHub PR review comments with an AI agent.

Workflow:
  1. Fetch all inline review threads and PR-level review bodies via gh CLI
     (GraphQL for threads, REST for PR-level reviews).  Threads already marked
     as resolved are skipped automatically.
  2. Build a structured prompt that includes file paths, line numbers, diff
     hunks, and suggestion blocks for every comment.
  3. Invoke Claude Code or Codex non-interactively. The selected agent reads
     AGENTS.md plus its tool-specific guidance, reads the referenced source
     files for context, and for each comment decides APPLY / REVISE / REJECT
     with explicit reasoning against project conventions.
  4. Accepted edits are applied as pending working-tree changes (not committed).
  5. Each processed inline-comment thread is marked as resolved on GitHub
     (unless --no-resolve is passed).
  6. A Markdown summary tagged by PR number is written to the repo root.

Usage:
  py utils/ai_agent_github_reviews.py --pr 42
  py utils/ai_agent_github_reviews.py --pr 42 --repo owner/repo
  py utils/ai_agent_github_reviews.py --pr 42 --agent claude
  py utils/ai_agent_github_reviews.py --pr 42 --prompt-only
  py utils/ai_agent_github_reviews.py --pr 42 --since-last-push --prompt-only
  py utils/ai_agent_github_reviews.py --pr 42 --no-resolve
  py utils/ai_agent_github_reviews.py --pr 42 --prompt-only --mark-resolved
  # --no-resolve and --mark-resolved are mutually exclusive

  With --since-last-push, only inline comments and PR-level reviews created after
  the latest reflog time for refs/remotes/<remote>/<PR_head_branch> are included.
  That time is when this clone last saw the remote ref move (push or fetch), not
  necessarily the exact server push timestamp.

  With --no-resolve, inline-comment threads are NOT marked as resolved after
  the selected agent processes them (useful for dry-runs or re-processing).

  With --mark-resolved, threads are marked as resolved even when --prompt-only is
  set (no agent invocation).  Useful for bulk-dismissing comments you have
  already handled manually.

  --no-resolve and --mark-resolved are mutually exclusive; passing both is an error.

Requirements:
  - gh CLI (GitHub CLI) — not a Python package; install separately:
      Windows: winget install GitHub.cli
      Then authenticate: gh auth login
  - Claude Code CLI — https://claude.ai/code
      Windows: winget install Anthropic.ClaudeCode
  - Codex CLI — default agent
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Union, cast


# ---------------------------------------------------------------------------
# Git / repo helpers
# ---------------------------------------------------------------------------


def git_fetch(repo_root: Path, remote: str, branch: str) -> None:
    """Run ``git fetch <remote> <branch>``, printing progress."""
    print(f"[*] Fetching {remote}/{branch} ...")
    try:
        subprocess.run(
            ["git", "fetch", remote, branch],
            check=True,
            cwd=repo_root,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] git fetch {remote} {branch} failed (exit {exc.returncode}).")
        sys.exit(exc.returncode)


def get_repo_root() -> Path:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            text=True,
            capture_output=True,
        )
        return Path(out.stdout.strip())
    except subprocess.CalledProcessError:
        print("[ERROR] Not inside a Git repository.")
        sys.exit(1)


def get_repo_slug(repo_root: Path) -> str:
    """Derive owner/repo from the git remote URL (upstream, falling back to origin)."""
    for remote_name in ("upstream", "origin"):
        try:
            out = subprocess.run(
                ["git", "remote", "get-url", remote_name],
                check=True,
                text=True,
                capture_output=True,
                cwd=repo_root,
            )
            remote = out.stdout.strip()
            break
        except subprocess.CalledProcessError:
            continue
    else:
        print("[ERROR] Could not read git remote URL from 'upstream' or 'origin'.")
        sys.exit(1)
    m = re.search(r"[:/]([^/:]+/[^/:]+?)(?:\.git)?$", remote)
    if not m:
        print(f"[ERROR] Cannot parse owner/repo from remote: {remote}")
        sys.exit(1)
    return m.group(1)


def parse_github_datetime(iso_str: str) -> datetime:
    """Parse GitHub API timestamps (may end with Z)."""
    s = iso_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _parse_reflog_timestamp(line: str) -> datetime:
    """Parse the @{...} timestamp from one `git reflog show -1` line."""
    m = re.search(r"@\{([^}]+)\}", line)
    if not m:
        raise ValueError(f"no reflog timestamp in line: {line!r}")
    return datetime.fromisoformat(m.group(1).strip())


def get_remote_reflog_cutoff(repo_root: Path, remote: str, head_ref: str) -> datetime:
    """
    Latest reflog time for refs/remotes/<remote>/<head_ref> (when the ref last
    moved in this clone).
    """
    ref = f"refs/remotes/{remote}/{head_ref}"
    cmd = ["git", "reflog", "show", "-1", "--date=iso-strict", ref]
    try:
        out = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
            cwd=repo_root,
            encoding="utf-8",
        )
    except FileNotFoundError:
        print('[ERROR] "git" not found in PATH.')
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or "").strip()
        print(f"[ERROR] Could not read reflog for {ref}.")
        if err:
            print(f"        {err}")
        print(f"        Hint: git fetch {remote} {head_ref}")
        sys.exit(1)

    line = out.stdout.strip().splitlines()
    if not line or not line[0].strip():
        print(f"[ERROR] Empty reflog for {ref}.")
        print(f"        Hint: git fetch {remote} {head_ref}")
        sys.exit(1)

    try:
        return _parse_reflog_timestamp(line[0])
    except ValueError as exc:
        print(f"[ERROR] Could not parse reflog timestamp: {exc}")
        sys.exit(1)


def filter_since_cutoff(
    thread_comments: list[dict],
    reviews: list[dict],
    cutoff: datetime,
) -> tuple[list[dict], list[dict]]:
    """
    Keep thread comments with created_at > cutoff and reviews with
    submitted_at > cutoff (submitted_at required).
    """
    filtered_inline: list[dict] = []
    for c in thread_comments:
        created = c.get("created_at")
        if not created:
            continue
        if parse_github_datetime(created) > cutoff:
            filtered_inline.append(c)

    filtered_reviews: list[dict] = []
    for r in reviews:
        submitted = r.get("submitted_at")
        if not submitted:
            continue
        if parse_github_datetime(submitted) > cutoff:
            filtered_reviews.append(r)

    return filtered_inline, filtered_reviews


# ---------------------------------------------------------------------------
# GitHub API helpers (via gh CLI)
# ---------------------------------------------------------------------------


def _gh_api(endpoint: str, *, paginate: bool = False) -> list | dict:
    """Call `gh api` and return parsed JSON. Merges paginated arrays."""
    cmd = ["gh", "api"]
    if paginate:
        cmd.append("--paginate")
    cmd.append(endpoint)
    try:
        out = subprocess.run(
            cmd, check=True, text=True, capture_output=True, encoding="utf-8"
        )
    except FileNotFoundError:
        print('[ERROR] "gh" CLI not found. Install from https://cli.github.com')
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.strip() if exc.stderr else ""
        print(f"[ERROR] gh api {endpoint} failed.")
        if msg:
            print(f"        {msg}")
        sys.exit(exc.returncode)

    raw = out.stdout.strip()
    if not raw:
        return []

    # gh --paginate emits concatenated JSON arrays; normalize to one list.
    if paginate:
        try:
            merged: list = []
            decoder = json.JSONDecoder()
            pos = 0
            while pos < len(raw):
                obj, pos = decoder.raw_decode(raw, pos)
                if isinstance(obj, list):
                    merged.extend(obj)
                else:
                    merged.append(obj)
                while pos < len(raw) and raw[pos] in " \t\n\r":
                    pos += 1
            return merged
        except json.JSONDecodeError:
            pass  # Fall through to standard parse

    return cast(Union[list, dict], json.loads(raw))


def _gh_graphql(query: str, variables: dict) -> dict:
    """
    Execute a GitHub GraphQL query/mutation via ``gh api graphql --input -``.

    Sends the query as a JSON body on stdin to avoid shell escaping and
    Windows command-line length limits.  Exits on network or GraphQL errors.
    """
    payload = json.dumps({"query": query, "variables": variables})
    try:
        out = subprocess.run(
            ["gh", "api", "graphql", "--input", "-"],
            input=payload,
            check=True,
            text=True,
            capture_output=True,
            encoding="utf-8",
        )
    except FileNotFoundError:
        print('[ERROR] "gh" CLI not found. Install from https://cli.github.com')
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.strip() if exc.stderr else ""
        print("[ERROR] gh api graphql failed.")
        if msg:
            print(f"        {msg}")
        sys.exit(exc.returncode)

    result: dict = json.loads(out.stdout.strip())
    if "errors" in result:
        print(f"[ERROR] GraphQL errors: {result['errors']}")
        sys.exit(1)
    return result


# GraphQL query — fetches one page of review threads (up to 100) with the
# first 50 comments per thread.  Both connections include pageInfo so the
# caller can paginate until exhausted.
_REVIEW_THREADS_QUERY = """
query GetPRReviewThreads(
  $owner: String!, $name: String!, $number: Int!, $after: String
) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          isResolved
          comments(first: 50) {
            pageInfo { hasNextPage endCursor }
            nodes {
              databaseId
              path
              line
              originalLine
              body
              createdAt
              diffHunk
              author { login }
            }
          }
        }
      }
    }
  }
}
"""

# Used when a thread's comment list was truncated in the main query.
_THREAD_COMMENTS_QUERY = """
query GetThreadComments($threadId: ID!, $after: String) {
  node(id: $threadId) {
    ... on PullRequestReviewThread {
      comments(first: 50, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          databaseId
          path
          line
          originalLine
          body
          createdAt
          diffHunk
          author { login }
        }
      }
    }
  }
}
"""

_RESOLVE_THREAD_MUTATION = """
mutation ResolveReviewThread($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) {
    thread { id isResolved }
  }
}
"""


def _parse_comment(c: dict, thread_id: str) -> dict:
    """Normalise a GraphQL comment node into the shape the rest of the code expects."""
    return {
        "id": c.get("databaseId"),
        "path": c.get("path", ""),
        "line": c.get("line"),
        "original_line": c.get("originalLine"),
        "body": c.get("body", ""),
        "created_at": c.get("createdAt", ""),
        "diff_hunk": c.get("diffHunk", ""),
        "user": {"login": (c.get("author") or {}).get("login", "unknown")},
        "_thread_id": thread_id,
    }


def _fetch_remaining_comments(thread_id: str, after: str) -> list[dict]:
    """
    Fetch comment pages for *thread_id* starting after *after* cursor.

    Called only when the comments connection returned by the main threads
    query reports ``hasNextPage = true``.
    """
    comments: list[dict] = []
    cursor: str | None = after
    while cursor:
        result = _gh_graphql(
            _THREAD_COMMENTS_QUERY, {"threadId": thread_id, "after": cursor}
        )
        conn = (result.get("data", {}).get("node") or {}).get("comments", {})
        for c in conn.get("nodes", []):
            comments.append(_parse_comment(c, thread_id))
        page_info = conn.get("pageInfo", {})
        cursor = page_info.get("endCursor") if page_info.get("hasNextPage") else None
    return comments


def fetch_review_threads(pr_number: int, repo: str) -> list[dict]:
    """
    Return all review threads for a PR via GraphQL, paginating both the
    thread list and per-thread comments until exhausted.

    Each entry has:
      ``id``         — GraphQL node ID (used for resolution mutation)
      ``isResolved`` — bool
      ``comments``   — list of comment dicts (keys: path, line, original_line,
                       diff_hunk, body, created_at, user.login, _thread_id)
    """
    owner, name = repo.split("/", 1)
    threads: list[dict] = []
    thread_cursor: str | None = None

    while True:
        result = _gh_graphql(
            _REVIEW_THREADS_QUERY,
            {"owner": owner, "name": name, "number": pr_number, "after": thread_cursor},
        )
        threads_conn = (
            result.get("data", {})
            .get("repository", {})
            .get("pullRequest", {})
            .get("reviewThreads", {})
        )

        for t in threads_conn.get("nodes", []):
            thread_id: str = t["id"]
            comments_conn = t.get("comments", {})

            comments = [
                _parse_comment(c, thread_id) for c in comments_conn.get("nodes", [])
            ]

            # Paginate comments if the first page was truncated.
            if comments_conn.get("pageInfo", {}).get("hasNextPage"):
                comments.extend(
                    _fetch_remaining_comments(
                        thread_id, comments_conn["pageInfo"]["endCursor"]
                    )
                )

            threads.append(
                {
                    "id": thread_id,
                    "isResolved": t.get("isResolved", False),
                    "comments": comments,
                }
            )

        page_info = threads_conn.get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            break
        thread_cursor = page_info["endCursor"]

    return threads


def fetch_pr_data(pr_number: int, repo: str) -> dict:
    result = _gh_api(f"repos/{repo}/pulls/{pr_number}")
    assert isinstance(result, dict)
    return result


def fetch_reviews(pr_number: int, repo: str) -> list[dict]:
    result = _gh_api(f"repos/{repo}/pulls/{pr_number}/reviews", paginate=True)
    return result if isinstance(result, list) else []


def resolve_review_threads(thread_ids: list[str], repo: str) -> None:
    """
    Mark each thread in *thread_ids* as resolved via the GitHub GraphQL API.

    Resolution failures print a warning but do not abort; other threads are
    still attempted.  PR-level reviews have no resolution concept and are not
    handled here.
    """
    if not thread_ids:
        return
    print(f"[*] Resolving {len(thread_ids)} inline-comment thread(s)...")
    owner, name = repo.split("/", 1)  # noqa: F841 — kept for future use
    for tid in thread_ids:
        try:
            _gh_graphql(_RESOLVE_THREAD_MUTATION, {"threadId": tid})
            print(f"    Resolved thread {tid}")
        except SystemExit:
            # _gh_graphql calls sys.exit on error; catch so we can continue.
            print(f"    [WARNING] Could not resolve thread {tid} — skipping.")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _extract_suggestion(body: str) -> tuple[str, str]:
    """
    Split a GitHub review comment body into (suggestion_code, note_text).

    GitHub / CodeRabbit suggestion blocks use the fenced form:
        ```suggestion
        <replacement lines>
        ```
    Returns ('', body) when no suggestion block is present.
    """
    pattern = re.compile(r"```suggestion\r?\n(.*?)```", re.DOTALL)
    m = pattern.search(body)
    if not m:
        return "", body
    suggestion = m.group(1).rstrip("\n")
    note = pattern.sub("", body).strip()
    return suggestion, note


def _format_review_bodies(reviews: list[dict]) -> str:
    relevant = [
        r
        for r in reviews
        if r.get("body", "").strip() and r.get("state") not in ("PENDING", "")
    ]
    if not relevant:
        return ""
    parts = ["## PR-level review comments\n"]
    for i, r in enumerate(relevant, 1):
        reviewer = r["user"]["login"]
        state = r.get("state", "")
        parts.append(f"### Review {i} — {reviewer} ({state})")
        parts.append(r["body"].strip())
        parts.append("")
    return "\n".join(parts)


def _format_inline_comments(comments: list[dict]) -> str:
    if not comments:
        return ""
    parts = ["## Inline review comments\n"]
    for i, c in enumerate(comments, 1):
        reviewer = c["user"]["login"]
        path = c["path"]
        line = c.get("line") or c.get("original_line", "?")
        diff_hunk = c.get("diff_hunk", "")
        suggestion, note = _extract_suggestion(c["body"])

        parts.append(f"### Inline comment {i}")
        parts.append(f"- **Reviewer:** `{reviewer}`")
        parts.append(f"- **File:** `{path}`")
        parts.append(f"- **Line:** {line}")
        if diff_hunk:
            parts.append(f"\nDiff context:\n```diff\n{diff_hunk}\n```")
        if suggestion:
            parts.append("\nSuggested replacement (replaces the highlighted lines):")
            parts.append(f"```\n{suggestion}\n```")
        if note:
            parts.append(f"\nReviewer note: {note}")
        parts.append("")
    return "\n".join(parts)


def _agent_specific_guidance_file(agent: str) -> str:
    """Return the conventional guidance filename for the selected AI agent."""
    if agent == "claude":
        return "CLAUDE.md"
    if agent == "codex":
        return "CODEX.md"
    raise ValueError(f"Unsupported AI agent: {agent}")


def _agent_guidance_files(agent: str, repo_root: Path) -> list[str]:
    """Return existing repo guidance files relevant to the selected AI agent."""
    files = ["AGENTS.md"]
    agent_file = _agent_specific_guidance_file(agent)
    if (repo_root / agent_file).exists():
        files.append(agent_file)
    return files


def _agent_scoped_guidance_files(agent: str, repo_root: Path) -> list[str]:
    """Return existing scoped guidance files relevant to the selected AI agent."""
    agent_file = _agent_specific_guidance_file(agent)
    scoped_file = Path("src") / "physiotwin4d" / "vtk_to_usd" / agent_file
    if (repo_root / scoped_file).exists():
        return [scoped_file.as_posix()]
    return []


def build_prompt(
    pr_number: int,
    pr_data: dict,
    reviews: list[dict],
    thread_comments: list[dict],
    summary_filename: str,
    agent: str,
    repo_root: Path,
) -> str:
    title = pr_data.get("title", f"PR #{pr_number}")
    branch = pr_data.get("head", {}).get("ref", "unknown")
    base = pr_data.get("base", {}).get("ref", "unknown")
    guidance_files = _agent_guidance_files(agent, repo_root)
    scoped_guidance_files = _agent_scoped_guidance_files(agent, repo_root)
    guidance_list = "\n".join(f"- `{path}`" for path in guidance_files)
    guidance_names = " / ".join(guidance_files)
    scoped_guidance_list = "\n".join(f"  - `{path}`" for path in scoped_guidance_files)
    scoped_guidance_section = (
        f"\n- If any comment touches `vtk_to_usd/`, also read:\n{scoped_guidance_list}"
        if scoped_guidance_files
        else ""
    )
    guidance_block = f"{guidance_list}{scoped_guidance_section}"

    review_bodies = [
        r
        for r in reviews
        if r.get("body", "").strip() and r.get("state") not in ("PENDING", "")
    ]
    total = len(thread_comments) + len(review_bodies)

    comments_block = "\n".join(
        filter(
            None,
            [
                _format_review_bodies(reviews),
                _format_inline_comments(thread_comments),
            ],
        )
    )

    prompt = textwrap.dedent(f"""\
        You are screening GitHub PR #{pr_number}: "{title}"
        Branch: `{branch}` -> `{base}`
        Total comments to assess: {total}

        ## Step 1 — Read project standards

        Before assessing any comment, read these files:
        __GUIDANCE_BLOCK__

        ## Step 2 — Assess each comment

        For every comment below, in order:

        1. Read the referenced source file (`path`, near `line`) to understand
           the full context — do not rely solely on the diff hunk.
        2. Decide:
           - **APPLY** — suggestion is correct and consistent with agent guidance.
             Apply it as-is.
           - **REVISE** — directionally right but conflicts with repo conventions.
             Apply your corrected version.
           - **REJECT** — wrong, unnecessary, or conflicts with explicit project rules.
             Do not edit the file. State the specific rule or reason.
        3. For APPLY / REVISE: edit the file to make the change.
           Do NOT run git add, git commit, or any git staging commands.
           Leave all edits as pending working-tree modifications only.

        Rejection triggers (from {guidance_names} — treat these as hard rules):
        - Introduces `X | None` instead of `Optional[X]` (ruff UP007 is suppressed)
        - Adds a deprecation shim, removed-symbol re-export, or removed-symbol
          stub instead of a `docs/developer/migration_next.md` entry (that entry
          needs a conversion script only when the break is substantial; it may
          record `None needed` otherwise)
        - Adds error handling for internal states that cannot happen
        - In classes that inherit from `PhysioTwin4DBase`, uses `print()` instead
          of `self.log_info()` / `self.log_debug()`
        - New runtime workflow / segmentation / registration class does not inherit
          from `PhysioTwin4DBase`; helper/data/container classes need not
        - Adds features or abstractions beyond what was requested
        - Calls `vtk_to_usd` internals from outside `convert_vtk_to_usd.py`
        - Applies coordinate conversion (LPS->USD Y-up) more than once, or
          treats internal PyVista surfaces as RAS (they are LPS)
        - Uses emojis in `.py` files
        - Omits the Windows `if __name__ == "__main__":` guard in scripts that
          instantiate `SegmentChestTotalSegmentator`
        - Exceeds 88-character line length

        ## Step 3 — Write summary

        After processing all {total} comment(s), write `{summary_filename}`
        to the repository root using this exact structure:

        ```markdown
        # PR #{pr_number} Review Summary

        **PR:** {title}
        **Branch:** `{branch}` -> `{base}`
        **Reviewed:** <today's date>
        **Comments processed:** {total}

        ## Results

        | # | File | Line | Reviewer | Decision | Reasoning |
        |---|------|------|----------|----------|-----------|
        | 1 | `path/to/file.py` | 42 | reviewer | APPLIED | one sentence |

        ## Applied changes

        For each APPLIED or REVISED item: file name, what changed, and why it
        was accepted or how it was adjusted.

        ## Rejected suggestions

        For each REJECTED item: what was suggested, and the specific agent
        guidance rule or reasoning that led to rejection.

        ## Observations

        Patterns across the review (recurring style disagreements, areas where
        the reviewer's model of the codebase differs from reality, etc.).
        ```

        ---

        {comments_block}
        """)
    return prompt.replace("__GUIDANCE_BLOCK__", guidance_block)


# ---------------------------------------------------------------------------
# AI agent invocation
# ---------------------------------------------------------------------------


def invoke_ai_agent(prompt: str, repo_root: Path, agent: str) -> None:
    """Invoke the selected AI agent non-interactively."""
    if agent == "claude":
        invoke_claude(prompt, repo_root)
    elif agent == "codex":
        invoke_codex(prompt, repo_root)
    else:
        raise ValueError(f"Unsupported AI agent: {agent}")


def invoke_claude(prompt: str, repo_root: Path) -> None:
    """
    Invoke Claude Code non-interactively via stdin.

    Uses stdin rather than a CLI argument to avoid Windows CreateProcess
    command-line length limits (~32 KB). Claude's output streams to the
    terminal so the developer can follow along.
    """
    print("[*] Invoking Claude Code to screen review comments...")
    print("    Claude Code will read source files, assess each suggestion, apply")
    print("    accepted edits, and write the summary. This may take a minute.\n")

    try:
        subprocess.run(
            ["claude", "--print", "--allowedTools", "Read,Write,Edit,Glob,Grep"],
            input=prompt,
            text=True,
            encoding="utf-8",
            cwd=repo_root,
            check=True,
        )
    except FileNotFoundError:
        print('[ERROR] "claude" CLI not found.')
        print("        Install Claude Code: https://claude.ai/code")
        _save_prompt_fallback(prompt, repo_root, agent="claude")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f'[ERROR] "claude" exited with status {exc.returncode}.')
        _save_prompt_fallback(prompt, repo_root, agent="claude")
        sys.exit(exc.returncode)


def invoke_codex(prompt: str, repo_root: Path) -> None:
    """
    Invoke Codex CLI non-interactively.

    The full review prompt is saved to a repo-local file first so the command
    line remains short on Windows.
    """
    prompt_path = _save_prompt_file(prompt, repo_root)
    instruction = (
        f"Read {prompt_path.name} in the repository root and carry out the "
        "GitHub review workflow exactly as described there. Apply accepted "
        "edits as pending working-tree changes only, write the requested "
        "summary, and do not stage or commit files."
    )

    print("[*] Invoking Codex to screen review comments...")
    print(f"    Review prompt saved to {prompt_path.name}.")
    print("    Codex will read source files, assess each suggestion, apply")
    print("    accepted edits, and write the summary. This may take a minute.\n")

    try:
        subprocess.run(
            [
                "codex",
                "--ask-for-approval",
                "never",
                "exec",
                "--sandbox",
                "workspace-write",
                instruction,
            ],
            text=True,
            encoding="utf-8",
            cwd=repo_root,
            check=True,
        )
    except FileNotFoundError:
        print('[ERROR] "codex" CLI not found.')
        _save_prompt_fallback(prompt, repo_root, agent="codex")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f'[ERROR] "codex" exited with status {exc.returncode}.')
        _save_prompt_fallback(prompt, repo_root, agent="codex")
        sys.exit(exc.returncode)

    # Prompt file is intermediate input to the agent; remove on success.
    # On failure (FileNotFoundError / CalledProcessError) the fallback path
    # above re-saves it for manual rerun, so we only delete here.
    prompt_path.unlink(missing_ok=True)


def _save_prompt_file(prompt: str, repo_root: Path) -> Path:
    fallback = repo_root / ".github_review_prompt.txt"
    fallback.write_text(prompt, encoding="utf-8")
    return fallback


def _save_prompt_fallback(prompt: str, repo_root: Path, *, agent: str) -> None:
    fallback = _save_prompt_file(prompt, repo_root)
    print(f"[*] Prompt saved to {fallback.name}")
    print("    Run manually with one of:")
    print(
        '      claude --print --allowedTools "Read,Write,Edit,Glob,Grep" '
        f"< {fallback.name}"
    )
    print(
        f"      codex --ask-for-approval never exec --sandbox workspace-write "
        f'"Read {fallback.name} and carry out the review '
        'workflow described there."'
    )
    print(f"    Requested agent was: {agent}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai_agent_github_reviews.py",
        description=(
            "Screen GitHub PR review comments with Claude Code or Codex and apply "
            "accepted suggestions as pending working-tree changes."
        ),
    )
    parser.add_argument(
        "--pr",
        type=int,
        required=True,
        metavar="NUMBER",
        help="Pull request number",
    )
    parser.add_argument(
        "--repo",
        metavar="OWNER/REPO",
        default=None,
        help=(
            "GitHub repo slug "
            "(default: inferred from git remote upstream, falling back to origin)"
        ),
    )
    parser.add_argument(
        "--agent",
        choices=("claude", "codex"),
        default="codex",
        help="AI agent to invoke for review processing (default: codex)",
    )
    parser.add_argument(
        "--prompt-only",
        "--dry-run",
        dest="prompt_only",
        action="store_true",
        help="Print the prompt that would be sent to the agent and exit without changes",
    )
    parser.add_argument(
        "--since-last-push",
        dest="since_last_push",
        action="store_true",
        help=(
            "Only include inline comments and reviews after the latest reflog time "
            "for refs/remotes/<remote>/<PR_head_branch> (this clone)"
        ),
    )
    parser.add_argument(
        "--remote",
        metavar="NAME",
        default="origin",
        help="Git remote name for fetch/reflog (default: origin; used with --since-last-push)",
    )
    resolve_group = parser.add_mutually_exclusive_group()
    resolve_group.add_argument(
        "--no-resolve",
        dest="no_resolve",
        action="store_true",
        help="Do not mark inline-comment threads as resolved after the agent processes them",
    )
    resolve_group.add_argument(
        "--mark-resolved",
        dest="mark_resolved",
        action="store_true",
        help=(
            "Mark threads as resolved even when --prompt-only is set "
            "(no agent invocation required)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    repo: str = args.repo or get_repo_slug(repo_root)
    pr_number: int = args.pr
    summary_filename = f"pr_{pr_number}_review_summary.md"

    print(f"[*] Repository : {repo}")
    print(f"[*] PR         : #{pr_number}")
    print(f"[*] Summary    : {summary_filename}")
    print()

    print("[*] Fetching PR metadata...")
    pr_data = fetch_pr_data(pr_number, repo)
    print(f'    "{pr_data.get("title", "")}"')

    print("[*] Fetching inline review threads (unresolved only)...")
    threads = fetch_review_threads(pr_number, repo)
    unresolved_threads = [t for t in threads if not t["isResolved"]]
    resolved_count = len(threads) - len(unresolved_threads)
    # Flatten to a comment list; each comment carries _thread_id for later resolution.
    thread_comments: list[dict] = [c for t in unresolved_threads for c in t["comments"]]
    thread_ids: list[str] = [t["id"] for t in unresolved_threads if t["comments"]]
    print(
        f"    {len(thread_comments)} comment(s) in {len(unresolved_threads)} "
        f"unresolved thread(s) ({resolved_count} already resolved, skipped)"
    )

    print("[*] Fetching PR-level reviews...")
    reviews = fetch_reviews(pr_number, repo)
    review_bodies = [
        r
        for r in reviews
        if r.get("body", "").strip() and r.get("state") not in ("PENDING", "")
    ]
    print(f"    {len(review_bodies)} review(s) with body text")

    if args.since_last_push:
        head_ref = pr_data.get("head", {}).get("ref")
        if not head_ref:
            print("[ERROR] PR has no head branch ref; cannot use --since-last-push.")
            sys.exit(1)
        if head_ref == "main":
            print(
                "[ERROR] PR head branch is 'main'; --since-last-push is not meaningful here."
            )
            sys.exit(1)
        git_fetch(repo_root, args.remote, head_ref)
        remote_ref = f"refs/remotes/{args.remote}/{head_ref}"
        cutoff = get_remote_reflog_cutoff(repo_root, args.remote, head_ref)
        print()
        print("[*] --since-last-push")
        print(f"    Remote ref : {remote_ref}")
        print(f"    Cutoff     : {cutoff.isoformat()}")
        thread_comments, reviews = filter_since_cutoff(thread_comments, reviews, cutoff)
        review_bodies = [
            r
            for r in reviews
            if r.get("body", "").strip() and r.get("state") not in ("PENDING", "")
        ]
        # Only resolve threads where every comment survived the cutoff filter.
        # A thread with pre-cutoff comments was not fully represented in the
        # prompt, so resolving it would be premature.
        surviving_thread_ids = {
            c["_thread_id"] for c in thread_comments if "_thread_id" in c
        }
        thread_ids = [
            t["id"]
            for t in unresolved_threads
            if t["id"] in surviving_thread_ids
            and t["comments"]
            and all(
                parse_github_datetime(c.get("created_at", "1970-01-01T00:00:00Z"))
                > cutoff
                for c in t["comments"]
            )
        ]
        print(
            f"    After filter: {len(thread_comments)} thread comment(s), "
            f"{len(review_bodies)} review(s) with body text, "
            f"{len(thread_ids)} thread(s) eligible for auto-resolve"
        )

    total = len(thread_comments) + len(review_bodies)
    if total == 0:
        print("\n[*] No unresolved review comments found. Nothing to do.")
        sys.exit(0)

    print(f"\n[*] Building prompt for {total} comment(s)...")
    prompt = build_prompt(
        pr_number=pr_number,
        pr_data=pr_data,
        reviews=reviews,
        thread_comments=thread_comments,
        summary_filename=summary_filename,
        agent=args.agent,
        repo_root=repo_root,
    )

    if args.prompt_only:
        separator = "=" * 60
        if args.since_last_push:
            print()
            print(
                "[prompt-only] --since-last-push: using cutoff and counts above "
                "(full prompt follows)."
            )
        print(f"\n{separator}")
        print(f"PROMPT (prompt-only — not sent to {args.agent})")
        print(separator)
        print(prompt)
        print(separator)
        print("\n[prompt-only] No files changed.")
        print(f"[prompt-only] Summary would be written to: {summary_filename}")
        if args.mark_resolved and thread_ids:
            resolve_review_threads(thread_ids, repo)
        sys.exit(0)

    invoke_ai_agent(prompt, repo_root, args.agent)

    if not args.no_resolve and thread_ids:
        resolve_review_threads(thread_ids, repo)
    elif args.no_resolve:
        print(f"[*] --no-resolve: skipped resolving {len(thread_ids)} thread(s).")

    print()
    summary_path = repo_root / summary_filename
    if summary_path.exists():
        print(f"[+] Summary written : {summary_filename}")
    else:
        print("[!] Summary file not found — check agent output above.")
    print("[*] Inspect changes : git diff")
    print("[*] Stage selectively: git add -p")


if __name__ == "__main__":
    main()
