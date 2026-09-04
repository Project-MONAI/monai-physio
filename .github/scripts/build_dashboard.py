"""
Build the nightly health dashboard for MONAI Physio.

Reads JUnit XML test results and coverage JSON from --results-dir,
generates an HTML dashboard (index.html), a shields.io-compatible
endpoint (status.json), and a GitHub Actions step-summary (summary.md)
in --output-dir.

Usage (called from nightly-health.yml):
    python .github/scripts/build_dashboard.py \\
        --results-dir results/ \\
        --output-dir dashboard/ \\
        --run-url "https://github.com/Project-MONAI/monai-physio/actions/runs/123" \\
        --timestamp "2026-03-31T07:05:42Z" \\
        --health-outcome "success"

Status publishing:
    ``nightly-health.yml`` uploads the whole output directory as a
    ``health-dashboard`` artifact, then force-pushes ``status.json`` alone to
    the orphan ``nightly-status`` branch.  ``docs.yml`` fetches it from that
    branch through the GitHub contents API during its ``deploy`` job and copies
    it into the Pages output, so the file is served at the live URL:

        https://<pages-root>/status.json

    The fetch tolerates a missing branch, so the first docs deploy (before any
    nightly run has pushed ``status.json``) succeeds without it.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────────────────────────────────────


def parse_junit(xml_path: Path) -> dict:
    """Return a test-count summary from a JUnit XML file."""
    if not xml_path.exists():
        return {
            "tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "available": False,
        }

    try:
        tree = ET.parse(xml_path)  # noqa: S314
        root = tree.getroot()

        suites = root.findall("testsuite") if root.tag == "testsuites" else [root]

        tests = sum(int(s.get("tests", 0)) for s in suites)
        failures = sum(int(s.get("failures", 0)) for s in suites)
        errors = sum(int(s.get("errors", 0)) for s in suites)
        skipped = sum(int(s.get("skipped", 0)) for s in suites)
        passed = max(0, tests - failures - errors - skipped)

        return {
            "tests": tests,
            "passed": passed,
            "failed": failures,
            "errors": errors,
            "skipped": skipped,
            "available": True,
        }
    except Exception as exc:
        return {
            "tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "available": False,
            "parse_error": str(exc),
        }


def parse_junit_dir(results_dir: Path) -> dict:
    """Merge every ``test-results*.xml`` in ``results_dir`` into one summary.

    The nightly suite runs as more than one pytest invocation, so that an
    overrun in one cannot take the other's results with it.  Each writes its
    own JUnit XML and the dashboard reports their sum.
    """
    reports = [
        parse_junit(path) for path in sorted(results_dir.glob("test-results*.xml"))
    ]
    available = [report for report in reports if report["available"]]
    if not available:
        return parse_junit(results_dir / "test-results.xml")

    merged = {
        key: sum(report[key] for report in available)
        for key in ("tests", "passed", "failed", "errors", "skipped")
    }
    merged["available"] = True
    return merged


def parse_coverage(json_path: Path) -> dict:
    """Return coverage percentage from a coverage.py JSON report."""
    if not json_path.exists():
        return {"percent": None, "available": False}

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        totals = data.get("totals", {})
        pct = totals.get("percent_covered_display") or totals.get("percent_covered")
        if pct is not None:
            return {"percent": float(pct), "available": True}
        return {"percent": None, "available": False}
    except Exception as exc:
        return {"percent": None, "available": False, "parse_error": str(exc)}


# ──────────────────────────────────────────────────────────────────────────────
# HTML dashboard
# ──────────────────────────────────────────────────────────────────────────────

_CSS = """\
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:#0d1117;color:#c9d1d9;min-height:100vh;padding:32px 24px}
h1{font-size:1.6rem;font-weight:600;margin-bottom:6px}
.sub{color:#8b949e;font-size:.9rem;margin-bottom:36px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
      gap:16px;margin-bottom:36px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px}
.card .label{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;
             color:#8b949e;margin-bottom:10px}
.card .value{font-size:2rem;font-weight:700}
.pass{color:#3fb950}.fail{color:#f85149}.warn{color:#d29922}.muted{color:#8b949e}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;
       font-size:.78rem;font-weight:600}
.badge-pass{background:#1a3a2a;color:#3fb950;border:1px solid #2d5a3d}
.badge-fail{background:#3a1a1a;color:#f85149;border:1px solid #5a2d2d}
.badge-unknown{background:#2a2a1a;color:#d29922;border:1px solid #4a4a2a}
.bar-bg{background:#21262d;border-radius:4px;height:8px;overflow:hidden;margin-top:10px}
.bar-fill{height:100%;border-radius:4px}
a{color:#58a6ff;text-decoration:none}a:hover{text-decoration:underline}
footer{margin-top:48px;padding-top:16px;border-top:1px solid #21262d;
       color:#8b949e;font-size:.8rem}
"""


def _cov_color(pct: float) -> str:
    if pct >= 80:
        return "#3fb950"
    if pct >= 60:
        return "#d29922"
    return "#f85149"


def build_html(data: dict) -> str:
    junit = data["junit"]
    cov = data["coverage"]
    outcome = data["health_outcome"]
    run_url = data["run_url"]
    ts = data["timestamp"]

    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        ts_display = dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        ts_display = ts

    # Determine overall badge
    if not junit.get("available"):
        badge_class, badge_label = "badge-unknown", "UNKNOWN"
    elif (
        outcome not in ("success", "skipped", "")
        or (junit["failed"] + junit["errors"]) > 0
    ):
        badge_class, badge_label = "badge-fail", "FAILING"
    else:
        badge_class, badge_label = "badge-pass", "PASSING"

    # Test card values
    if junit.get("available"):
        n_pass = junit["passed"]
        n_fail = junit["failed"]
        n_err = junit["errors"]
        n_skip = junit["skipped"]
        n_total = junit["tests"]
        pass_cls = "pass" if (n_fail + n_err) == 0 else "fail"
        fail_cls = "fail" if (n_fail + n_err) > 0 else "muted"
        extra_err = (
            f'<div class="label" style="margin-top:6px">+ {n_err} error(s)</div>'
            if n_err > 0
            else ""
        )
    else:
        n_pass = n_fail = n_err = n_skip = n_total = "—"
        pass_cls = fail_cls = "muted"
        extra_err = ""

    # Coverage card
    if cov.get("available") and cov["percent"] is not None:
        pct = cov["percent"]
        cov_display = f"{pct:.1f}%"
        cov_col = _cov_color(pct)
        cov_bar = (
            f'<div class="bar-bg">'
            f'<div class="bar-fill" style="width:{pct:.1f}%;background:{cov_col}"></div>'
            f"</div>"
        )
    else:
        cov_display = "—"
        cov_col = "#8b949e"
        cov_bar = ""

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MONAI Physio — Nightly Health</title>
<style>{_CSS}</style>
</head>
<body>
<h1>MONAI Physio &mdash; Nightly Health</h1>
<p class="sub">
  Last run: {ts_display}&nbsp;&middot;&nbsp;
  <span class="badge {badge_class}">{badge_label}</span>&nbsp;&middot;&nbsp;
  <a href="{run_url}" target="_blank" rel="noopener">View run &rarr;</a>
</p>

<div class="grid">
  <div class="card">
    <div class="label">Tests Passed</div>
    <div class="value {pass_cls}">{n_pass}</div>
    <div class="label" style="margin-top:6px">of {n_total} total</div>
  </div>
  <div class="card">
    <div class="label">Tests Failed</div>
    <div class="value {fail_cls}">{n_fail}</div>
    {extra_err}
  </div>
  <div class="card">
    <div class="label">Tests Skipped</div>
    <div class="value muted">{n_skip}</div>
  </div>
  <div class="card">
    <div class="label">Coverage</div>
    <div class="value" style="color:{cov_col}">{cov_display}</div>
    {cov_bar}
  </div>
</div>

<footer>
  Generated by the
  <a href="https://github.com/Project-MONAI/monai-physio/actions/workflows/nightly-health.yml"
     target="_blank" rel="noopener">nightly-health</a> workflow
  &middot;
  <a href="https://github.com/Project-MONAI/monai-physio"
     target="_blank" rel="noopener">monai_physio</a>
</footer>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────────────
# shields.io endpoint JSON
# ──────────────────────────────────────────────────────────────────────────────


def build_status_json(data: dict) -> dict:
    """Return a shields.io dynamic endpoint object.

    Usage in README:
        ![Health](https://img.shields.io/endpoint?url=https://project-monai.github.io/monai-physio/status.json)
    """
    junit = data["junit"]

    if not junit.get("available"):
        # JUnit XML missing or unparseable — step was cancelled or never ran.
        color, message = "critical", "unknown"
    elif (junit["failed"] + junit["errors"]) > 0:
        n_bad = junit["failed"] + junit["errors"]
        color, message = "critical", f"{n_bad} failed"
    else:
        color, message = "brightgreen", f"{junit['passed']} passed"

    return {
        "schemaVersion": 1,
        "label": "nightly health",
        "message": message,
        "color": color,
    }


# ──────────────────────────────────────────────────────────────────────────────
# GitHub Actions Markdown summary
# ──────────────────────────────────────────────────────────────────────────────


def build_summary_md(data: dict) -> str:
    junit = data["junit"]
    cov = data["coverage"]
    outcome = data["health_outcome"]
    run_url = data["run_url"]
    ts = data["timestamp"]

    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        ts_display = dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        ts_display = ts

    if not junit.get("available"):
        status_icon = ":warning:"
        status_text = "Unknown (no results)"
    elif (
        outcome not in ("success", "skipped", "")
        or (junit["failed"] + junit["errors"]) > 0
    ):
        status_icon = ":x:"
        status_text = "Failing"
    else:
        status_icon = ":white_check_mark:"
        status_text = "Passing"

    lines = [
        f"# {status_icon} MONAI Physio Nightly Health — {ts_display}",
        "",
        f"**Status:** {status_text} &nbsp; | &nbsp; [View run]({run_url})",
        "",
        "## Test Results",
        "",
    ]

    if junit.get("available"):
        lines += [
            "| | Count |",
            "|---|---|",
            f"| :white_check_mark: Passed | {junit['passed']} |",
            f"| :x: Failed | {junit['failed']} |",
            f"| :warning: Errors | {junit['errors']} |",
            f"| :fast_forward: Skipped | {junit['skipped']} |",
            f"| **Total** | **{junit['tests']}** |",
            "",
        ]
    else:
        lines += ["_Test results not available._", ""]

    if cov.get("available") and cov["percent"] is not None:
        pct = cov["percent"]
        lines += [
            "## Coverage",
            "",
            f"**{pct:.1f}%** line coverage",
            "",
        ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build MONAI Physio nightly health dashboard"
    )
    parser.add_argument(
        "--results-dir",
        default="results/",
        help="Directory containing the test-results*.xml files and coverage.json",
    )
    parser.add_argument(
        "--output-dir",
        default="dashboard/",
        help="Output directory for dashboard files",
    )
    parser.add_argument("--run-url", default="#", help="URL of the GitHub Actions run")
    parser.add_argument(
        "--timestamp", default="", help="ISO 8601 UTC timestamp of the run"
    )
    parser.add_argument(
        "--health-outcome",
        default="",
        help="Step outcome of the health tests (success/failure/...)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = args.timestamp or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    data = {
        "junit": parse_junit_dir(results_dir),
        "coverage": parse_coverage(results_dir / "coverage.json"),
        "run_url": args.run_url,
        "timestamp": timestamp,
        "health_outcome": args.health_outcome,
    }

    (output_dir / "index.html").write_text(build_html(data), encoding="utf-8")
    (output_dir / "status.json").write_text(
        json.dumps(build_status_json(data), indent=2), encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(build_summary_md(data), encoding="utf-8")
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")

    print(f"Dashboard written to {output_dir}/")
    for name in ("index.html", "status.json", "summary.md", ".nojekyll"):
        print(f"  {name}")


if __name__ == "__main__":
    main()
