"""Compare tuning-run JSONs with the Greedy baseline on the same phases.

Usage:
    py tuning_compare.py <dataset> <run.json> [<run.json> ...]

Prints seconds per pair, mean NCC and mean negative-Jacobian fraction of each
run next to the Greedy baseline restricted to the run's phases, and whether the
D8 constraints (NCC >= Greedy NCC - 0.002, neg-Jacobian <= Greedy's) hold.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
NCC_MARGIN = 0.002


def means(rows: list[dict]) -> tuple[float, float, float]:
    """Mean seconds, NCC and negative-Jacobian fraction of per-phase rows."""
    return (
        statistics.mean(r["seconds"] for r in rows),
        statistics.mean(r["ncc"] for r in rows),
        statistics.mean(r["neg_jacobian_fraction"] for r in rows),
    )


def main() -> None:
    """Print one comparison line per run JSON."""
    dataset = sys.argv[1]
    greedy = json.loads(
        (RESULTS / f"{dataset}_greedy_baseline.json").read_text(encoding="utf-8")
    )
    for path in sys.argv[2:]:
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        phases = [r["phase"] for r in run["per_phase"]]
        reference = [r for r in greedy["per_phase"] if r["phase"] in phases]
        seconds, ncc, neg_jac = means(run["per_phase"])
        g_seconds, g_ncc, g_neg_jac = means(reference)
        ok_ncc = ncc >= g_ncc - NCC_MARGIN
        ok_jac = neg_jac <= g_neg_jac
        print(
            f"{Path(path).parent.name}/{Path(path).name}: phases {phases} "
            f"s/pair {seconds:.2f} (greedy {g_seconds:.2f}) "
            f"NCC {ncc:.5f} (greedy {g_ncc:.5f}, ok={ok_ncc}) "
            f"negJ {neg_jac:.2e} (greedy {g_neg_jac:.2e}, ok={ok_jac})"
        )


if __name__ == "__main__":
    main()
