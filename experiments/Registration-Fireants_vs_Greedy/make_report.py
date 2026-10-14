"""Generate REPORT.md from the benchmark and tuning JSONs, deterministically.

Reads ``experiments/Registration-Fireants_vs_Greedy/results`` (greedy_baseline, fireants_default and
fireants_tuned JSONs for heart and lung, ``tuning/**/notes.json`` with the
candidate run JSONs, and ``affine_diagnostic.json``) and writes a Markdown
report. The output contains no timestamps or paths; the same inputs always give
byte-identical output.

Usage:
    py make_report.py [--out experiments/Registration-Fireants_vs_Greedy/results/REPORT.md]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
DEFAULT_OUT = RESULTS / "REPORT.md"

NCC_TOLERANCE = 0.002
METHODS = (
    ("greedy_baseline", "Greedy"),
    ("fireants_default", "FireANTs-default"),
    ("fireants_tuned", "FireANTs-tuned"),
)
DATASETS = (("heart", "Heart"), ("lung", "Lung"))

Runs = dict[str, dict[str, dict[str, Any]]]


def load(path: Path) -> Any:
    """Read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def mean(rows: list[dict[str, Any]], key: str) -> float:
    """Mean of ``key`` over the rows that have it."""
    return statistics.fmean(r[key] for r in rows if key in r)


def rows_of(run: dict[str, Any], phases: Optional[set[int]] = None) -> list[Any]:
    """Per-phase rows of a run, optionally restricted to ``phases``."""
    return [r for r in run["per_phase"] if phases is None or r["phase"] in phases]


def parse_phases(text: str) -> set[int]:
    """Parse '3,9,17' into a set."""
    return {int(p) for p in text.split(",")}


def fmt_set(phases: set[int]) -> str:
    """Sorted comma-separated phase list."""
    return ",".join(str(p) for p in sorted(phases))


def fmt_list(values: list[Any]) -> str:
    """Compact bracketed list."""
    return "[" + ",".join(str(v) for v in values) + "]"


def table(header: list[str], body: list[list[str]]) -> list[str]:
    """Markdown table lines."""
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return lines


def metrics(run: dict[str, Any], phases: Optional[set[int]] = None) -> dict[str, Any]:
    """Mean metrics of a run over the given phases (default all)."""
    rows = rows_of(run, phases)
    out: dict[str, Any] = {
        "n": len(rows),
        "total": sum(r["seconds"] for r in rows),
        "per_pair": mean(rows, "seconds"),
        "ncc": mean(rows, "ncc"),
        "mse": mean(rows, "mse"),
        "negj": 100.0 * mean(rows, "neg_jacobian_fraction"),
        "inv": mean(rows, "inverse_consistency_mm"),
        "ncc_unr": mean(rows, "ncc_unregistered"),
        "mse_unr": mean(rows, "mse_unregistered"),
    }
    if "dice" in rows[0]:
        out["dice"] = mean(rows, "dice")
        out["dice_unr"] = mean(rows, "dice_unregistered")
    return out


def tuning_dir(name: str) -> Path:
    """Tuning directory of a dataset."""
    return RESULTS / "tuning" / name


def tuning_rounds(name: str) -> list[dict[str, Any]]:
    """Notes of every round of a dataset, in numeric round order."""
    paths = sorted(
        tuning_dir(name).glob("round_*/notes.json"),
        key=lambda p: int(p.parent.name.split("_")[1]),
    )
    return [{**load(p), "dir": p.parent} for p in paths]


def tuning_phases_used(name: str) -> set[int]:
    """Union of the tuning phases of all rounds of a dataset."""
    used: set[int] = set()
    for r in tuning_rounds(name):
        used |= parse_phases(r["tuning_phases"])
    return used


def dataset_section(
    name: str, label: str, runs: dict[str, dict[str, Any]]
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Results table for one dataset and its mean metrics."""
    stats = {key: metrics(run) for key, run in runs.items()}
    greedy = stats["greedy_baseline"]
    lung = name == "lung"
    header = [
        "Method",
        "total s",
        "per-pair s (mean)",
        "speedup vs Greedy",
        "NCC",
        "MSE",
        "neg-Jacobian %",
        "inverse-consistency mm",
    ]
    if lung:
        header.append("Dice")
    body: list[list[str]] = []
    for key, method_label in METHODS:
        s = stats[key]
        row = [
            method_label,
            f"{s['total']:.1f}",
            f"{s['per_pair']:.2f}",
            f"{greedy['total'] / s['total']:.2f}x",
            f"{s['ncc']:.5f}",
            f"{s['mse']:.1f}",
            f"{s['negj']:.5f}",
            f"{s['inv']:.5f}",
        ]
        if lung:
            row.append(f"{s['dice']:.5f}")
        body.append(row)
    unr = ["Unregistered", "-", "-", "-", f"{greedy['ncc_unr']:.5f}"]
    unr += [f"{greedy['mse_unr']:.1f}", "-", "-"]
    if lung:
        unr.append(f"{greedy['dice_unr']:.5f}")
    body.append(unr)

    run = runs["greedy_baseline"]
    lines = [f"## {label}", ""]
    lines.append(
        f"{greedy['n']} pairs (every phase except the reference frame "
        f"{run['reference_frame']}, {run['reference_frame_name']}); "
        "metrics are means over pairs."
    )
    lines.append("")
    lines.extend(table(header, body))
    lines.append("")
    lines.append(
        "Times are the wall clock of `register()` per pair (preprocessing, "
        "registration, inverse transform; one warm-up pair excluded; metrics "
        "outside the clock). Repeat timings of phase "
        f"{run['repeat_phase']} (median of {len(run['repeat_seconds'])} "
        "repeats): "
        + ", ".join(
            f"{lab} {runs[key]['repeat_median_seconds']:.2f} s" for key, lab in METHODS
        )
        + "."
    )
    lines.append("")
    return lines, stats


def settings_section(runs: Runs) -> list[str]:
    """Greedy settings, FireANTs default and selected tuned configurations."""
    lines = ["## Configurations", ""]
    lines.append("### Greedy settings used (not tuned)")
    lines.append("")
    for name, label in DATASETS:
        c = runs[name]["greedy_baseline"]["config"]
        lines.append(
            f"- {label}: iterations {fmt_list(c['number_of_iterations'])}, "
            f"metric {c['metric']} ({c['greedy_metric']}), transform "
            f"{c['transform_type']}, deformable smoothing "
            f"{c['deformable_smoothing']} (the tutorial_03 registrar)."
        )
    lines.append("")
    keys = [
        "number_of_iterations",
        "scales",
        "init_affine",
        "workflow",
        "optimizer_lr",
        "smooth_warp_sigma",
        "smooth_grad_sigma",
        "cc_kernel_size",
        "cc_kernel_type",
        "deformation_type",
        "integrator_n",
        "blur",
    ]
    lines.append("### FireANTs-default and FireANTs-tuned (selected) configurations")
    lines.append("")
    header = ["Setting"] + [
        f"{label} {m}" for name, label in DATASETS for m in ("default", "tuned")
    ]
    body = []
    for key in keys:
        row = [key]
        for name, _ in DATASETS:
            for method in ("fireants_default", "fireants_tuned"):
                value = runs[name][method]["config"][key]
                row.append(fmt_list(value) if isinstance(value, list) else str(value))
        body.append(row)
    lines.extend(table(header, body))
    lines.append("")
    lines.append(
        "Common to all FireANTs runs: metric CC (loss cc), transform "
        "Deformable, Adam optimizer, tolerance "
        f"{runs['heart']['fireants_tuned']['config']['tolerance']}."
    )
    lines.append("")
    return lines


def tuning_section(runs: Runs) -> list[str]:
    """What was evaluated in the tuning rounds, and the lung margin decision."""
    lines = ["## Tuning record", ""]
    lines.append(
        "Objective (D8): minimize seconds per pair subject to NCC >= Greedy "
        f"NCC - {NCC_TOLERANCE} and neg-Jacobian <= Greedy's, both measured on "
        "the same tuning phases of that round. Greedy was never tuned. A round "
        "is a batch of at most 3 candidate configurations."
    )
    lines.append("")
    for name, label in DATASETS:
        rounds = tuning_rounds(name)
        greedy = runs[name]["greedy_baseline"]
        counts = [len(r["candidates"]) for r in rounds]
        files = sorted(p.name for p in tuning_dir(name).glob("round_*/r*.json"))
        lines.append(f"### {label}")
        lines.append("")
        lines.append(
            f"- Round folders: {len(rounds)}; configurations evaluated: "
            f"{sum(counts)} (run JSONs on disk: {len(files)}); largest batch: "
            f"{max(counts)}."
        )
        lines.append("")
        header = [
            "Round",
            "Run",
            "iterations",
            "scales",
            "init",
            "lr",
            "warp sigma",
            "kernel",
            "tuning phases",
            "s/pair",
            "NCC",
            "neg-Jac %",
            "NCC margin",
            "selected",
        ]
        body = []
        notes_lines: list[str] = []
        for r in rounds:
            phases = parse_phases(r["tuning_phases"])
            greedy_ncc = mean(rows_of(greedy, phases), "ncc")
            for cand in r["candidates"]:
                c = cand["config"]
                m = metrics(load(r["dir"] / cand["run_json"]))
                run_id = cand["run_json"].removesuffix(".json")
                body.append(
                    [
                        str(r["round"]),
                        run_id,
                        fmt_list(c["number_of_iterations"]),
                        fmt_list(c["scales"]),
                        c["init_affine"].split()[0],
                        str(c["optimizer_lr"]),
                        str(c["smooth_warp_sigma"]),
                        str(c["cc_kernel_size"]),
                        r["tuning_phases"],
                        f"{m['per_pair']:.2f}",
                        f"{m['ncc']:.5f}",
                        f"{m['negj']:.5f}",
                        f"{m['ncc'] - (greedy_ncc - NCC_TOLERANCE):+.5f}",
                        "yes" if "SELECTED" in cand["verdict"] else "no",
                    ]
                )
                notes_lines.append(f"- {run_id}: " + cand["verdict"].replace("|", "/"))
        lines.extend(table(header, body))
        lines.append("")
        lines.append(
            "NCC margin = candidate NCC minus (Greedy NCC on the same phases "
            f"- {NCC_TOLERANCE}); negative means the NCC constraint failed."
        )
        lines.append("")
        lines.append("Recorded rationale per configuration:")
        lines.append("")
        lines.extend(notes_lines)
        lines.append("")
    lines.extend(lung_margin_section(runs["lung"]["greedy_baseline"]))
    return lines


def lung_margin_section(greedy: dict[str, Any]) -> list[str]:
    """Margin criterion for lung r3_a versus r3_c."""
    round3 = next(r for r in tuning_rounds("lung") if r["round"] == 3)
    phases = parse_phases(round3["tuning_phases"])
    greedy_ncc = mean(rows_of(greedy, phases), "ncc")
    threshold = greedy_ncc - NCC_TOLERANCE
    cands = {
        c["run_json"].removesuffix(".json"): (c, load(round3["dir"] / c["run_json"]))
        for c in round3["candidates"]
    }
    a, c_ = cands["r3_a"], cands["r3_c"]
    ma, mc = metrics(a[1]), metrics(c_[1])
    lines = ["### Lung round 3: r3_a versus r3_c (margin criterion)", ""]
    lines.append(
        "Both candidates satisfy the D8 constraints on the round-3 tuning "
        f"phases ({round3['tuning_phases']}); the NCC threshold is "
        f"{threshold:.5f} (Greedy {greedy_ncc:.5f} - {NCC_TOLERANCE})."
    )
    lines.append("")
    for tag, cand, m in (("r3_a", a, ma), ("r3_c", c_, mc)):
        cfg = cand[0]["config"]
        lines.append(
            f"- {tag} (iterations {fmt_list(cfg['number_of_iterations'])}, "
            f"scales {fmt_list(cfg['scales'])}): {m['per_pair']:.2f} s/pair, "
            f"NCC {m['ncc']:.5f}, margin {m['ncc'] - threshold:+.5f}."
        )
    lines.append(
        f"- r3_c is {ma['per_pair'] - mc['per_pair']:+.2f} s/pair faster than "
        f"r3_a; its NCC differs from r3_a by {mc['ncc'] - ma['ncc']:+.5f}."
    )
    lines.append(f"- Recorded decision for r3_a: {a[0]['verdict']}")
    lines.append(f"- Recorded decision for r3_c: {c_[0]['verdict']}")
    lines.append("")
    lines.append(
        "A strict reading of D8 (minimize seconds per pair among candidates "
        "that pass the constraints) would have selected r3_c, the faster "
        "candidate. r3_a was selected instead by a judgement call recorded in "
        "the round notes, not a pre-registered numeric rule: the larger NCC "
        "margin was preferred over a speed gain of a fraction of a second per "
        "pair, which is within the pair-to-pair timing variation of repeated "
        "runs (about a second). No numeric margin threshold was defined."
    )
    lines.append("")
    return lines


def heldout_section(runs: Runs) -> list[str]:
    """Constraint check on phases never used for tuning."""
    lines = ["## Held-out-phase constraint check", ""]
    lines.append(
        "The tuning rounds used a few phases per dataset, and those phases are "
        "also among the reported phases above, so the all-phase numbers are "
        "not independent of tuning. The check below uses only phases that no "
        "round used for tuning, with the D8 constraints applied to the means "
        "over those phases."
    )
    lines.append("")
    for name, label in DATASETS:
        greedy = runs[name]["greedy_baseline"]
        tuned = runs[name]["fireants_tuned"]
        used = tuning_phases_used(name)
        held = {r["phase"] for r in greedy["per_phase"]} - used
        mg, mt = metrics(greedy, held), metrics(tuned, held)
        gmap = {r["phase"]: r for r in rows_of(greedy, held)}
        tmap = {r["phase"]: r for r in rows_of(tuned, held)}
        worst = min(tmap[p]["ncc"] - gmap[p]["ncc"] for p in held)
        ncc_ok = mt["ncc"] >= mg["ncc"] - NCC_TOLERANCE
        negj_ok = mt["negj"] <= mg["negj"]
        lines.append(f"### {label}")
        lines.append("")
        lines.append(
            f"- Phases used for tuning (excluded): {fmt_set(used)}; held-out "
            f"phases ({len(held)}): {fmt_set(held)}."
        )
        lines.append(
            f"- Mean NCC: tuned {mt['ncc']:.5f} vs Greedy {mg['ncc']:.5f} "
            f"(difference {mt['ncc'] - mg['ncc']:+.5f}; constraint >= "
            f"{-NCC_TOLERANCE:+.3f}): {'met' if ncc_ok else 'NOT met'}."
        )
        lines.append(
            f"- Mean neg-Jacobian: tuned {mt['negj']:.5f}% vs Greedy "
            f"{mg['negj']:.5f}% (constraint tuned <= Greedy): "
            f"{'met' if negj_ok else 'NOT met'}."
        )
        lines.append(
            f"- Worst single-phase NCC difference (tuned - Greedy): {worst:+.5f}."
        )
        lines.append("")
        body = [
            [
                str(p),
                f"{gmap[p]['ncc']:.5f}",
                f"{tmap[p]['ncc']:.5f}",
                f"{100 * gmap[p]['neg_jacobian_fraction']:.5f}",
                f"{100 * tmap[p]['neg_jacobian_fraction']:.5f}",
            ]
            for p in sorted(held)
        ]
        header = [
            "Phase",
            "Greedy NCC",
            "tuned NCC",
            "Greedy neg-Jac %",
            "tuned neg-Jac %",
        ]
        lines.extend(table(header, body))
        lines.append("")
    return lines


def affine_section(runs: Runs) -> list[str]:
    """Identity-initialised tuning and the default affine divergence."""
    lines = ["## Affine initialisation", ""]
    default_init = runs["heart"]["fireants_default"]["config"]["init_affine"]
    lines.append(
        f"- FireANTs-default starts the deformable stage from: {default_init}."
    )
    lines.append(
        "- FireANTs-tuned starts from: "
        f"{runs['heart']['fireants_tuned']['config']['init_affine']} (heart) and "
        f"{runs['lung']['fireants_tuned']['config']['init_affine']} (lung), i.e. "
        "it is deformable-only and does not use the FireANTs affine stage; the "
        "tuned speed and accuracy therefore do not include an affine alignment."
    )
    greedy = {r["phase"]: r for r in runs["heart"]["greedy_baseline"]["per_phase"]}
    default = {r["phase"]: r for r in runs["heart"]["fireants_default"]["per_phase"]}
    tuned = {r["phase"]: r for r in runs["heart"]["fireants_tuned"]["per_phase"]}
    lines.append("")
    lines.append("Heart phases 13 and 15 in the benchmark runs:")
    lines.append("")
    body = [
        [
            str(p),
            f"{default[p]['ncc_unregistered']:.5f}",
            f"{greedy[p]['ncc']:.5f}",
            f"{default[p]['ncc']:.5f}",
            f"{tuned[p]['ncc']:.5f}",
            f"{100 * default[p]['neg_jacobian_fraction']:.5f}",
        ]
        for p in (13, 15)
    ]
    header = [
        "Phase",
        "unregistered NCC",
        "Greedy NCC",
        "default NCC",
        "tuned NCC",
        "default neg-Jac %",
    ]
    lines.extend(table(header, body))
    lines.append("")
    path = RESULTS / "affine_diagnostic.json"
    if not path.exists():
        lines.append("Affine diagnostic (affine_diagnostic.json) not available.")
        lines.append("")
        return lines
    diag = load(path)
    lines.append(
        "Diagnostic (`affine_diagnostic.py`): Affine-only registration of "
        "heart phases to the reference frame, Greedy affine (tutorial "
        "registrar) versus FireANTs affine (class defaults, the stage "
        "FireANTs-default uses as initialisation)."
    )
    lines.append("")
    body = []
    for g, f in zip(diag["greedy"], diag["fireants"]):
        body.append(
            [
                str(g["phase"]),
                f"{g['ncc_unregistered']:.5f}",
                f"{g['ncc']:.5f}",
                f"{f['ncc']:.5f}",
                f"{g['seconds']:.1f}",
                f"{f['seconds']:.1f}",
                fmt_list([round(v, 1) for v in g["affine"]["translation_mm"]]),
                fmt_list([round(v, 1) for v in f["affine"]["translation_mm"]]),
                f"{g['affine']['determinant']:.4f}",
                f"{f['affine']['determinant']:.4f}",
            ]
        )
    header = [
        "Phase",
        "unregistered NCC",
        "Greedy affine NCC",
        "FireANTs affine NCC",
        "Greedy s",
        "FireANTs s",
        "Greedy translation mm",
        "FireANTs translation mm",
        "Greedy det",
        "FireANTs det",
    ]
    lines.extend(table(header, body))
    lines.append("")
    bad = [f for f in diag["fireants"] if f["ncc"] < f["ncc_unregistered"] - 0.01]
    bad_phases = ", ".join(str(f["phase"]) for f in bad) or "none"
    lines.append(
        "Finding: FireANTs affine with library defaults ends worse than no "
        f"registration at heart phases {bad_phases} (NCC below the "
        "unregistered value by more than 0.01, translations of tens of mm "
        "along one axis), while Greedy affine stays near identity on the "
        "same pairs. The deformable stage cannot fully recover from such an "
        "initialisation, which matches the low FireANTs-default NCC at those "
        "phases above. Identity initialisation avoids the affine stage "
        "altogether; the heart pairs are already close to aligned (the "
        "unregistered NCC is high). The diagnostic points to "
        "optimiser behaviour of the FireANTs affine stage at default "
        "settings rather than a defect in the transform conversion: the "
        "conversion is exercised by the known-affine tests of the class and "
        "gives sane transforms on the other phases. Tuning the affine stage "
        "was not attempted."
    )
    lines.append("")
    return lines


def discussion_section(stats: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    """Speed and accuracy trade-offs, and the recommendation."""
    lines = ["## Discussion", ""]
    lines.append("### Speed")
    lines.append("")
    for name, label in DATASETS:
        g = stats[name]["greedy_baseline"]
        d = stats[name]["fireants_default"]
        t = stats[name]["fireants_tuned"]
        lines.append(
            f"- {label}: FireANTs-default is {g['total'] / d['total']:.2f}x and "
            f"FireANTs-tuned {g['total'] / t['total']:.2f}x faster than Greedy "
            f"({g['per_pair']:.1f} s/pair for Greedy, {d['per_pair']:.1f} s for "
            f"default, {t['per_pair']:.1f} s for tuned). The tuned gain over "
            f"the default is {d['total'] / t['total']:.2f}x; the remaining time "
            "is largely preprocessing and inverse-transform computation inside "
            "`register()`, which the iteration counts do not change."
        )
    lines.append("")
    lines.append("### Accuracy")
    lines.append("")
    for name, label in DATASETS:
        g = stats[name]["greedy_baseline"]
        d = stats[name]["fireants_default"]
        t = stats[name]["fireants_tuned"]
        text = (
            f"- {label}: mean NCC Greedy {g['ncc']:.5f}, default {d['ncc']:.5f}, "
            f"tuned {t['ncc']:.5f} (unregistered {g['ncc_unr']:.5f}); mean MSE "
            f"Greedy {g['mse']:.1f}, default {d['mse']:.1f}, tuned "
            f"{t['mse']:.1f}. Tuned MSE is "
            f"{100 * (t['mse'] / g['mse'] - 1):+.1f}% relative to Greedy, so "
            "the NCC-based constraint can hide a larger intensity-difference "
            f"gap. Neg-Jacobian: Greedy {g['negj']:.5f}%, default "
            f"{d['negj']:.5f}%, tuned {t['negj']:.5f}%. Inverse consistency: "
            f"Greedy {g['inv']:.5f} mm, default {d['inv']:.5f} mm, tuned "
            f"{t['inv']:.5f} mm"
        )
        if "dice" in g:
            text += (
                f"; Dice Greedy {g['dice']:.5f}, default {d['dice']:.5f}, "
                f"tuned {t['dice']:.5f} (unregistered {g['dice_unr']:.5f})"
            )
        lines.append(text + ".")
    lines.append("")
    lines.append(
        "In short: FireANTs-tuned meets the NCC and neg-Jacobian constraints "
        "it was tuned for, but it is not uniformly as accurate as Greedy; "
        "MSE (and Dice for lung) are worse than Greedy's. The NCC tolerance of "
        f"{NCC_TOLERANCE} was set by the task, not derived from a downstream "
        "requirement. FireANTs-default is the least accurate configuration "
        "on heart and has the highest neg-Jacobian fractions on both datasets."
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        "This is a recommendation only; changing the default registrar is out "
        "of scope for this work."
    )
    lines.append("")
    lines.append(
        "- Keep Greedy as the default registrar. Offer RegisterImagesFireants "
        "as an opt-in registrar for GPU users where a 3x (heart) to 5x (lung) "
        "reduction in registration time outweighs a small loss of accuracy "
        "relative to Greedy."
    )
    lines.append(
        "- When FireANTs is used, use the tuned, Identity-initialised "
        "configurations above rather than the class defaults, which diverge "
        "in the affine stage on some heart phases and give higher "
        "neg-Jacobian fractions."
    )
    lines.append(
        "- Before any default change, repeat the comparison on more than one "
        "patient per dataset and on a downstream task, since the limitations "
        "below prevent generalising from these runs."
    )
    lines.append("")
    return lines


def limitations_section(runs: Runs) -> list[str]:
    """Limitations, with machine facts read from the run JSONs."""
    g = runs["heart"]["greedy_baseline"]
    t = runs["heart"]["fireants_tuned"]
    lines = ["## Limitations", ""]
    lines.append(
        "- Single patient per dataset (one 4D heart CT, one 4D lung CT): no "
        "between-subject variability; differences between methods are not "
        "statistically tested."
    )
    lines.append(
        f"- Different hardware: Greedy ran on the CPU ({g['hardware']['cpu']}, "
        f"{g['hardware']['cpu_logical_cores']} logical cores); FireANTs ran on "
        f"the GPU ({t['hardware']['gpu']}, {t['hardware']['gpu_memory_gb']} GB). "
        "The speedups compare a CPU implementation with a GPU one, not "
        "algorithms on equal hardware."
    )
    lines.append(
        "- Windows pure-PyTorch FireANTs path: fused_ops "
        f"{str(t['versions']['fused_ops']).lower()} (fireants_fused_ops not "
        "installed, standard downsampling used) and fcntl_stub "
        f"{str(t['versions']['fcntl_stub']).lower()} (an fcntl stub is used on "
        f"Windows; Greedy run: {str(g['versions']['fcntl_stub']).lower()}). "
        "The fused kernels may change FireANTs timing and numerics on other "
        "platforms."
    )
    for name, label in DATASETS:
        used = tuning_phases_used(name)
        n = runs[name]["greedy_baseline"]["n_phases"] - 1
        lines.append(
            f"- {label}: the tuning phases ({fmt_set(used)}) are "
            f"{len(used)} of the {n} reported phases; all-phase results are "
            "not independent of tuning (see the held-out check)."
        )
    lines.append(
        "- The inverse-consistency metric (mean |inverse(forward(p)) - p| on "
        "a coarse interior lattice) is low-information: every method gives "
        "values of order 0.01 mm, and it measures the invertibility of the "
        "composed transform, not registration accuracy."
    )
    lines.append(
        "- Dice (lung) uses a threshold-based lung mask (-320 HU) that drops "
        "connected components touching the in-plane (x or y) image border "
        "and keeps the two largest remaining; z faces are not used, so lung "
        "cut by the scan range is kept. It is a proxy, not an expert "
        "segmentation."
    )
    lines.append(
        "- Timing scope: only `register()` per pair is timed; image loading, "
        "metric computation, warping and the 4D workflow around it are not. "
        "The timings include FireANTs image conversion and the displacement "
        "field inversion."
    )
    lines.append(
        "- NCC, MSE, neg-Jacobian and inverse consistency are proxy metrics "
        "against the fixed image; no landmark or target-registration-error "
        "data were used."
    )
    lines.append("")
    return lines


def build_report() -> str:
    """Assemble the full report text."""
    runs: Runs = {
        name: {key: load(RESULTS / f"{name}_{key}.json") for key, _ in METHODS}
        for name, _ in DATASETS
    }
    lines = ["# FireANTs vs Greedy registration report", ""]
    lines.append(
        "Generated by `experiments/Registration-Fireants_vs_Greedy/"
        "make_report.py` from the JSON files in `experiments/Registration-Fireants_vs_Greedy/results`. "
        "Methods: Greedy (the tutorial registrar), FireANTs-default (class "
        "defaults) and FireANTs-tuned (selected by the tuning rounds below)."
    )
    lines.append("")
    stats: dict[str, dict[str, dict[str, Any]]] = {}
    lines.append("# Results")
    lines.append("")
    for name, label in DATASETS:
        part, stats[name] = dataset_section(name, label, runs[name])
        lines.extend(part)
    lines.extend(settings_section(runs))
    lines.extend(tuning_section(runs))
    lines.extend(heldout_section(runs))
    lines.extend(affine_section(runs))
    lines.extend(discussion_section(stats))
    lines.extend(limitations_section(runs))
    return "\n".join(lines).rstrip("\n") + "\n"


def main() -> None:
    """Write the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(build_report())


if __name__ == "__main__":
    main()
