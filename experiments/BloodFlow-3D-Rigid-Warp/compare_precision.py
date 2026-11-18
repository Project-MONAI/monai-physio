"""Compare float64 and float32 runs of the 3D flow with each other and a reference.

All folders are required. A missing or empty float64 or float32 folder is
first computed by ``run.py`` (which refuses a folder with partial results),
and the GPU utilization just before that run is
recorded because the timings depend on other GPU work. Speed, GMRES behaviour
and accuracy go to ``OUTPUT.json`` and a Markdown table ``OUTPUT.md``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent


def gpu_utilization() -> str:
    """Return the current GPU utilization reported by nvidia-smi."""
    query = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"]
    result = subprocess.run(query, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def performance(folder: Path) -> dict[str, Any]:
    """Step time, setup time, GMRES iterations and stability of one run."""
    report = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    steps = np.load(folder / "steps.npz")
    velocity = np.load(folder / "velocity_m_s.npy", mmap_mode="r")
    return {
        "dt_s": report["dt_s"],
        "mean_step_s": float(steps["seconds"].mean()),
        "median_step_s": float(np.median(steps["seconds"])),
        "setup_s": report["setup_seconds"],
        "gmres_iterations_mean": float(steps["iterations"].mean()),
        "gmres_iterations_max": int(steps["iterations"].max()),
        "unconverged_steps": report["last_beat"]["unconverged_steps"],
        "nonfinite_frames": int(np.sum(~np.isfinite(velocity).all(axis=(1, 2)))),
        "warp_mempool_peak_mb": report["warp_mempool_peak_mb"],
    }


def accuracy(folder: Path, base: Path) -> dict[str, Any]:
    """Field, pressure and outlet-flow differences of ``folder`` from ``base``."""
    times, base_times = (
        np.load(f / "time.npz")["frame_times_s"] for f in (folder, base)
    )
    if not np.allclose(times, base_times):
        raise ValueError(f"{folder} and {base} save different frames")
    mesh, base_mesh = (np.load(f / "mesh.npz") for f in (folder, base))
    if any(not np.array_equal(mesh[key], base_mesh[key]) for key in mesh.files):
        raise ValueError(f"{folder} and {base} use different meshes")
    steps, base_steps = (np.load(f / "steps.npz") for f in (folder, base))
    if list(steps["port_names"]) != list(base_steps["port_names"]):
        raise ValueError(f"{folder} and {base} order their ports differently")
    fields = {}
    for name in ("velocity_m_s", "pressure_pa"):
        value, reference = (np.load(f / f"{name}.npy") for f in (folder, base))
        fields[name] = float(
            np.linalg.norm(value - reference) / np.linalg.norm(reference)
        )
    beat, base_beat = (
        json.loads((f / "run.json").read_text(encoding="utf-8"))["last_beat"]
        for f in (folder, base)
    )
    flows = steps["port_flow_m3_s"]
    base_flows = np.stack(
        [
            np.interp(steps["time_s"], base_steps["time_s"], column)
            for column in base_steps["port_flow_m3_s"].T
        ],
        axis=1,
    )
    return {
        "velocity_relative_l2": fields["velocity_m_s"],
        "pressure_relative_l2": fields["pressure_pa"],
        "port_flow_relative_l2": float(
            np.linalg.norm(flows - base_flows) / np.linalg.norm(base_flows)
        ),
        **{
            f"root_{kind}_difference_mmhg": beat[f"root_{kind}_mmhg"]
            - base_beat[f"root_{kind}_mmhg"]
            for kind in ("systolic", "diastolic", "mean")
        },
        "largest_flow_share_difference_points": max(
            abs(100 * (outlet["flow_share"] - base_beat["outlets"][name]["flow_share"]))
            for name, outlet in beat["outlets"].items()
        ),
        "peak_speed_beyond_10mm_difference_m_s": beat[
            "peak_speed_beyond_10mm_of_outlets_m_s"
        ]
        - base_beat["peak_speed_beyond_10mm_of_outlets_m_s"],
    }


def table(result: dict[str, Any]) -> str:
    """Markdown summary of the comparison."""
    rows = {
        "time step (s)": "{dt_s}",
        "s per step, mean / median": "{mean_step_s:.3f} / {median_step_s:.3f}",
        "setup s": "{setup_s:.1f}",
        "GMRES iterations, mean / max": "{gmres_iterations_mean:.1f} / "
        "{gmres_iterations_max}",
        "unconverged steps / non-finite frames": "{unconverged_steps} / "
        "{nonfinite_frames}",
        "Warp mempool peak MB": "{warp_mempool_peak_mb:.0f}",
        "GPU utilization before run": "{gpu_before_run}",
    }
    runs = [result["runs"][precision] for precision in ("float64", "float32")]
    lines = ["| | float64 | float32 |", "|---|---|---|"]
    for label, form in rows.items():
        cells = [form.format(**run) for run in runs]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    pairs = result["differences"]
    lines += ["", "| difference | " + " | ".join(pairs) + " |"]
    lines.append("|---" * (len(pairs) + 1) + "|")
    for key in pairs["float32 vs float64"]:
        cells = [f"{pair[key]:.3g}" for pair in pairs.values()]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the missing cases, compare them and write JSON and Markdown."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("float64", "float32", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    folders = {"float64": args.float64, "float32": args.float32}
    utilization = {}
    for precision, folder in folders.items():
        if (folder / "run.json").exists():
            utilization[precision] = "not measured (existing run)"
            continue
        utilization[precision] = gpu_utilization()
        command = [sys.executable, str(ROOT / "run.py"), "--precision", precision]
        subprocess.run([*command, "--output", str(folder)], check=True)
    result = {
        "folders": {key: str(value) for key, value in folders.items()},
        "reference": str(args.reference),
        "runs": {
            precision: {**performance(folder), "gpu_before_run": utilization[precision]}
            for precision, folder in folders.items()
        },
        "differences": {
            "float32 vs float64": accuracy(args.float32, args.float64),
            **{
                f"{precision} vs reference": accuracy(folder, args.reference)
                for precision, folder in folders.items()
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = args.output.with_suffix(".json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    markdown = table(result)
    output.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
