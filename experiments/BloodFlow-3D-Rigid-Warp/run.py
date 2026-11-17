"""Simulate rigid-wall 3D aortic blood flow with Warp FEM and RCR outlets.

The smoothed LV-derived flow enters through a blunt root profile. Every
outlet carries a three-element Windkessel whose share of the whole-body
resistance and compliance follows the configured flow fractions. The run
starts from rest at the initial outlet pressure and saves full fields every
``frame_interval_s``; ``run.json`` summarizes the last beat. ``precision``
selects float64 or float32 operators and linear solves.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp
from scipy.spatial import cKDTree

from monai_physio.warp_navier_stokes import WarpNavierStokes, Windkessel, inlet_flow

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
MMHG = 133.322


def summarize(
    names: list[str],
    steps: dict[str, np.ndarray],
    speeds: np.ndarray,
    near_outlet: np.ndarray,
    tolerance: float,
    start: float,
) -> dict[str, Any]:
    """Pressures, flow split and speeds over the last beat, from ``start``.

    ``speeds`` holds the last-beat nodal speeds (frames x nodes) and
    ``near_outlet`` marks nodes within 10 mm of an outlet centre.
    """
    times = steps["time_s"]
    last = times >= start
    previous = (times >= start - 1) & (times < start)
    inflow = -steps["port_flow_m3_s"][:, names.index("root_inlet")]
    root = steps["root_pressure_pa"] / MMHG
    outlets = {}
    for column, name in enumerate(n for n in names if n != "root_inlet"):
        flow = steps["port_flow_m3_s"][last, names.index(name)]
        pressure = steps["outlet_pressure_pa"][last, column] / MMHG
        outlets[name] = {
            "mean_flow_ml_s": float(flow.mean() * 1e6),
            "flow_share": float(flow.mean() / inflow[last].mean()),
            "systolic_mmhg": float(pressure.max()),
            "diastolic_mmhg": float(pressure.min()),
        }
    net = np.abs(steps["port_flow_m3_s"].sum(axis=1)).max() / inflow.max()
    return {
        "root_systolic_mmhg": float(root[last].max()),
        "root_diastolic_mmhg": float(root[last].min()),
        "root_mean_mmhg": float(root[last].mean()),
        "root_mean_change_from_previous_beat_mmhg": float(
            root[last].mean() - root[previous].mean()
        ),
        "inlet_mean_flow_ml_s": float(inflow[last].mean() * 1e6),
        "outlets": outlets,
        "peak_speed_m_s": float(speeds.max()),
        "peak_speed_beyond_10mm_of_outlets_m_s": float(speeds[:, ~near_outlet].max()),
        "speed_99_9_percentile_m_s": float(np.percentile(speeds, 99.9)),
        "max_net_port_flow_relative": float(net),
        "max_linear_residual": float(steps["residual"].max()),
        "mean_gmres_iterations": float(steps["iterations"].mean()),
        "unconverged_steps": int(np.sum(~(steps["residual"] <= tolerance))),
    }


def run(
    preparation: Path,
    output: Path,
    config: dict[str, Any],
    dt: float,
    precision: str,
) -> dict[str, Any]:
    """Solve the configured duration and write fields, port series and summary."""
    geometry = np.load(preparation / "geometry.npz")
    ports = json.loads((preparation / "ports.json").read_text(encoding="utf-8"))
    flux = np.load(REPOSITORY / config["waveform"])["inlet_interval_flux_m3_s"]
    windkessel = config["windkessel"]
    outlets = {
        port["region_id"]: Windkessel.from_total(
            windkessel["flow_fractions"][name],
            windkessel["total_resistance_pa_s_m3"],
            windkessel["total_compliance_m3_pa"],
            windkessel["proximal_fraction"],
            windkessel["initial_pressure_pa"],
        )
        for name, port in ports.items()
        if port["role"] == "outlet"
    }
    names = sorted(ports, key=lambda n: ports[n]["region_id"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    solver = WarpNavierStokes(
        geometry["points_m"],
        geometry["tetrahedra"],
        geometry["boundary_faces"],
        geometry["boundary_regions"],
        inlet_region=ports["root_inlet"]["region_id"],
        outlets=outlets,
        dt=dt,
        density=config["density_kg_m3"],
        viscosity=config["viscosity_pa_s"],
        backflow=config["backflow_beta"],
        blunt_fraction=config["inlet_blunt_fraction"],
        precision=precision,
    )
    setup_seconds = time.perf_counter() - started
    faces = geometry["boundary_faces"]
    regions = geometry["boundary_regions"]
    root = regions == ports["root_inlet"]["region_id"]
    root_area = np.linalg.norm(solver.area_vectors[root], axis=1)
    root_weights = np.zeros(len(solver.points))
    np.add.at(root_weights, faces[root], root_area[:, None] / 3)
    root_weights /= root_weights.sum()

    count = round(config["duration_s"] / dt)
    stride = round(config["frame_interval_s"] / dt)
    times = np.arange(1, count + 1) * dt
    inflow = inlet_flow(
        times, flux, config["waveform_interval_s"], config["waveform_smoothing_s"]
    )[0]
    frames = count // stride
    velocity = np.lib.format.open_memmap(
        output / "velocity_m_s.npy",
        mode="w+",
        dtype=np.float64,
        shape=(frames, len(solver.dof_points), 3),
    )
    pressure = np.lib.format.open_memmap(
        output / "pressure_pa.npy",
        mode="w+",
        dtype=np.float64,
        shape=(frames, len(solver.points)),
    )
    steps: dict[str, np.ndarray] = {
        "time_s": times,
        "inlet_flow_m3_s": inflow,
        "port_flow_m3_s": np.zeros((count, len(names))),
        "outlet_pressure_pa": np.zeros((count, len(outlets))),
        "root_pressure_pa": np.zeros(count),
        "iterations": np.zeros(count, dtype=np.int64),
        "residual": np.zeros(count),
        "seconds": np.zeros(count),
    }
    stepping = time.perf_counter()
    for index, flow in enumerate(inflow):
        tic = time.perf_counter()
        result = solver.step(float(flow))
        if not np.isfinite(result["residual"]):
            raise RuntimeError(f"Linear solve diverged at step {index}: {result}")
        steps["seconds"][index] = time.perf_counter() - tic
        steps["iterations"][index] = result["iterations"]
        steps["residual"][index] = result["residual"]
        steps["port_flow_m3_s"][index] = [
            result["flows_m3_s"][ports[n]["region_id"]] for n in names
        ]
        steps["outlet_pressure_pa"][index] = [
            result["pressures_pa"][ports[n]["region_id"]]
            for n in names
            if n != "root_inlet"
        ]
        current_pressure = solver.pressure()
        steps["root_pressure_pa"][index] = root_weights @ current_pressure
        if (index + 1) % stride == 0:
            frame = (index + 1) // stride - 1
            velocity[frame] = solver.velocity()
            pressure[frame] = current_pressure
        if index % 100 == 0:
            print(
                f"t={times[index]:.4f} s Q={flow * 1e6:.1f} mL/s "
                f"root={steps['root_pressure_pa'][index] / MMHG:.1f} mmHg "
                f"GMRES={result['iterations']} {steps['seconds'][index]:.2f} s",
                flush=True,
            )
    stepping_seconds = time.perf_counter() - stepping
    solver.close()
    velocity.flush()
    pressure.flush()
    frame_times = times[stride - 1 :: stride]
    start = config["report_from_s"]
    speeds = np.linalg.norm(velocity[frame_times >= start], axis=2)
    centers = [p["center_m"] for p in ports.values() if p["role"] == "outlet"]
    near_outlet = np.asarray(cKDTree(centers).query(solver.dof_points)[0]) < 0.010
    np.savez_compressed(
        output / "mesh.npz",
        points_m=solver.points,
        tetrahedra=solver.tetrahedra,
        edges=solver.edges,
        boundary_faces=faces,
        boundary_regions=regions,
    )
    np.savez_compressed(
        output / "time.npz",
        frame_times_s=frame_times,
        frame_inlet_flow_m3_s=inflow[stride - 1 :: stride],
    )
    series: dict[str, Any] = {"port_names": np.asarray(names), **steps}
    np.savez_compressed(output / "steps.npz", **series)
    report = {
        "dt_s": dt,
        "precision": precision,
        "gmres_tolerance": solver.tolerance,
        "steps": count,
        "frames": frames,
        "velocity_nodes": len(solver.dof_points),
        "pressure_nodes": len(solver.points),
        "tetrahedra": len(solver.tetrahedra),
        "device": wp.get_device().name,
        "setup_seconds": setup_seconds,
        "stepping_seconds": stepping_seconds,
        "mean_step_seconds": float(steps["seconds"].mean()),
        "warp_mempool_peak_mb": wp.get_mempool_used_mem_high(wp.get_device()) / 2**20,
        "last_beat": summarize(
            names, steps, speeds, near_outlet, solver.tolerance, start
        ),
        "config": config,
    }
    (output / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Run the 3D flow simulation on a prepared lumen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument(
        "--preparation", type=Path, default=ROOT / "output" / "prepared"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "flow")
    parser.add_argument("--dt", type=float, help="Override the configured step")
    parser.add_argument("--precision", choices=["float32", "float64"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    wp.init()
    report = run(
        args.preparation.resolve(),
        args.output.resolve(),
        config,
        args.dt or config["dt_s"],
        args.precision or config["precision"],
    )
    print(json.dumps(report["last_beat"], indent=2))


if __name__ == "__main__":
    main()
