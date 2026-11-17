"""Simulate two-way coupled blood flow in a compliant aorta (FSI demonstration).

The rigid-flow setup of experiment 2 (P2/P1 Warp FEM, smoothed LV inflow with
a blunt root profile, RCR outlets with backflow stabilization) runs on the
candidate-20 lumen, now bounded by a Neo-Hookean wall that is prestressed to
the initial outlet pressure, clamped at the port collars and tethered on its
outer surface. ``WarpFluidStructure`` couples them each step. Fields are saved
every ``frame_interval_s``; ``run.json`` summarizes the reported beat.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import warp as wp

from monai_physio.warp_fluid_structure import WarpFluidStructure, WarpHyperelasticWall
from monai_physio.warp_navier_stokes import WarpNavierStokes, Windkessel, inlet_flow

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
MMHG = 133.322


def summarize(
    names: list[str], steps: dict[str, np.ndarray], start: float, cap: int
) -> dict[str, Any]:
    """Pressures, flow split, wall motion, coupling and conservation from ``start``.

    ``cap`` is the coupling iteration limit; steps that reach it are counted
    as capped even when their re-solve converges.
    """
    beat = steps["time_s"] >= start
    flows = steps["port_flow_m3_s"]
    inflow = -flows[:, names.index("root_inlet")]
    root = steps["root_pressure_pa"][beat] / MMHG
    outlets = {}
    for column, name in enumerate(n for n in names if n != "root_inlet"):
        flow = flows[beat, names.index(name)]
        pressure = steps["outlet_pressure_pa"][beat, column] / MMHG
        outlets[name] = {
            "mean_flow_ml_s": float(flow.mean() * 1e6),
            "flow_share": float(flow.mean() / inflow[beat].mean()),
            "systolic_mmhg": float(pressure.max()),
            "diastolic_mmhg": float(pressure.min()),
        }
    # Inflow - outflow - dV/dt, with V the volume of the moved fluid mesh.
    imbalance = flows.sum(axis=1) + steps["volume_rate_m3_s"]
    volume = steps["volume_m3"][beat]
    iterations = steps["coupling_iterations"][beat]
    return {
        "root_systolic_mmhg": float(root.max()),
        "root_diastolic_mmhg": float(root.min()),
        "root_mean_mmhg": float(root.mean()),
        "inlet_mean_flow_ml_s": float(inflow[beat].mean() * 1e6),
        "outlets": outlets,
        "peak_wall_displacement_mm": float(
            steps["max_displacement_m"][beat].max() * 1e3
        ),
        "mean_wall_displacement_mm": float(
            steps["mean_displacement_m"][beat].mean() * 1e3
        ),
        "lumen_volume_change_ml": float(np.ptp(volume) * 1e6),
        "lumen_volume_change_percent": float(np.ptp(volume) / volume.min() * 100),
        "coupling_iterations_mean": float(iterations.mean()),
        "coupling_iterations_max": int(iterations.max()),
        "capped_steps": int((iterations > cap).sum()),
        "nonconverged_steps": int((~steps["converged"][beat]).sum()),
        "max_coupling_residual": float(steps["coupling_residual"][beat].max()),
        "median_coupling_residual": float(np.median(steps["coupling_residual"][beat])),
        "wall_tangent_updates": int(steps["wall_tangent_updates"].sum()),
        "max_volume_imbalance_relative": float(
            np.abs(imbalance[beat]).max() / inflow.max()
        ),
        "beat_volume_imbalance_relative": float(
            abs(imbalance[beat].sum()) / inflow[beat].sum()
        ),
        "max_linear_residual": float(steps["residual"].max()),
        "mean_step_seconds": float(steps["seconds"][beat].mean()),
    }


def run(
    preparation: Path, output: Path, config: dict[str, Any], count: Optional[int]
) -> dict[str, Any]:
    """Solve ``count`` steps (default: the configured duration) and save results."""
    mesh = np.load(preparation / "mesh.npz")
    ports = json.loads((preparation / "ports.json").read_text(encoding="utf-8"))
    flux = np.load(REPOSITORY / config["waveform"])["inlet_interval_flux_m3_s"]
    settings = config["windkessel"]
    outlets = {
        port["region_id"]: Windkessel.from_total(
            settings["flow_fractions"][name],
            settings["total_resistance_pa_s_m3"],
            settings["total_compliance_m3_pa"],
            settings["proximal_fraction"],
            settings["initial_pressure_pa"],
        )
        for name, port in ports.items()
        if port["role"] == "outlet"
    }
    names = sorted(ports, key=lambda n: ports[n]["region_id"])
    dt = config["dt_s"]
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    faces, regions = mesh["fluid_boundary_faces"], mesh["fluid_boundary_regions"]
    fluid = WarpNavierStokes(
        mesh["fluid_points_m"],
        mesh["fluid_tetrahedra"],
        faces,
        regions,
        inlet_region=ports["root_inlet"]["region_id"],
        outlets=outlets,
        dt=dt,
        density=config["density_kg_m3"],
        viscosity=config["viscosity_pa_s"],
        backflow=config["backflow_beta"],
        blunt_fraction=config["inlet_blunt_fraction"],
        tolerance=config["gmres_tolerance"],
        precision=config["precision"],
    )
    wall = WarpHyperelasticWall(
        mesh["solid_points_m"],
        mesh["solid_tetrahedra"],
        mesh["solid_clamped"],
        young_modulus=config["wall"]["young_modulus_pa"],
        poisson_ratio=config["wall"]["poisson_ratio"],
        tether_faces=mesh["solid_outer_faces"],
        tether_stiffness=config["wall"]["tether_stiffness_pa_m"],
    )
    coupled = WarpFluidStructure(
        fluid,
        wall,
        regions,
        reference_pressure=settings["initial_pressure_pa"],
        **config["coupling"],
    )
    setup_seconds = time.perf_counter() - started
    root = regions == ports["root_inlet"]["region_id"]
    root_weights = np.zeros(len(fluid.points))
    np.add.at(
        root_weights,
        faces[root],
        np.linalg.norm(fluid.area_vectors[root], axis=1)[:, None] / 3,
    )
    root_weights /= root_weights.sum()

    count = count or round(config["duration_s"] / dt)
    stride = round(config["frame_interval_s"] / dt)
    times = np.arange(1, count + 1) * dt
    inflow = inlet_flow(
        times, flux, config["waveform_interval_s"], config["waveform_smoothing_s"]
    )[0]
    frames = count // stride
    fields = {
        name: np.lib.format.open_memmap(
            output / f"{name}.npy", mode="w+", dtype=np.float64, shape=shape
        )
        for name, shape in {
            "velocity_m_s": (frames, len(fluid.dof_points), 3),
            "pressure_pa": (frames, len(fluid.points)),
            "displacement_m": (frames, len(fluid.points), 3),
        }.items()
    }
    series = {
        "port_flow_m3_s": (len(names),),
        "outlet_pressure_pa": (len(outlets),),
        "root_pressure_pa": (),
        "coupling_iterations": (),
        "converged": (),
        "coupling_residual": (),
        "gmres_iterations": (),
        "wall_iterations": (),
        "wall_tangent_updates": (),
        "residual": (),
        "max_displacement_m": (),
        "mean_displacement_m": (),
        "volume_m3": (),
        "volume_rate_m3_s": (),
        "seconds": (),
    }
    steps = {name: np.zeros((count, *shape)) for name, shape in series.items()}
    steps["converged"] = np.zeros(count, dtype=bool)
    for index, flow in enumerate(inflow):
        tic = time.perf_counter()
        result = coupled.step(float(flow))
        steps["seconds"][index] = time.perf_counter() - tic
        for name in series:
            if name in result:
                steps[name][index] = result[name]
        steps["port_flow_m3_s"][index] = [
            result["flows_m3_s"][ports[n]["region_id"]] for n in names
        ]
        steps["outlet_pressure_pa"][index] = [
            result["pressures_pa"][ports[n]["region_id"]]
            for n in names
            if n != "root_inlet"
        ]
        current_pressure = fluid.pressure()
        steps["root_pressure_pa"][index] = root_weights @ current_pressure
        steps["volume_m3"][index] = coupled.volume
        if (index + 1) % stride == 0:
            frame = (index + 1) // stride - 1
            fields["velocity_m_s"][frame] = fluid.velocity()
            fields["pressure_pa"][frame] = current_pressure
            fields["displacement_m"][frame] = coupled.mesh_displacement
        print(
            f"t={times[index]:.4f} s Q={flow * 1e6:.1f} mL/s "
            f"root={steps['root_pressure_pa'][index] / MMHG:.1f} mmHg "
            f"wall={result['max_displacement_m'] * 1e3:.3f} mm "
            f"coupling={result['coupling_iterations']}"
            f"{'' if result['converged'] else ' (not converged)'} "
            f"r={result['coupling_residual']:.1e} GMRES={result['gmres_iterations']} "
            f"{steps['seconds'][index]:.1f} s",
            flush=True,
        )
    coupled.close()
    for array in fields.values():
        array.flush()
    np.savez_compressed(
        output / "mesh.npz",
        points_m=fluid.points,
        tetrahedra=fluid.tetrahedra,
        edges=fluid.edges,
        boundary_faces=faces,
        boundary_regions=regions,
    )
    np.savez_compressed(
        output / "time.npz",
        frame_times_s=times[stride - 1 :: stride][:frames],
        frame_inlet_flow_m3_s=inflow[stride - 1 :: stride][:frames],
    )
    saved: dict[str, Any] = {
        "port_names": np.asarray(names),
        "time_s": times,
        "inlet_flow_m3_s": inflow,
        **steps,
    }
    np.savez_compressed(output / "steps.npz", **saved)
    start = config["report_from_s"] if times[-1] > config["report_from_s"] else 0.0
    report = {
        "dt_s": dt,
        "precision": config["precision"],
        "gmres_tolerance": fluid.tolerance,
        "steps": count,
        "frames": frames,
        "fluid_velocity_nodes": len(fluid.dof_points),
        "fluid_tetrahedra": len(fluid.tetrahedra),
        "wall_nodes": len(wall.dof_points),
        "wall_tetrahedra": len(wall.tetrahedra),
        "device": wp.get_device().name,
        "setup_seconds": setup_seconds,
        "report_from_s": start,
        "summary": summarize(names, saved, start, coupled.max_iterations),
        "config": config,
    }
    (output / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Run the coupled simulation on the prepared meshes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument(
        "--preparation", type=Path, default=ROOT / "output" / "prepared"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "fsi")
    parser.add_argument("--steps", type=int, help="Run only this many steps")
    parser.add_argument("--precision", choices=["float32", "float64"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["precision"] = args.precision or config["precision"]
    wp.init()
    report = run(args.preparation.resolve(), args.output.resolve(), config, args.steps)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
