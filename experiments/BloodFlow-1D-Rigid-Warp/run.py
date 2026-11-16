"""Evaluate rigid 1D aortic pressure and mean velocity on CUDA.

In a rigid tube incompressible flow Q(t) is the same at every section, the
section-mean velocity is Q/A and the pressure along the trunk is

    p(s, t) = P_out(t) + dQ/dt int_s^L rho/A ds + Q int_s^L 8 pi mu/A^2 ds,

where P_out is the pressure of a three-element Windkessel with the whole-body
resistance and compliance (all flow leaves through the trunk). Convective
acceleration, branch flow and wall motion are not modelled.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from monai_physio.simulate_aorta_flow_warp import SimulateAortaFlowWarp
from monai_physio.warp_navier_stokes import Windkessel

ROOT = Path(__file__).resolve().parent
MMHG = 133.322
LAST_BEAT_START_S = 2.01  # Third LV ejection onset, source sample 67.


@wp.kernel
def integrate_closure(
    arc: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
    area: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
    flux: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
    derivative: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
    density: wp.float64,
    drag: wp.float64,
    velocity: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
    drop: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
) -> None:
    """Trapezoidal inertance and Poiseuille resistance from a section to the outlet."""
    t, x = wp.tid()  # type: ignore[misc]
    inertance = wp.float64(0.0)
    resistance = wp.float64(0.0)
    for j in range(x, arc.shape[0] - 1):
        length = arc[j + 1] - arc[j]
        a = wp.float64(1.0) / area[j]
        b = wp.float64(1.0) / area[j + 1]
        inertance += density * length * (a + b) / wp.float64(2.0)
        resistance += drag * length * (a * a + b * b) / wp.float64(2.0)
    velocity[t, x] = flux[t] / area[x]
    drop[t, x] = inertance * derivative[t] + resistance * flux[t]


def evaluate(
    arc: np.ndarray,
    area: np.ndarray,
    flux: np.ndarray,
    derivative: np.ndarray,
    density: float,
    viscosity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean velocity and the pressure drop to the outlet (time x section)."""
    arrays: list[Any] = [
        wp.array(values, dtype=wp.float64) for values in (arc, area, flux, derivative)
    ]
    velocity = wp.empty((len(flux), len(arc)), dtype=wp.float64)
    drop = wp.empty_like(velocity)
    # 8 pi mu is formed in Python: Warp would round a literal to float32.
    drag = wp.float64(8 * math.pi * viscosity)
    wp.launch(
        integrate_closure,
        dim=velocity.shape,
        inputs=[*arrays, wp.float64(density), drag, velocity, drop],
    )
    return velocity.numpy(), drop.numpy()


def run(preparation: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the trunk fields over the configured duration and summarize them."""
    started = time.perf_counter()
    geometry = np.load(preparation / "geometry.npz")
    interval_flux = np.load(preparation / "waveform.npz")["inlet_interval_flux_m3_s"]
    dt = config["dt_s"]
    times = (np.arange(round(config["duration_s"] / dt)) + 0.5) * dt
    flux, derivative = SimulateAortaFlowWarp.inlet_flow(
        times, interval_flux, config["waveform_smoothing_s"]
    )
    arc, area = geometry["arc_length_m"], geometry["area_m2"]
    velocity, drop = evaluate(
        arc, area, flux, derivative, config["density_kg_m3"], config["viscosity_pa_s"]
    )
    settings = config["windkessel"]
    windkessel = Windkessel.from_total(
        1.0,
        settings["total_resistance_pa_s_m3"],
        settings["total_compliance_m3_pa"],
        settings["proximal_fraction"],
        settings["initial_pressure_pa"],
    )
    outlet = np.asarray([windkessel.advance(q, dt) for q in flux])
    pressure = drop + outlet[:, None]
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "fields.npz",
        times_s=times,
        arc_length_m=arc,
        area_m2=area,
        inlet_q_m3_s=flux,
        inlet_qdot_m3_s2=derivative,
        velocity_m_s=velocity,
        pressure_pa=pressure,
        outlet_pressure_pa=outlet,
    )
    last = times >= LAST_BEAT_START_S
    root = pressure[last, 0] / MMHG
    report = {
        "device": wp.get_device().name,
        "time_samples": len(times),
        "sections": len(arc),
        "last_beat": {
            "root_systolic_mmhg": float(root.max()),
            "root_diastolic_mmhg": float(root.min()),
            "root_mean_mmhg": float(root.mean()),
            "outlet_systolic_mmhg": float(outlet[last].max() / MMHG),
            "outlet_diastolic_mmhg": float(outlet[last].min() / MMHG),
            "max_root_to_outlet_drop_mmhg": float(drop[last, 0].max() / MMHG),
            "peak_inlet_flow_ml_s": float(flux[last].max() * 1e6),
            "peak_mean_velocity_m_s": float(velocity[last].max()),
        },
        "run_seconds": time.perf_counter() - started,
        "config": config,
    }
    (output / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Evaluate the 1D flow on a prepared trunk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument(
        "--preparation", type=Path, default=ROOT / "output" / "prepared"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "flow")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    wp.init()
    report = run(args.preparation.resolve(), args.output.resolve(), config)
    print(json.dumps(report["last_beat"], indent=2))


if __name__ == "__main__":
    main()
