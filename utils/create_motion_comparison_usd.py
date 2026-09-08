"""Create a synchronized predicted-versus-ground-truth motion USD package."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pyvista as pv
from pxr import Sdf, Usd, UsdGeom, UsdShade

from monai_physio import ConvertVTKToUSD


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predicted",
        nargs="+",
        type=Path,
        required=True,
        help="Ordered predicted VTP frames.",
    )
    parser.add_argument(
        "--ground-truth",
        nargs="+",
        type=Path,
        required=True,
        help="Ordered ground-truth VTP frames.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output comparison USD; two referenced USD layers are written beside it.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=3.0,
        help="Playback rate in frames per second. Default: 3.",
    )
    return parser.parse_args()


def _load_sequence(paths: Sequence[Path], series_name: str) -> list[pv.PolyData]:
    """Load and validate one fixed-topology surface sequence."""
    if len(paths) < 2:
        raise ValueError(f"{series_name} needs at least two frames.")

    meshes: list[pv.PolyData] = []
    expected_topology: Optional[tuple[int, int]] = None
    expected_connectivity: Optional[np.ndarray] = None
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{series_name} frame not found: {path}")
        mesh = pv.read(str(path))
        if not isinstance(mesh, pv.PolyData):
            raise TypeError(f"{series_name} frame is not PolyData: {path}")
        topology = (mesh.n_points, mesh.n_cells)
        if expected_topology is None:
            expected_topology = topology
            expected_connectivity = mesh.faces.copy()
        else:
            assert expected_connectivity is not None
            if topology != expected_topology or not np.array_equal(
                mesh.faces,
                expected_connectivity,
            ):
                raise ValueError(
                    f"{series_name} topology changes at {path}: {topology}, "
                    f"expected {expected_topology} with identical face connectivity."
                )
        meshes.append(mesh)
    return meshes


def _set_preview_opacity(stage: Usd.Stage, opacity: float) -> None:
    """Set opacity on every UsdPreviewSurface shader in a generated layer."""
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        shader_id = shader.GetIdAttr().Get()
        if shader_id != "UsdPreviewSurface":
            continue
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)


def _write_series(
    meshes: Sequence[pv.PolyData],
    output_file: Path,
    project_name: str,
    color: tuple[float, float, float],
    opacity: float,
    fps: float,
) -> None:
    """Write one solid-color, time-sampled USD layer."""
    converter = ConvertVTKToUSD(
        data_basename=project_name,
        input_polydata=meshes,
        frames_per_second=fps,
        separate_by="none",
        solid_color=color,
    )
    stage = converter.convert(str(output_file))
    _set_preview_opacity(stage, opacity)
    stage.Save()


def _write_composite(
    output_file: Path,
    prediction_layer: Path,
    ground_truth_layer: Path,
    frame_count: int,
    fps: float,
) -> None:
    """Reference two animated layers under named comparison prims."""
    if output_file.exists():
        output_file.unlink()
    stage = Usd.Stage.CreateNew(str(output_file))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    prediction = UsdGeom.Xform.Define(stage, "/World/Prediction")
    prediction.GetPrim().GetReferences().AddReference(f"./{prediction_layer.name}")
    ground_truth = UsdGeom.Xform.Define(stage, "/World/GroundTruth")
    ground_truth.GetPrim().GetReferences().AddReference(f"./{ground_truth_layer.name}")

    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(float(frame_count - 1))
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)
    stage.Save()


def main() -> int:
    """Create the paired animation and print its package paths."""
    args = _parse_args()
    if args.fps <= 0.0:
        raise ValueError(f"fps must be positive, got {args.fps}.")
    if len(args.predicted) != len(args.ground_truth):
        raise ValueError(
            "Prediction and ground-truth frame counts differ: "
            f"{len(args.predicted)} versus {len(args.ground_truth)}."
        )

    predicted = _load_sequence(args.predicted, "Prediction")
    ground_truth = _load_sequence(args.ground_truth, "Ground truth")
    predicted_topology = (predicted[0].n_points, predicted[0].n_cells)
    ground_truth_topology = (ground_truth[0].n_points, ground_truth[0].n_cells)
    if predicted_topology != ground_truth_topology:
        raise ValueError(
            "Prediction and ground-truth topologies differ: "
            f"{predicted_topology} versus {ground_truth_topology}."
        )

    output_file = args.output.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    prediction_layer = output_file.with_name(f"{output_file.stem}_prediction.usd")
    ground_truth_layer = output_file.with_name(f"{output_file.stem}_ground_truth.usd")

    _write_series(
        predicted,
        prediction_layer,
        f"{output_file.stem}_prediction",
        color=(1.0, 0.42, 0.42),
        opacity=0.65,
        fps=args.fps,
    )
    _write_series(
        ground_truth,
        ground_truth_layer,
        f"{output_file.stem}_ground_truth",
        color=(0.32, 0.81, 0.40),
        opacity=0.35,
        fps=args.fps,
    )
    _write_composite(
        output_file,
        prediction_layer,
        ground_truth_layer,
        len(predicted),
        args.fps,
    )

    print(f"Comparison USD: {output_file}")
    print(f"Prediction layer: {prediction_layer}")
    print(f"Ground-truth layer: {ground_truth_layer}")
    print(
        f"Frames: {len(predicted)}, points per series: {predicted[0].n_points}, "
        f"playback: {args.fps:g} FPS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
