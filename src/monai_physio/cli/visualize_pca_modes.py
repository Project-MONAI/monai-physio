"""
Command-line interface to visualize PCA modes of variation.

Displays the first three principal components in a 1x3 PyVista plotter. A
slider (0 to 4) controls the standard-deviation magnitude; each subplot
shows the mean shape (gray), +sigma shape (coral), and -sigma shape (blue)
for that PC (matching experiments/Heart-Create_Statistical_Model/
5-compute_pca_model.ipynb cell 11).

Inputs: pca_mean_surface.vtp (mean mesh topology and mean shape) and
pca_model.json (components and eigenvalues).
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pyvista as pv


def _shape_at_sigma(
    mean_shape: "npt.NDArray[np.float64]",
    components: list["npt.NDArray[np.float64]"],
    eigenvalues: "npt.NDArray[np.float64]",
    pc_index: int,
    sigma: float,
) -> "npt.NDArray[np.float64]":
    """Return (n_points, 3) mesh points for mean + sigma * std_dev * component."""
    import numpy as np

    pc = np.asarray(components[pc_index], dtype=np.float64)
    std_dev = np.sqrt(eigenvalues[pc_index])
    variation = mean_shape + (sigma * std_dev * pc)
    n_points = mean_shape.size // 3
    return cast("npt.NDArray[np.float64]", variation.reshape(n_points, 3))


def _generate_pc_variation(
    mean_mesh: "pv.DataSet",
    mean_shape: "npt.NDArray[np.float64]",
    components: list["npt.NDArray[np.float64]"],
    eigenvalues: "npt.NDArray[np.float64]",
    pc_index: int,
    std_dev_multiplier: float = 3.0,
) -> tuple["pv.DataSet", "pv.DataSet", "pv.DataSet"]:
    """Generate shape variations along a principal component.

    Parameters
    ----------
    mean_mesh : pv.PolyData
        Template mesh (topology only; points are replaced).
    mean_shape : np.ndarray
        Flattened mean shape (n_points * 3,).
    components : list
        List of component vectors (each length n_points * 3).
    eigenvalues : list
        Variance (eigenvalue) for each component.
    pc_index : int
        Index of the principal component (0-based).
    std_dev_multiplier : float
        How many standard deviations to vary (default: +/-3 sigma).

    Returns
    -------
    tuple of (negative_mesh, mean_mesh, positive_mesh)
    """
    import numpy as np

    pc = np.asarray(components[pc_index], dtype=np.float64)
    std_dev = np.sqrt(eigenvalues[pc_index])

    negative_variation = mean_shape - (std_dev_multiplier * std_dev * pc)
    positive_variation = mean_shape + (std_dev_multiplier * std_dev * pc)

    n_points = mean_shape.size // 3
    negative_points = cast(
        "npt.NDArray[np.float64]", negative_variation.reshape(n_points, 3)
    )
    positive_points = cast(
        "npt.NDArray[np.float64]", positive_variation.reshape(n_points, 3)
    )
    mean_points = cast("npt.NDArray[np.float64]", mean_shape.reshape(n_points, 3))

    negative_mesh = mean_mesh.copy()
    negative_mesh.points = negative_points

    positive_mesh = mean_mesh.copy()
    positive_mesh.points = positive_points

    mean_mesh_out = mean_mesh.copy()
    mean_mesh_out.points = mean_points

    return negative_mesh, mean_mesh_out, positive_mesh


def main() -> int:
    """Command-line interface for visualizing PCA modes."""
    parser = argparse.ArgumentParser(
        description="Visualize the first three PCA modes with a 0-4 sigma slider; each subplot shows mean, +sigma, -sigma (PyVista 1x3 plot).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Slider 0-4 sigma; each subplot shows mean (gray), +sigma (coral), -sigma (blue)
  %(prog)s pca_mean_surface.vtp pca_model.json

  # Start with slider at 2 sigma
  %(prog)s --std-dev 2.0 mean.vtp pca_model.json
        """,
    )

    parser.add_argument(
        "pca_mean_surface",
        type=Path,
        metavar="PCA_MEAN_SURFACE.vtp",
        help="Path to PCA mean surface (.vtp); provides topology and mean shape.",
    )
    parser.add_argument(
        "pca_model_json",
        type=Path,
        metavar="pca_model.json",
        help="Path to PCA model JSON (components and eigenvalues).",
    )
    parser.add_argument(
        "--std-dev",
        type=float,
        default=0.0,
        metavar="N",
        help="Initial slider position 0-4 std dev (default: 0).",
    )

    args = parser.parse_args()

    if not args.pca_mean_surface.exists():
        print(f"Error: PCA mean surface not found: {args.pca_mean_surface}")
        return 1
    if not args.pca_model_json.exists():
        print(f"Error: PCA model JSON not found: {args.pca_model_json}")
        return 1

    try:
        import pyvista as pv

        mean_mesh = pv.read(str(args.pca_mean_surface))
    except (OSError, RuntimeError) as e:
        print(f"Error loading PCA mean surface: {e}")
        traceback.print_exc()
        return 1

    if not isinstance(mean_mesh, (pv.PolyData, pv.UnstructuredGrid)):
        print(
            "Error: PCA mean surface must be PolyData or UnstructuredGrid.",
            f"Type: {type(mean_mesh)}",
        )
        return 1

    try:
        with open(args.pca_model_json, encoding="utf-8") as f:
            pca_model = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error loading PCA model JSON: {e}")
        traceback.print_exc()
        return 1

    for key in ("components", "eigenvalues"):
        if key not in pca_model:
            print(f"Error: PCA model JSON must contain '{key}'.")
            return 1

    raw_components = pca_model["components"]
    raw_eigenvalues = pca_model["eigenvalues"]

    if len(raw_components) < 3:
        print(
            f"Error: PCA model has only {len(raw_components)} component(s); need at least 3 for visualization."
        )
        return 1

    n_points = mean_mesh.n_points
    n_features = n_points * 3
    if len(raw_components[0]) != n_features:
        print(
            f"Error: Mean surface has {n_points} points ({n_features} features), "
            f"but PCA components have {len(raw_components[0])} entries. Shapes must match."
        )
        return 1

    import numpy as np

    components = [
        np.asarray(component, dtype=np.float64) for component in raw_components
    ]
    eigenvalues = np.asarray(raw_eigenvalues, dtype=np.float64)
    mean_shape = mean_mesh.points.astype(np.float64).flatten()
    n_points = mean_shape.size // 3

    # Precompute component arrays for slider updates
    pc_arrays = components[:3]
    std_devs = [np.sqrt(eigenvalues[i]) for i in range(3)]

    plotter = pv.Plotter(shape=(1, 3))

    # Slider 0 to 4; clamp initial to that range
    initial_sigma = max(0.0, min(4.0, args.std_dev))

    # Mean shape for reference (same in all subplots)
    mean_ref = mean_mesh.copy()
    mean_ref.points = mean_shape.reshape(n_points, 3)

    # Per subplot: mean (static), +sigma mesh, -sigma mesh (updated by slider)
    plus_meshes: list[pv.DataSet] = []
    minus_meshes: list[pv.DataSet] = []
    for col, pc_index in enumerate(range(3)):
        points_plus = _shape_at_sigma(
            mean_shape, components, eigenvalues, pc_index, initial_sigma
        )
        points_minus = _shape_at_sigma(
            mean_shape, components, eigenvalues, pc_index, -initial_sigma
        )
        mesh_plus = mean_mesh.copy()
        mesh_plus.points = points_plus
        mesh_minus = mean_mesh.copy()
        mesh_minus.points = points_minus
        plus_meshes.append(cast(pv.DataSet, mesh_plus))
        minus_meshes.append(cast(pv.DataSet, mesh_minus))
        plotter.subplot(0, col)
        plotter.add_mesh(
            mean_ref.copy(),
            color="lightgray",
            opacity=0.4,
            show_edges=False,
        )
        plotter.add_mesh(
            mesh_minus,
            color="lightblue",
            opacity=1.0,
            show_edges=False,
        )
        plotter.add_mesh(
            mesh_plus,
            color="lightcoral",
            opacity=1.0,
            show_edges=False,
        )
        plotter.add_text(
            f"PC{pc_index + 1}",
            font_size=12,
        )
        plotter.camera_position = "iso"

    def _on_slider(sigma: float) -> None:
        for i in range(3):
            plus_pts = mean_shape + (sigma * std_devs[i] * pc_arrays[i])
            minus_pts = mean_shape - (sigma * std_devs[i] * pc_arrays[i])
            plus_meshes[i].points = plus_pts.reshape(n_points, 3)
            minus_meshes[i].points = minus_pts.reshape(n_points, 3)
        plotter.render()

    # Slider: 0 to 4 std dev (shows mean, +sigma, -sigma)
    plotter.subplot(0, 0)
    plotter.add_slider_widget(
        cast(Any, _on_slider),
        rng=(0.0, 4.0),
        value=initial_sigma,
        title="Std dev",
        pointa=(0.02, 0.1),
        pointb=(0.98, 0.1),
    )

    plotter.link_views()
    plotter.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
