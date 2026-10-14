"""Benchmark RegisterImagesFireants against RegisterImagesGreedy.

Rebuilds the tutorial_03 (heart and lung) workflow: same data, same reference
frame, ``WorkflowReconstructHighres4DCT`` with an injected registrar. Every
non-reference frame is registered to the reference frame; speed and proxy
accuracy metrics are written to one result JSON (schema: VERIFY.md).

Usage:
    py benchmark.py --dataset heart --method greedy --out <json>
    py benchmark.py --dataset lung --method fireants --out <json> [--smoke]
        [--config <json>]

``--smoke`` uses the heart test subset (``data/test/slicer_heart_small``) and
minimal iterations; the lung smoke run uses the full lung files with minimal
iterations. ``--config`` overrides ``number_of_iterations``,
``transform_type``, ``metric``, ``scales`` and the FireANTs deformable options
(``RegisterImagesFireants.set_deformable_options``). ``--phases`` registers only
the listed phases directly with the registrar (tuning rounds; no workflow).

Timing: one warm-up pair is registered and discarded, then the workflow runs
and every ``register()`` call is timed (CUDA synchronized before and after for
FireANTs). Metrics are computed outside the timed region, directly after each
pair, because the transforms of all pairs would not fit in memory together;
the workflow receives identity placeholders in their place.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
import json
import logging
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import itk
import numpy as np
import torch
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tutorials"))

from parameters_heart_ct_kcl import HEART_CT_KCL
from parameters_tcia_4d_lung import TCIA_4D_LUNG

from monai_physio import (
    ProcessTransforms,
    RegisterImagesBase,
    RegisterImagesFireants,
    WorkflowReconstructHighres4DCT,
)

LOGGER = logging.getLogger("benchmark")

LUNG_THRESHOLD_HU = -320.0
BACKGROUND_HU = -1000.0
SMOKE_ITERATIONS = [1, 0]
CONFIG_KEYS = (
    "number_of_iterations",
    "transform_type",
    "metric",
    "scales",
    "init_affine",
    *sorted(RegisterImagesFireants._DEFORMABLE_OPTIONS),
)
INVERSE_POINTS_PER_AXIS = 28
INTERIOR_MARGIN = 0.1
DEFORMABLE_KEYS = RegisterImagesFireants._DEFORMABLE_OPTIONS


def load_dataset(dataset: str, smoke: bool) -> tuple[list[Path], int]:
    """Return the phase files and the tutorial's reference frame index."""
    if dataset == "heart":
        data_dir = HEART_CT_KCL.data_directory(smoke) / (
            "slicer_heart_small" if smoke else "Slicer-Heart-CT"
        )
        files = sorted(data_dir.glob("slice_???.mha"))
        reference = int(0.7 * len(files))
    else:
        data_dir = TCIA_4D_LUNG.input_directory(False)
        files = sorted(data_dir.glob("100_HM10395_g0??.nii.gz"))
        reference = next(i for i, p in enumerate(files) if "g070" in p.name)
    if not files:
        raise FileNotFoundError(f"No {dataset} phase images found under {data_dir}")
    return files, reference


def build_registrar(
    dataset: str, method: str, smoke: bool, overrides: dict[str, Any]
) -> RegisterImagesBase:
    """Greedy: the tutorial registrar; FireANTs: the class defaults."""
    if method == "greedy":
        parameters = HEART_CT_KCL if dataset == "heart" else TCIA_4D_LUNG
        registrar = parameters.registrar(smoke, log_level=logging.WARNING)
    else:
        registrar = RegisterImagesFireants(log_level=logging.WARNING)
        registrar.set_metric("CC")
        if smoke:
            registrar.set_number_of_iterations(list(SMOKE_ITERATIONS))
    if "number_of_iterations" in overrides:
        registrar.set_number_of_iterations(list(overrides["number_of_iterations"]))  # type: ignore[attr-defined]
    if "transform_type" in overrides:
        registrar.set_transform_type(overrides["transform_type"])  # type: ignore[attr-defined]
    if "metric" in overrides:
        registrar.set_metric(overrides["metric"])  # type: ignore[attr-defined]
    if "scales" in overrides:
        registrar.set_scales(list(overrides["scales"]))  # type: ignore[attr-defined]
    if "init_affine" in overrides:
        registrar.set_init_affine(overrides["init_affine"])  # type: ignore[attr-defined]
    options = {k: v for k, v in overrides.items() if k in DEFORMABLE_KEYS}
    if options:
        registrar.set_deformable_options(**options)  # type: ignore[attr-defined]
    registrar.set_modality("ct")
    return registrar


def registrar_config(method: str, registrar: Any) -> dict[str, Any]:
    """Full resolved configuration of the registrar, using P-list names."""
    config: dict[str, Any] = {
        "number_of_iterations": list(registrar.number_of_iterations),
        "metric": registrar.metric,
        "transform_type": registrar.transform_type,
    }
    if method == "greedy":
        config["deformable_smoothing"] = registrar.deformable_smoothing
        config["greedy_metric"] = registrar._greedy_metric()
        return config
    from monai_physio.register_images_fireants import _try_import_fireants

    greedy_class = _try_import_fireants("registration.greedy").GreedyRegistration
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(
            greedy_class.__init__
        ).parameters.items()
    }
    defaults.update(registrar.deformable_options)
    config["scales"] = registrar._scales()
    config["loss_type"] = registrar._FIREANTS_LOSS[registrar.metric]
    config["workflow"] = (
        "FireANTs AffineRegistration (library defaults) -> GreedyRegistration"
        if registrar.init_affine == "Affine"
        else "GreedyRegistration from identity"
    )
    config["init_affine"] = registrar.init_affine
    for name in sorted(DEFORMABLE_KEYS):
        config[name] = defaults[name]
    return config


def hardware_info() -> dict[str, Any]:
    """CPU, RAM, GPU and OS description."""
    import psutil

    gpu = torch.cuda.get_device_properties(0)
    return {
        "cpu": platform.processor(),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": gpu.name,
        "gpu_memory_gb": round(gpu.total_memory / 2**30, 1),
        "os": platform.platform(),
    }


def version_info() -> dict[str, Any]:
    """Library versions plus which FireANTs code path is in use."""
    from importlib import metadata

    def version(name: str) -> str:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return "not installed"

    fcntl_module = sys.modules.get("fcntl")
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "fireants": version("fireants"),
        "picsl_greedy": version("picsl_greedy"),
        "itk": version("itk"),
        "simpleitk": version("simpleitk"),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "fused_ops": importlib.util.find_spec("fireants_fused_ops") is not None,
        "fcntl_stub": fcntl_module is not None and not hasattr(fcntl_module, "flock"),
    }


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Global Pearson correlation over all voxels."""
    a = a.astype("float64").ravel()
    b = b.astype("float64").ravel()
    a -= a.mean()
    b -= b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared intensity difference over all voxels."""
    return float(np.mean((a.astype("float64") - b.astype("float64")) ** 2))


def lung_mask(array: np.ndarray) -> np.ndarray:
    """Threshold, drop in-plane-border components, keep the two largest.

    ``array`` axes are (z, y, x); components touching a y or x face are air
    outside the body. Z faces are not used because a truncated scan can cut
    through the lungs.
    """
    labels, count = ndimage.label(array < LUNG_THRESHOLD_HU)
    border = np.unique(
        np.concatenate(
            [
                labels[:, 0, :].ravel(),
                labels[:, -1, :].ravel(),
                labels[:, :, 0].ravel(),
                labels[:, :, -1].ravel(),
            ]
        )
    )
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    sizes[border] = 0
    keep = np.argsort(sizes)[-2:]
    return np.isin(labels, keep[sizes[keep] > 0])


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap of two boolean masks."""
    total = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / total if total else 1.0


def mask_image(mask: np.ndarray, reference: itk.Image) -> itk.Image:
    """Wrap a boolean mask array as a uint8 image on ``reference``'s grid."""
    image = itk.image_from_array(mask.astype(np.uint8))
    image.CopyInformation(reference)
    return image


def resample(
    image: itk.Image,
    transform: itk.Transform,
    reference: itk.Image,
    nearest: bool = False,
    background: float = BACKGROUND_HU,
) -> itk.Image:
    """Resample ``image`` onto ``reference`` with a point transform."""
    interpolator = (
        itk.NearestNeighborInterpolateImageFunction.New(image)
        if nearest
        else itk.LinearInterpolateImageFunction.New(image)
    )
    dtype = itk.array_view_from_image(image).dtype
    default = round(background) if np.issubdtype(dtype, np.integer) else background
    return itk.resample_image_filter(
        Input=image,
        Transform=transform,
        Interpolator=interpolator,
        ReferenceImage=reference,
        UseReferenceImage=True,
        DefaultPixelValue=default,
    )


def interior_points(reference: itk.Image) -> np.ndarray:
    """Physical positions of a coarse lattice inside the grid, shape (n, 3)."""
    size = np.array([int(s) for s in reference.GetLargestPossibleRegion().GetSize()])
    axes = [
        np.linspace(
            INTERIOR_MARGIN * (n - 1),
            (1 - INTERIOR_MARGIN) * (n - 1),
            INVERSE_POINTS_PER_AXIS,
        )
        for n in size
    ]
    index = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    spacing = np.array(itk.spacing(reference))
    direction = itk.array_from_matrix(reference.GetDirection())
    origin = np.array(itk.origin(reference))
    return origin + (index * spacing) @ direction.T


def inverse_consistency_mm(
    points: np.ndarray, forward: itk.Transform, inverse: itk.Transform
) -> float:
    """Mean |inverse(forward(p)) - p| over ``points``."""
    errors = np.empty(len(points))
    for i, point in enumerate(points):
        there = forward.TransformPoint([float(v) for v in point])
        back = inverse.TransformPoint([float(v) for v in there])
        errors[i] = np.linalg.norm(np.array([back[k] for k in range(3)]) - point)
    return float(errors.mean())


class Evaluator:
    """Per-pair accuracy metrics against the fixed image."""

    def __init__(self, fixed: itk.Image, dataset: str, images_dir: Path) -> None:
        self.fixed = fixed
        self.fixed_array = itk.array_from_image(fixed)
        self.dataset = dataset
        self.images_dir = images_dir
        self.points = interior_points(fixed)
        self.tools = ProcessTransforms()
        self.fixed_mask = lung_mask(self.fixed_array) if dataset == "lung" else None
        images_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_path = images_dir / "fixed.mha"
        itk.imwrite(fixed, str(self.fixed_path), compression=True)

    def evaluate(
        self,
        phase: int,
        moving: itk.Image,
        forward: itk.Transform,
        inverse: itk.Transform,
    ) -> dict[str, Any]:
        """Warp ``moving`` onto the fixed grid and measure the result."""
        field = self.tools.convert_transform_to_displacement_field(
            forward, self.fixed, np_component_type=np.float64
        )
        field_transform = itk.DisplacementFieldTransform[itk.D, 3].New()
        field_transform.SetDisplacementField(field)
        warped = resample(moving, field_transform, self.fixed)
        warped_path = self.images_dir / f"phase_{phase:03d}_warped.mha"
        itk.imwrite(warped, str(warped_path), compression=True)
        warped_array = itk.array_from_image(warped)

        identity = itk.IdentityTransform[itk.D, 3].New()
        unregistered = itk.array_from_image(resample(moving, identity, self.fixed))
        jacobian = itk.array_from_image(
            self.tools.compute_jacobian_determinant_from_field(field)
        )
        row: dict[str, Any] = {
            "ncc": ncc(warped_array, self.fixed_array),
            "mse": mse(warped_array, self.fixed_array),
            "neg_jacobian_fraction": float(np.mean(jacobian <= 0.0)),
            "inverse_consistency_mm": inverse_consistency_mm(
                self.points, forward, inverse
            ),
            "ncc_unregistered": ncc(unregistered, self.fixed_array),
            "mse_unregistered": mse(unregistered, self.fixed_array),
        }
        if self.fixed_mask is not None:
            moving_mask = mask_image(lung_mask(itk.array_from_image(moving)), moving)
            warped_mask = itk.array_from_image(
                resample(moving_mask, field_transform, self.fixed, True, 0.0)
            ).astype(bool)
            unregistered_mask = itk.array_from_image(
                resample(moving_mask, identity, self.fixed, True, 0.0)
            ).astype(bool)
            row["dice"] = dice(self.fixed_mask, warped_mask)
            row["dice_unregistered"] = dice(self.fixed_mask, unregistered_mask)
        row["warped"] = relative_path(warped_path)
        row["fixed"] = relative_path(self.fixed_path)
        return row


def relative_path(path: Path) -> str:
    """Path relative to the repository root when inside it, else absolute."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


class TimedRegistrar:
    """Time ``register`` calls and hand pairs to an evaluator.

    The wrapped registrar's ``register`` is replaced on the instance, so the
    workflow reaches it through the usual ``RegisterTimeSeriesImages`` path.
    """

    def __init__(
        self,
        registrar: RegisterImagesBase,
        method: str,
        on_pair: Optional[Callable[[itk.Image, dict], dict[str, Any]]] = None,
    ) -> None:
        self.registrar = registrar
        self.cuda = method == "fireants"
        self.on_pair = on_pair
        self.rows: list[dict[str, Any]] = []
        self.affine_seconds = 0.0
        self.original_register = registrar.register
        registrar.register = self.register  # type: ignore[method-assign]
        if hasattr(registrar, "_registration_method_affine_or_rigid"):
            self.original_affine = registrar._registration_method_affine_or_rigid
            registrar._registration_method_affine_or_rigid = (  # type: ignore[attr-defined]
                self.affine
            )

    def sync(self) -> None:
        """Wait for queued GPU work (FireANTs only)."""
        if self.cuda:
            torch.cuda.synchronize()

    def affine(self, *args: Any, **kwargs: Any) -> Any:
        """Timed FireANTs affine stage."""
        self.sync()
        start = time.perf_counter()
        result = self.original_affine(*args, **kwargs)
        self.sync()
        self.affine_seconds += time.perf_counter() - start
        return result

    def register(self, moving_image: itk.Image, **kwargs: Any) -> dict[str, Any]:
        """Timed ``register``; metrics are computed after the clock stops."""
        self.affine_seconds = 0.0
        self.sync()
        start = time.perf_counter()
        result = self.original_register(moving_image, **kwargs)
        self.sync()
        seconds = time.perf_counter() - start
        affine_seconds = self.affine_seconds
        if self.on_pair is None:
            self.last_seconds = seconds
            return result
        row: dict[str, Any] = {"seconds": seconds}
        if self.cuda:
            row["affine_seconds"] = affine_seconds
            row["deformable_seconds"] = seconds - affine_seconds
        row["loss"] = float(result["loss"])
        row.update(self.on_pair(moving_image, result))
        self.rows.append(row)
        identity = itk.IdentityTransform[itk.D, 3].New()
        del result["fixed_to_moving_transform"], result["moving_to_fixed_transform"]
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "fixed_to_moving_transform": identity,
            "moving_to_fixed_transform": identity,
            "loss": row["loss"],
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one benchmark and return the result dictionary."""
    overrides: dict[str, Any] = {}
    if args.config:
        overrides = json.loads(Path(args.config).read_text(encoding="utf-8"))
        unknown = set(overrides) - set(CONFIG_KEYS)
        if unknown:
            raise ValueError(f"Unsupported --config keys: {sorted(unknown)}")

    phases = (
        [int(p) for p in args.phases.split(",")] if args.phases is not None else None
    )
    files, reference_index = load_dataset(args.dataset, args.smoke)
    images = [itk.imread(str(path)) for path in files]
    reference = images[reference_index]
    others = [i for i in range(len(images)) if i != reference_index]
    LOGGER.info(
        "%s: %d phases, reference %s",
        args.dataset,
        len(images),
        files[reference_index].name,
    )

    registrar = build_registrar(args.dataset, args.method, args.smoke, overrides)
    config = registrar_config(args.method, registrar)
    LOGGER.info("config: %s", config)

    images_dir = args.images_dir or args.out.parent / "images" / args.out.stem
    evaluator = Evaluator(reference, args.dataset, images_dir)

    registrar.set_fixed_image(reference)
    LOGGER.info("warm-up pair (discarded)")
    registrar.register(images[others[0]])

    def on_pair(moving: itk.Image, result: dict) -> dict[str, Any]:
        phase = next(i for i, image in enumerate(images) if image is moving)
        LOGGER.info("evaluating phase %d", phase)
        row = evaluator.evaluate(
            phase,
            moving,
            result["fixed_to_moving_transform"],
            result["moving_to_fixed_transform"],
        )
        row["phase"] = phase
        row["phase_name"] = files[phase].name
        return row

    timed = TimedRegistrar(registrar, args.method, on_pair)
    if phases is not None:
        for phase in phases:
            registrar.register(images[phase])
    else:
        workflow = WorkflowReconstructHighres4DCT(
            time_series_images=images,
            reference_image=reference,
            reference_time_frame=reference_index,
            register_reference_time_frame_to_reference_image=False,
            registration_method=registrar,
            log_level=logging.WARNING,
        )
        workflow.set_modality("ct")
        workflow.register_time_series()

    repeat_phase = others[len(others) // 2]
    timed.on_pair = None
    repeat_seconds: list[float] = []
    for repeat in range(args.repeats):
        LOGGER.info("timed repeat %d of phase %d", repeat + 1, repeat_phase)
        registrar.register(images[repeat_phase])
        repeat_seconds.append(timed.last_seconds)

    per_phase = timed.rows
    ordered = ["phase", "phase_name", "seconds"]
    per_phase = [
        {
            **{k: row[k] for k in ordered},
            **{k: v for k, v in row.items() if k not in ordered},
        }
        for row in per_phase
    ]
    return {
        "dataset": args.dataset,
        "method": args.method,
        "smoke": args.smoke,
        "config": config,
        "hardware": hardware_info(),
        "versions": version_info(),
        "n_phases": len(images),
        "reference_frame": reference_index,
        "reference_frame_name": files[reference_index].name,
        "warmup_excluded": True,
        "cuda_synchronized": args.method == "fireants",
        "timing_notes": (
            "seconds = wall clock of registrar.register() (preprocess, "
            "registration, inverse transform); metrics are outside the clock. "
            "affine_seconds/deformable_seconds (FireANTs only): deformable is "
            "register() minus the affine stage. repeat_seconds re-times the "
            f"pair repeat_phase={repeat_phase} after the workflow run."
        ),
        "repeat_phase": repeat_phase,
        "total_seconds": float(sum(row["seconds"] for row in per_phase)),
        "per_phase": per_phase,
        "repeat_seconds": repeat_seconds,
        "repeat_median_seconds": (
            statistics.median(repeat_seconds) if repeat_seconds else None
        ),
    }


def main() -> None:
    """Parse arguments, run the benchmark and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=["heart", "lung"], required=True)
    parser.add_argument("--method", choices=["greedy", "fireants"], required=True)
    parser.add_argument(
        "--config", help="JSON overriding registrar settings (CONFIG_KEYS)"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phases", help="comma-separated phases (tuning rounds)")
    parser.add_argument("--images-dir", type=Path, help="warped-image directory")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    result = run(args)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s (total %.1f s)", args.out, result["total_seconds"])


if __name__ == "__main__":
    main()
