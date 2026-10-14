"""Affine-only diagnostic: FireANTs vs Greedy on selected heart phases.

Registers each listed phase to the heart reference frame with an Affine-only
registrar (FireANTs class defaults vs the tutorial Greedy registrar) and
records NCC/MSE of the warped image plus the affine matrix summary.

Usage:
    py affine_diagnostic.py --out experiments/Registration-Fireants_vs_Greedy/results/affine_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import itk
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark


def affine_summary(transform: itk.Transform) -> dict[str, Any]:
    """Matrix determinant, singular values and translation of an affine."""
    matrix = itk.array_from_matrix(transform.GetMatrix())
    singular = np.linalg.svd(matrix, compute_uv=False)
    return {
        "determinant": float(np.linalg.det(matrix)),
        "singular_values": [float(v) for v in singular],
        "translation_mm": [float(v) for v in transform.GetTranslation()],
    }


def main() -> None:
    """Run the diagnostic and write the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phases", default="3,13,15")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    files, reference_index = benchmark.load_dataset("heart", False)
    images = [itk.imread(str(path)) for path in files]
    reference = images[reference_index]
    fixed_array = itk.array_from_image(reference)
    identity = itk.IdentityTransform[itk.D, 3].New()

    result: dict[str, Any] = {"dataset": "heart", "reference_frame": reference_index}
    rows: dict[str, list[dict[str, Any]]] = {"greedy": [], "fireants": []}
    for method in ("greedy", "fireants"):
        registrar = benchmark.build_registrar("heart", method, False, {})
        registrar.set_transform_type("Affine")
        registrar.set_fixed_image(reference)
        registrar.register(images[0])
        for phase in (int(p) for p in args.phases.split(",")):
            moving = images[phase]
            if method == "fireants":
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = registrar.register(moving)
            if method == "fireants":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            forward = out["fixed_to_moving_transform"]
            warped = itk.array_from_image(
                benchmark.resample(moving, forward, reference)
            )
            unregistered = itk.array_from_image(
                benchmark.resample(moving, identity, reference)
            )
            rows[method].append(
                {
                    "phase": phase,
                    "seconds": seconds,
                    "ncc": benchmark.ncc(warped, fixed_array),
                    "mse": benchmark.mse(warped, fixed_array),
                    "ncc_unregistered": benchmark.ncc(unregistered, fixed_array),
                    "mse_unregistered": benchmark.mse(unregistered, fixed_array),
                    "affine": affine_summary(forward),
                }
            )
    result["greedy"] = rows["greedy"]
    result["fireants"] = rows["fireants"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
