"""Derived primvar computations for VTK-to-USD conversion.

This module hosts a small registry of functions that compute new scalar or
vector primvars from existing ones, so that downstream colormap workflows can
visualize physically meaningful quantities (von Mises stress, principal
invariants, trace, magnitudes, etc.) without depending on the raw tensor
representation.

Adding a new derivation
-----------------------

1. Write a function ``derive_<name>(array: GenericArray) -> list[GenericArray]``.
   It inspects ``array`` (name, num_components, interpolation, data) and returns
   zero or more derived ``GenericArray`` objects. Returning an empty list means
   "this derivation does not apply."

2. Append the function to ``PRIMVAR_DERIVATIONS`` below.

The conversion pipeline calls ``derive_primvars`` to apply every registered
function to every existing primvar; the results are appended to the mesh's
``generic_arrays`` list.

Naming convention
-----------------

Use ``<source_name>_<DerivedName>`` (capitalized derived suffix). Capital
letters sort before lowercase in ASCII, so ``stress_VonMises`` is selected over
``stress_c0`` by ``USDTools.pick_color_primvar``'s alphabetical tiebreak.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .data_structures import DataType, GenericArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tensor reductions (pure math; no GenericArray plumbing)
# ---------------------------------------------------------------------------


def compute_von_mises_stress(stress_tensor: NDArray) -> NDArray:
    """Compute scalar von Mises stress from a row-major 9-component tensor.

    The input tensor layout is::

        [s_xx, s_xy, s_xz, s_yx, s_yy, s_yz, s_zx, s_zy, s_zz]

    Off-diagonal pairs are averaged to symmetrize, which is a no-op for an
    already-symmetric Cauchy stress tensor.

    Formula::

        sigma_VM = sqrt(0.5 * [(sxx-syy)^2 + (syy-szz)^2 + (szz-sxx)^2]
                        + 3.0 * (sxy^2 + syz^2 + szx^2))

    Args:
        stress_tensor: Array of shape ``(N, 9)`` or a flat length ``N * 9``.

    Returns:
        Float32 array of shape ``(N,)`` with per-element von Mises stress.
    """
    arr = np.asarray(stress_tensor, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size % 9 != 0:
            raise ValueError(
                f"Flat stress array length {arr.size} is not divisible by 9"
            )
        arr = arr.reshape(-1, 9)
    elif arr.ndim != 2 or arr.shape[1] != 9:
        raise ValueError(f"Expected (N, 9) stress tensor, got shape {arr.shape}")

    sxx, sxy, sxz = arr[:, 0], arr[:, 1], arr[:, 2]
    syx, syy, syz = arr[:, 3], arr[:, 4], arr[:, 5]
    szx, szy, szz = arr[:, 6], arr[:, 7], arr[:, 8]
    sym_xy = 0.5 * (sxy + syx)
    sym_yz = 0.5 * (syz + szy)
    sym_zx = 0.5 * (sxz + szx)

    deviatoric = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
    shear = 3.0 * (sym_xy**2 + sym_yz**2 + sym_zx**2)
    result: np.ndarray = np.sqrt(np.maximum(deviatoric + shear, 0.0)).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Per-primvar derivation functions
# ---------------------------------------------------------------------------


PrimvarDerivation = Callable[[GenericArray], list[GenericArray]]
"""Signature for a primvar-derivation function.

A derivation inspects one source ``GenericArray`` and returns zero or more
newly-computed ``GenericArray`` objects. Return ``[]`` when the derivation
does not apply to the source array.
"""


def derive_von_mises_from_stress(array: GenericArray) -> list[GenericArray]:
    """Derive a scalar VonMises primvar from any 9-component stress tensor.

    Matches any source array whose name contains ``"stress"``
    (case-insensitive) and has 9 components.
    """
    if array.num_components != 9 or "stress" not in array.name.lower():
        return []
    try:
        vm = compute_von_mises_stress(array.data)
    except Exception as exc:
        logger.debug("Skipping VonMises derivation for '%s': %s", array.name, exc)
        return []
    return [
        GenericArray(
            name=f"{array.name}_VonMises",
            data=vm,
            num_components=1,
            data_type=DataType.FLOAT,
            interpolation=array.interpolation,
        )
    ]


# ---------------------------------------------------------------------------
# Registry and dispatch
# ---------------------------------------------------------------------------


PRIMVAR_DERIVATIONS: list[PrimvarDerivation] = [
    derive_von_mises_from_stress,
]
"""Ordered list of derivations applied to every input primvar.

Append a function to extend the pipeline. External callers may also mutate
this list (e.g. to register project-specific derivations at import time), but
keep in mind the list is module-global.
"""


def derive_primvars(arrays: list[GenericArray]) -> list[GenericArray]:
    """Apply every registered derivation to every array in ``arrays``.

    Args:
        arrays: Source primvars. Not modified.

    Returns:
        The newly-derived primvars only (does not include the originals).
        Callers typically extend their source list with the result.
    """
    derived: list[GenericArray] = []
    for array in arrays:
        for deriver in PRIMVAR_DERIVATIONS:
            derived.extend(deriver(array))
    return derived
