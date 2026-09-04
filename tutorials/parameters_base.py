"""Base class holding the three directory roots every tutorial reads and writes.

The dataset-specific parameter modules (:mod:`parameters_duke_heart_labelmaps`,
:mod:`parameters_lung_ct_dirlab`, :mod:`parameters_heart_ct_kcl`) derive from
:class:`ParametersBase`, so a tutorial resolves every path through the same
object it already imports for its label ids and iteration counts. A tutorial
that needs only the roots -- one that reads a dataset no shape model is built
from, say -- instantiates :class:`ParametersBase` directly.

Each root is overridable by environment variable, so a machine that keeps them
outside the clone -- a CI runner whose checkout is wiped every run, for
instance -- can point at them without editing any script:

===================================  =====================================
Variable                             Default
===================================  =====================================
``MONAI_PHYSIO_INPUT_DATA_DIR``        ``<repo>/data``
``MONAI_PHYSIO_OUTPUT_DATA_DIR``       ``<repo>/tutorials/output``
``MONAI_PHYSIO_WEIGHTS_DIR``           ``<repo>/tutorials/network_weights``
===================================  =====================================

Every root has a ``test`` subdirectory holding the small, fast counterpart used
when a tutorial runs under ``TestTools.running_as_test``. Keeping the test data,
the test results and the test checkpoints in their own subtree is what stops a
test run from reading or overwriting the datasets, results and trained networks
of a full run.

The weights root is kept apart from the output root because a trained network is
not a per-run result: Tutorial 9 trains one and Tutorial 10 loads it, so it
outlives the run that produced it. The pretrained weights ICON fetches for
itself are not here either; ICON resolves those on its own.
"""

from __future__ import annotations

import os
from pathlib import Path

# tutorials/parameters_base.py -> tutorials -> repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent


class ParametersBase:
    """The dataset, result and trained-network roots, per run mode.

    Not a dataclass: it carries no settings of its own, only the resolution the
    dataset-specific subclasses build their paths on. Those subclasses are
    frozen dataclasses, which a plain base class supports and a non-frozen
    dataclass base would not.
    """

    @staticmethod
    def _root(variable: str, default: Path, test_mode: bool) -> Path:
        """Return the root ``variable`` names, or ``default``, plus any test subtree.

        An unset variable and one set to nothing are treated alike, because a CI
        expression that interpolates an undefined value yields the empty string,
        and reading that as a path would silently resolve to the working
        directory.
        """
        value = os.environ.get(variable, "").strip()
        root = Path(value) if value else default
        return root / "test" if test_mode else root

    def data_directory(self, test_mode: bool) -> Path:
        """Return the root every dataset is read from.

        Args:
            test_mode: Return the ``test`` subtree, holding the downsampled
                subsets the pytest fixtures build, rather than the full
                datasets.
        """
        return self._root("MONAI_PHYSIO_INPUT_DATA_DIR", _REPO_ROOT / "data", test_mode)

    def output_directory(self, test_mode: bool) -> Path:
        """Return the root every tutorial writes its results to.

        Args:
            test_mode: Return the ``test`` subtree, so that a test run cannot
                overwrite the results a full run left behind.
        """
        return self._root(
            "MONAI_PHYSIO_OUTPUT_DATA_DIR",
            _REPO_ROOT / "tutorials" / "output",
            test_mode,
        )

    def weights_directory(self, test_mode: bool) -> Path:
        """Return the root the tutorials train their networks into.

        Args:
            test_mode: Return the ``test`` subtree, so that the two-epoch models
                a test run trains cannot overwrite a real checkpoint.
        """
        return self._root(
            "MONAI_PHYSIO_WEIGHTS_DIR",
            _REPO_ROOT / "tutorials" / "network_weights",
            test_mode,
        )
