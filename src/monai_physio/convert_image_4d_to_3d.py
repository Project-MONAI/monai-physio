"""Convert a 3D or 4D image into a sequence of 3D images.

Reads a 3D or 4D medical image and, for the 4D case, splits it along the
temporal axis into individual 3D ITK volumes.  Origin, spacing, and direction
are preserved in each per-frame volume.  A pure 3D input becomes a one-element
time series.

Three reader paths are used:

* A *directory* path is treated as a DICOM series and read with ``pydicom``.
  The slices are grouped by temporal position (``TemporalPositionIdentifier``
  or ``TriggerTime``); each group yields one 3D ITK image.  Directories that
  contain a single phase produce a single 3D image.
* ``.nrrd`` files: 4D Slicer ``.seq.nrrd`` heart sequences (whose per-voxel
  vector dimension exceeds ITK Python's wrapped Vector sizes) go through
  ``pynrrd``.  Plain 3D NRRDs fall back to ``itk.imread``.
* Every other format goes through ``itk.imread`` and may be either 3D or 4D
  (e.g. NIfTI ``.nii.gz`` with ``dim[0] == 3`` or ``4``).
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Union

import itk
import nrrd
import numpy as np
import pydicom

from .monai_physio_base import MONAIPhysioBase


class ConvertImage4DTo3D(MONAIPhysioBase):
    """Split a 3D/4D ITK image into a list of 3D ITK images."""

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize the 4D-to-3D image converter.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)
        self.img_3d: list[itk.Image] = []

        # Public DICOM tags used to split a 4D series into 3D phases.  Maps
        # tag keyword → default value (the value type also implies how the
        # tag is parsed: ``float`` for numeric tags, ``str`` for string tags).
        # External users may add, remove, or replace entries to tune phase
        # grouping for their vendor-specific exports.
        self.dicom_phase_keys: dict[str, Union[float, str]] = {
            "TemporalPositionIdentifier": 0.0,
            "TriggerTime": 0.0,
            "NominalCardiacTriggerDelayTime": 0.0,
            "ActualCardiacTriggerDelayTime": 0.0,
            "NominalPercentageOfCardiacPhase": 0.0,
            "FrameReferenceDateTime": "",
            # "AcquisitionTime": "",
            "ScanOptions": "",
        }

    def load_image_4d(self, filename: Union[str, Path]) -> None:
        """Load a 3D or 4D image and populate ``self.img_3d`` with 3D frames.

        Dispatch rules:

        * A *directory* path is read as a DICOM series via ``pydicom``.
          Slices are grouped by temporal phase; each group becomes one 3D
          ITK image.  A 3D-only directory produces a single 3D image.
        * ``.nrrd`` files use ``pynrrd`` for true 4D Slicer ``.seq.nrrd``
          inputs and fall back to ``itk.imread`` for plain 3D NRRDs.
        * All other formats go through ``itk.imread``; the array may be
          3D or 4D and is treated uniformly as a (1 or T)-frame sequence.

        Args:
            filename: Path to a 3D/4D image file, or a DICOM series directory.
        """
        path = Path(filename)
        if path.is_dir():
            self._load_dicom_directory(path)
            return

        name = str(path)
        if name.lower().endswith(".nrrd"):
            data, header = nrrd.read(name)
            arr_data = np.asarray(data)
            if arr_data.ndim == 4:
                self._load_nrrd_4d(name, arr_data, header)
                return
            if arr_data.ndim != 3:
                raise ValueError(
                    f"Expected 3D or 4D NRRD, got array shape {arr_data.shape}: {name}"
                )
            # 3D NRRD: defer to the standard ITK reader for correctness.

        self._load_itk_file(name)

    def _load_itk_file(self, filename: str) -> None:
        """Read a 3D or 4D image with ``itk.imread`` and slice along T."""
        img = itk.imread(filename)
        arr = itk.array_view_from_image(img)
        if arr.ndim not in (3, 4):
            raise ValueError(
                f"Expected a 3D or 4D image, got array shape {arr.shape}: {filename}"
            )
        origin_3d = np.asarray(img.GetOrigin())[:3]
        spacing_3d = np.asarray(img.GetSpacing())[:3]
        direction_3d = itk.array_from_matrix(img.GetDirection())[:3, :3]

        if arr.ndim == 3:
            arr_4d = arr[np.newaxis, ...]
        else:
            arr_4d = arr

        self._build_frames(arr_4d, origin_3d, spacing_3d, direction_3d)

    def _load_nrrd_4d(
        self,
        filename: str,
        data: np.ndarray,
        header: dict[str, Any],
    ) -> None:
        """Build per-frame 3D ITK images from a Slicer 4D ``.seq.nrrd``."""
        # pynrrd returns the data in (T, X, Y, Z) order for a 4D NRRD.
        # ITK numpy views use (T, Z, Y, X) - transpose the spatial axes.
        arr_4d = np.ascontiguousarray(data.transpose(0, 3, 2, 1))

        required_keys = ("space origin", "space directions", "measurement frame")
        missing = [k for k in required_keys if k not in header]
        if missing:
            raise ValueError(
                f"{filename!r} is not a valid Slicer 4D .seq.nrrd: "
                f"missing NRRD header field(s) {missing}"
            )
        space_directions = np.asarray(header["space directions"])
        measurement_frame = np.asarray(header["measurement frame"])
        if (
            space_directions.ndim != 2
            or space_directions.shape[0] < 4
            or space_directions.shape[1] < 3
        ):
            raise ValueError(
                f"{filename!r} is not a valid Slicer 4D .seq.nrrd: "
                f"'space directions' has shape {space_directions.shape}, "
                "expected a 2-D array of at least (4, 3)"
            )
        if (
            measurement_frame.ndim != 2
            or measurement_frame.shape[0] < 3
            or measurement_frame.shape[1] < 3
        ):
            raise ValueError(
                f"{filename!r} is not a valid Slicer 4D .seq.nrrd: "
                f"'measurement frame' has shape {measurement_frame.shape}, "
                "expected a 2-D array of at least (3, 3)"
            )

        origin_3d = np.asarray(header["space origin"], dtype=float)
        spacing_3d = np.array(
            [abs(space_directions[x + 1][x]) for x in range(3)],
            dtype=float,
        )
        direction_3d = np.array([measurement_frame[x] for x in range(3)], dtype=float)
        space = header.get("space", "")
        if "right" in space:
            direction_3d[0][0] *= -1
        if "anterior" in space:
            direction_3d[1][1] *= -1
        if "inferior" in space:
            direction_3d[2][2] *= -1

        self._build_frames(arr_4d, origin_3d, spacing_3d, direction_3d)

    def _build_frames(
        self,
        arr_4d: np.ndarray,
        origin_3d: np.ndarray,
        spacing_3d: np.ndarray,
        direction_3d: np.ndarray,
    ) -> None:
        """Materialize ``self.img_3d`` from a time-series array + geometry."""
        direction_matrix = itk.matrix_from_array(np.ascontiguousarray(direction_3d))
        self.img_3d = []
        for t in range(arr_4d.shape[0]):
            # Copy so each 3D image owns its buffer independently.
            arr_3d = np.ascontiguousarray(arr_4d[t])
            img3d = itk.image_from_array(arr_3d)
            img3d.SetOrigin(origin_3d.tolist())
            img3d.SetSpacing(spacing_3d.tolist())
            img3d.SetDirection(direction_matrix)
            self.img_3d.append(img3d)

    def _load_dicom_directory(self, dirpath: Path) -> None:
        """Read a DICOM directory and build one 3D image per temporal phase.

        Files in ``dirpath`` are inspected with ``pydicom`` to identify valid
        DICOM image slices, group them by temporal phase, and sort them along
        the slice normal.  The resulting ordered filename list for each phase
        is handed to ``itk.imread``, which constructs the 3D image with proper
        origin, spacing, and direction in LPS world space via its DICOM IO.

        Slices are grouped by a composite key built from the DICOM tags
        listed in ``self.dicom_phase_keys`` (the default set covers
        ``TemporalPositionIdentifier``, ``TriggerTime``, the cardiac trigger
        delay / phase tags, ``FrameReferenceDateTime``, and ``ScanOptions``).
        Any tag whose
        value differs between slices will split them into separate phases;
        missing tags fall back to the per-tag default.  When none of the
        configured tags differ across slices, all slices form a single 3D
        volume.  Non-DICOM files and files without the geometry tags are
        skipped.

        Args:
            dirpath: Directory holding a DICOM series (3D or 4D).
        """
        entries: list[tuple[str, pydicom.Dataset]] = []
        for fp in sorted(dirpath.iterdir()):
            if not fp.is_file():
                continue
            try:
                ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=False)
            except (pydicom.errors.InvalidDicomError, OSError):
                continue
            if "ImageOrientationPatient" not in ds:
                continue
            if "ImagePositionPatient" not in ds:
                continue
            entries.append((str(fp), ds))

        if not entries:
            raise ValueError(f"No readable DICOM image slices in {dirpath}")

        self.log_info(f"Read {len(entries)} DICOM slice file(s) from {dirpath}")

        groups: dict[
            tuple[Union[float, str], ...], list[tuple[str, pydicom.Dataset]]
        ] = defaultdict(list)
        for fname, slice_ds in entries:
            key_parts: list[Union[float, str]] = []
            for tag, default in self.dicom_phase_keys.items():
                if tag not in slice_ds:
                    key_parts.append(default)
                elif isinstance(default, str):
                    key_parts.append(str(slice_ds[tag].value))
                else:
                    key_parts.append(float(slice_ds[tag].value))
            groups[tuple(key_parts)].append((fname, slice_ds))

        sorted_keys = sorted(groups.keys())
        self.log_info(f"Grouped DICOM slices into {len(sorted_keys)} phase(s)")

        self.img_3d = []
        for key in sorted_keys:
            group_entries = groups[key]
            iop = np.asarray(group_entries[0][1].ImageOrientationPatient, dtype=float)
            slice_normal = np.cross(iop[:3], iop[3:6])

            def proj(
                ds: pydicom.Dataset,
                normal: np.ndarray = slice_normal,
            ) -> float:
                ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
                return float(np.dot(ipp, normal))

            ordered = sorted(group_entries, key=lambda item: proj(item[1]))
            filenames = [fname for fname, _ in ordered]
            self.img_3d.append(itk.imread(filenames))

    def get_3d_image(self, index: int) -> itk.Image:
        """Return the 3D ITK image at the given time index."""
        return self.img_3d[index]

    def get_3d_images(self) -> list[itk.Image]:
        """Return the list of 3D ITK images."""
        return self.img_3d

    def get_number_of_3d_images(self) -> int:
        """Return the number of 3D images currently held."""
        return len(self.img_3d)

    def save_3d_images(
        self,
        directory: Union[str, Path],
        basename: str,
        suffix: str = "mha",
    ) -> None:
        """Write each held 3D image to ``{directory}/{basename}_{i:03d}.{suffix}``.

        Args:
            directory: Output directory; created if it does not exist.
            basename: Filename stem used for every saved volume.
            suffix: File extension (default: ``mha``).
        """
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        for i in range(self.get_number_of_3d_images()):
            itk.imwrite(
                self.img_3d[i],
                str(dir_path / f"{basename}_{i:03d}.{suffix}"),
                compression=True,
            )
