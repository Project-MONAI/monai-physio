"""
Dataset download and verification helpers.

Slicer-Heart-CT, KCL-Heart-Model, CHOP-Valve4D, and Chest-CT are downloaded
automatically. Other datasets require manual download, and the verification
helpers check the file layouts used by the repository tutorials,
experiments, and tests.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Union

import itk
import numpy as np

from .convert_image_4d_to_3d import ConvertImage4DTo3D

_DOWNLOAD_TIMEOUT_SECONDS = 60.0
_logger = logging.getLogger(__name__)


class DataDownloadTools:
    """Download and verify optional MONAI Physio example datasets."""

    SLICER_HEART_CT_URL = (
        "https://github.com/Project-MONAI/monai-physio/releases/download/"
        "2026.07.1/TruncalValve_4DCT.seq.nrrd"
    )
    SLICER_HEART_CT_FILENAME = "TruncalValve_4DCT.seq.nrrd"

    SLICER_HEART_CT_SLICE_BASENAME = "slice"

    @staticmethod
    def DownloadSlicerHeartCTData(dirname: Union[str, Path]) -> Path:  # noqa: N802
        """Download the Slicer-Heart-CT 4-D CT sample into ``dirname``.

        Also splits the downloaded 4-D sequence into per-frame
        ``slice_???.mha`` volumes in the same directory (via
        ``ConvertImage4DTo3D``), skipping the split if those files already
        exist.

        Args:
            dirname: Directory where ``TruncalValve_4DCT.seq.nrrd`` and the
                converted ``slice_???.mha`` files should live.

        Returns:
            Path to the downloaded or already-cached ``.seq.nrrd`` file.
        """
        data_dir = Path(dirname)
        data_dir.mkdir(parents=True, exist_ok=True)

        data_file = data_dir / DataDownloadTools.SLICER_HEART_CT_FILENAME
        if not (data_file.exists() and data_file.stat().st_size > 0):
            DataDownloadTools._DownloadFile(
                DataDownloadTools.SLICER_HEART_CT_URL, data_file
            )

        slice_basename = DataDownloadTools.SLICER_HEART_CT_SLICE_BASENAME
        if not any(data_dir.glob(f"{slice_basename}_???.mha")):
            # Convert into a temp directory first, then atomically move each
            # finished frame into data_dir, so a failed or interrupted
            # conversion leaves no partial slice_???.mha files behind and
            # concurrent callers never observe a half-written frame.
            with tempfile.TemporaryDirectory(dir=str(data_dir)) as tmp_dir_name:
                tmp_dir = Path(tmp_dir_name)
                conv = ConvertImage4DTo3D()
                conv.load_image_4d(str(data_file))
                conv.save_3d_images(tmp_dir, slice_basename)
                for tmp_slice_file in sorted(tmp_dir.glob(f"{slice_basename}_???.mha")):
                    tmp_slice_file.replace(data_dir / tmp_slice_file.name)
            _logger.info(
                "Converted %s to %s_???.mha frames",
                DataDownloadTools.SLICER_HEART_CT_FILENAME,
                slice_basename,
            )
        return data_file

    @staticmethod
    def _DownloadFile(url: str, target_file: Path) -> None:  # noqa: N802
        """Stream-download ``url`` and atomically replace ``target_file``.

        Streams to a unique temp file in the target's directory with an
        explicit timeout, then atomically replaces the target on success. The
        temp name is unique so concurrent callers do not clobber each other.
        Avoids partial files on interrupt and the indefinite hang that
        urlretrieve has without a timeout.
        """
        tmp_handle = tempfile.NamedTemporaryFile(
            dir=str(target_file.parent),
            prefix=f".{target_file.name}.",
            suffix=".tmp",
            delete=False,
        )
        tmp_file = Path(tmp_handle.name)
        try:
            with (
                urllib.request.urlopen(  # noqa: S310
                    url, timeout=_DOWNLOAD_TIMEOUT_SECONDS
                ) as response,
                tmp_handle as out,
            ):
                shutil.copyfileobj(response, out)
            if tmp_file.stat().st_size == 0:
                raise RuntimeError(f"Downloaded file is empty: {url}")
            tmp_file.replace(target_file)
            _logger.info("Downloaded %s", target_file.name)
        except BaseException:
            tmp_handle.close()
            if tmp_file.exists():
                tmp_file.unlink()
            raise

    @staticmethod
    def VerifySlicerHeartCTData(dirname: Union[str, Path]) -> bool:  # noqa: N802
        """Return True when Slicer-Heart-CT has the expected 4-D CT file."""
        return (Path(dirname) / DataDownloadTools.SLICER_HEART_CT_FILENAME).is_file()

    KCL_HEART_MODEL_MESH_COUNT = 20
    KCL_HEART_MODEL_INDIVIDUAL_URL_TEMPLATE = (
        "https://zenodo.org/records/4590294/files/{index:02d}.tar.gz?download=1"
    )
    KCL_HEART_MODEL_AVERAGE_URL = (
        "https://zenodo.org/records/4593739/files/average.tar.gz?download=1"
    )

    @staticmethod
    def DownloadKCLHeartModelData(dirname: Union[str, Path]) -> Path:  # noqa: N802
        """Download the KCL-Heart-Model dataset into ``dirname``.

        Downloads and extracts the 20 individual four-chamber heart meshes
        from https://zenodo.org/records/4590294 into
        ``dirname/input_meshes/01.vtk`` through ``20.vtk``, and the average
        heart mesh from https://zenodo.org/records/4593739 into
        ``dirname/average_mesh.vtk``. Already-extracted files are reused.

        Args:
            dirname: Directory where the KCL-Heart-Model dataset should live.

        Returns:
            Path to ``dirname``.
        """
        data_dir = Path(dirname)
        input_meshes_dir = data_dir / "input_meshes"
        input_meshes_dir.mkdir(parents=True, exist_ok=True)

        for index in range(1, DataDownloadTools.KCL_HEART_MODEL_MESH_COUNT + 1):
            target_file = input_meshes_dir / f"{index:02d}.vtk"
            if target_file.exists() and target_file.stat().st_size > 0:
                continue
            url = DataDownloadTools.KCL_HEART_MODEL_INDIVIDUAL_URL_TEMPLATE.format(
                index=index
            )
            DataDownloadTools._DownloadAndExtractTarMember(
                url, member_name=f"{index:02d}.vtk", target_file=target_file
            )
            _logger.info(
                "Downloaded %02d.vtk (%d/%d)",
                index,
                index,
                DataDownloadTools.KCL_HEART_MODEL_MESH_COUNT,
            )

        average_file = data_dir / "average_mesh.vtk"
        if not (average_file.exists() and average_file.stat().st_size > 0):
            DataDownloadTools._DownloadAndExtractTarMember(
                DataDownloadTools.KCL_HEART_MODEL_AVERAGE_URL,
                member_name="average.vtk",
                target_file=average_file,
            )
            _logger.info("Downloaded average_mesh.vtk")

        return data_dir

    @staticmethod
    def _DownloadAndExtractTarMember(  # noqa: N802
        url: str, member_name: str, target_file: Path
    ) -> None:
        """Download a ``.tar.gz`` archive and extract one member to ``target_file``."""
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(target_file.parent)) as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            archive_file = tmp_dir / "archive.tar.gz"
            with (
                urllib.request.urlopen(  # noqa: S310
                    url, timeout=_DOWNLOAD_TIMEOUT_SECONDS
                ) as response,
                open(archive_file, "wb") as out,
            ):
                shutil.copyfileobj(response, out)
            if archive_file.stat().st_size == 0:
                raise RuntimeError(f"Downloaded archive is empty: {url}")

            with tarfile.open(archive_file) as tar:
                try:
                    tar.extract(member_name, path=tmp_dir, filter="data")
                except TypeError:
                    # Python < 3.12 (without the PEP 706 backport) does not
                    # accept the ``filter`` keyword argument.
                    tar.extract(member_name, path=tmp_dir)
            extracted_file = tmp_dir / member_name
            if not extracted_file.is_file():
                raise RuntimeError(
                    f"Expected member {member_name!r} not found in archive: {url}"
                )
            extracted_file.replace(target_file)

    CHOP_VALVE4D_RELEASE_URL = (
        "https://github.com/Project-MONAI/monai-physio/releases/download/2026.07.1/"
    )
    # subdirectory name -> release asset filename. Alterra and TPV25 are
    # each >1 GB (per-frame valve mesh time series); CT is the smaller
    # image/segmentation bundle. See data/CHOP-Valve4D/README.md.
    CHOP_VALVE4D_ASSETS = {
        "Alterra": "CHOP-Valve4D-Alterra.zip",
        "CT": "CHOP-Valve4D-CT.zip",
        "TPV25": "CHOP-Valve4D-TPV25.zip",
    }

    @staticmethod
    def DownloadCHOPValve4DData(dirname: Union[str, Path]) -> Path:  # noqa: N802
        """Download the CHOP-Valve4D convenience release into ``dirname``.

        Downloads the three zip archives attached to the MONAI Physio
        2026.07.1 GitHub release
        (https://github.com/Project-MONAI/monai-physio/releases/tag/2026.07.1)
        and extracts each into its matching subdirectory: ``Alterra/`` and
        ``TPV25/`` (valve mesh time series, >1 GB each) and ``CT/`` (source
        CT volume and Simpleware segmentation). A subdirectory that already
        has its expected files (see ``_CHOPValve4DSubdirIsPopulated``) is
        left alone, so re-running resumes an interrupted download without
        re-fetching a subdirectory a partial extraction left behind. See
        ``data/CHOP-Valve4D/README.md`` for what this converted data
        contains and how it relates to the original FEBio source model.

        Args:
            dirname: Directory where the CHOP-Valve4D dataset should live.

        Returns:
            Path to ``dirname``.
        """
        data_dir = Path(dirname)
        for subdir_name, asset_name in DataDownloadTools.CHOP_VALVE4D_ASSETS.items():
            target_dir = data_dir / subdir_name
            if DataDownloadTools._CHOPValve4DSubdirIsPopulated(target_dir):
                continue
            url = DataDownloadTools.CHOP_VALVE4D_RELEASE_URL + asset_name
            DataDownloadTools._DownloadAndExtractZip(url, target_dir)
            _logger.info("Downloaded %s (%s)", subdir_name, asset_name)
        return data_dir

    @staticmethod
    def _CHOPValve4DSubdirIsPopulated(  # noqa: N802
        target_dir: Path,
    ) -> bool:
        """Return True when ``target_dir`` already has subdir_name's expected files.

        Used to decide whether a subdirectory can be skipped on a re-run.
        Checking for *any* file is not enough — an interrupted extraction
        can leave a partially-populated directory that would then be
        skipped forever — so this checks for the specific files each
        subdirectory is expected to contain once fully extracted.
        """
        if not target_dir.is_dir():
            return False
        subdir_name = target_dir.name
        if subdir_name == "CT":
            has_ct_volume = any(
                (target_dir / filename).is_file()
                for filename in ("RVOT28-Dias.nii.gz", "RVOT28-Dias.mha")
            )
            has_simpleware_parts = (target_dir / "Simpleware" / "parts").is_dir()
            return has_ct_volume or has_simpleware_parts
        # Alterra and TPV25 are valve mesh time series.
        return any(target_dir.glob("*.vtk"))

    @staticmethod
    def _DownloadAndExtractZip(url: str, target_dir: Path) -> None:  # noqa: N802
        """Stream-download a ``.zip`` archive and extract it into ``target_dir``."""
        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(target_dir.parent)) as tmp_dir_name:
            archive_file = Path(tmp_dir_name) / "archive.zip"
            with (
                urllib.request.urlopen(  # noqa: S310
                    url, timeout=_DOWNLOAD_TIMEOUT_SECONDS
                ) as response,
                open(archive_file, "wb") as out,
            ):
                shutil.copyfileobj(response, out, length=1024 * 1024)
            if archive_file.stat().st_size == 0:
                raise RuntimeError(f"Downloaded archive is empty: {url}")

            with zipfile.ZipFile(archive_file) as archive:
                archive.extractall(target_dir.parent)

    @staticmethod
    def VerifyCHOPValve4DData(dirname: Union[str, Path]) -> bool:  # noqa: N802
        """Return True when CHOP-Valve4D files referenced by the repo exist.

        Accepted layouts are the CT volume used by Simpleware/model-to-patient
        experiments and the valve time-series folders used by VTK-to-USD
        experiments.
        """
        data_dir = Path(dirname)
        has_ct = DataDownloadTools._CHOPValve4DSubdirIsPopulated(data_dir / "CT")
        has_alterra = DataDownloadTools._CHOPValve4DSubdirIsPopulated(
            data_dir / "Alterra"
        )
        has_tpv25 = DataDownloadTools._CHOPValve4DSubdirIsPopulated(data_dir / "TPV25")
        return has_ct or (has_alterra and has_tpv25)

    CHEST_CT_URL = (
        "https://github.com/Project-MONAI/monai-physio/releases/download/2026.07.1/"
        "Chest-CT.mha"
    )
    CHEST_CT_FILENAME = "Chest-CT.mha"

    @staticmethod
    def DownloadChestCTData(dirname: Union[str, Path]) -> Path:  # noqa: N802
        """Download the Chest-CT sample volume into ``dirname``.

        Fetches ``Chest-CT.mha`` — an ungated 3-D chest CT — from the
        MONAI Physio 2026.07.1 GitHub release. An existing non-empty file is
        reused, so re-running resumes an interrupted download. See
        ``data/Chest-CT/README.md`` for the data source and required citation.

        Args:
            dirname: Directory where ``Chest-CT.mha`` should live.

        Returns:
            Path to the downloaded or already-cached ``Chest-CT.mha`` file.
        """
        data_dir = Path(dirname)
        data_dir.mkdir(parents=True, exist_ok=True)

        data_file = data_dir / DataDownloadTools.CHEST_CT_FILENAME
        if not (data_file.exists() and data_file.stat().st_size > 0):
            DataDownloadTools._DownloadFile(DataDownloadTools.CHEST_CT_URL, data_file)
        return data_file

    @staticmethod
    def VerifyChestCTData(dirname: Union[str, Path]) -> bool:  # noqa: N802
        """Return True when Chest-CT has its expected CT volume."""
        return (Path(dirname) / DataDownloadTools.CHEST_CT_FILENAME).is_file()

    @staticmethod
    def _MetaImageHeaderHasBackingData(mhd_file: Path) -> bool:  # noqa: N802
        """Return True when a MetaImage ``.mhd`` header's pixel data exists.

        Committed ``.mhd`` headers are tiny text files (a few hundred
        bytes) that ship with the repository as documentation of the
        expected DirLab-4DCT layout; their presence alone does not mean
        the (large, gitignored) raw pixel data has been downloaded. This
        checks that the file the header's ``ElementDataFile`` line points
        to actually exists, resolved relative to the header's directory.
        """
        try:
            lines = mhd_file.read_text(errors="ignore").splitlines()
        except OSError:
            return False
        for line in lines:
            key, sep, value = line.partition("=")
            if sep and key.strip() == "ElementDataFile":
                data_file = value.strip()
                if not data_file or data_file.upper() == "LOCAL":
                    # Pixel data is embedded in this same file.
                    return True
                return (mhd_file.parent / data_file).is_file()
        return False

    @staticmethod
    def VerifyDirLab4DCTData(dirname: Union[str, Path]) -> bool:  # noqa: N802
        """Return True when a supported DirLab-4DCT case layout exists."""
        data_dir = Path(dirname)
        case1_dir = data_dir / "Case1"
        has_case_dir_layout = case1_dir.is_dir() and any(case1_dir.glob("*.mha"))
        has_case_dir_layout = has_case_dir_layout or (
            case1_dir.is_dir()
            and any(
                DataDownloadTools._MetaImageHeaderHasBackingData(mhd_file)
                for mhd_file in case1_dir.glob("*.mhd")
            )
        )

        has_pack_layout = any(data_dir.glob("Case1Pack_T*.mha")) or any(
            DataDownloadTools._MetaImageHeaderHasBackingData(mhd_file)
            for mhd_file in data_dir.glob("Case1Pack_T*.mhd")
        )
        return has_case_dir_layout or has_pack_layout

    DIRLAB_4DCT_HU_OFFSET = 1024
    DIRLAB_4DCT_HU_CLIP_RANGE = (-1024, 1024)

    @staticmethod
    def FixDirLab4DCTData(  # noqa: N802
        dirname: Union[str, Path],
        output_dirname: Optional[Union[str, Path]] = None,
    ) -> list[Path]:
        """Convert DirLab-4DCT raw intensities to clipped Hounsfield units.

        DirLab-4DCT's raw ``.img``/``.mhd`` volumes store intensities offset
        by +1024 from Hounsfield units. This finds every ``.mhd`` header
        under ``dirname`` (recursively) whose raw data has been downloaded,
        subtracts 1024, clips the result to ``DIRLAB_4DCT_HU_CLIP_RANGE``,
        and writes the corrected volume with the same base filename but a
        ``.mha`` extension, compressed.

        Args:
            dirname: Directory to search recursively for ``.mhd`` headers.
            output_dirname: Directory to write the ``.mha`` files into.
                Defaults to writing each result next to its source ``.mhd``
                header.

        Returns:
            Paths to the written ``.mha`` files.
        """
        output_dir = Path(output_dirname) if output_dirname is not None else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        output_files = []
        for mhd_file in sorted(Path(dirname).rglob("*.mhd")):
            if not DataDownloadTools._MetaImageHeaderHasBackingData(mhd_file):
                continue
            image = itk.imread(str(mhd_file))
            image_arr = (
                itk.array_from_image(image) - DataDownloadTools.DIRLAB_4DCT_HU_OFFSET
            )
            image_arr = np.clip(image_arr, *DataDownloadTools.DIRLAB_4DCT_HU_CLIP_RANGE)
            fixed_image = itk.image_from_array(image_arr)
            fixed_image.CopyInformation(image)
            target_dir = output_dir if output_dir is not None else mhd_file.parent
            output_file = target_dir / mhd_file.with_suffix(".mha").name
            itk.imwrite(fixed_image, str(output_file), compression=True)
            output_files.append(output_file)
            _logger.info("Fixed %s -> %s", mhd_file.name, output_file.name)
        return output_files

    @staticmethod
    def VerifyKCLHeartModelData(dirname: Union[str, Path]) -> bool:  # noqa: N802
        """Return True when KCL-Heart-Model has its expected mesh inputs."""
        data_dir = Path(dirname)
        input_meshes_dir = data_dir / "input_meshes"
        return (data_dir / "average_mesh.vtk").is_file() and any(
            input_meshes_dir.glob("*.vtk")
        )
