"""
Test for downloading and converting Slicer-Heart-CT data.

This test replicates the functionality from cells 0-2 of the notebook
Heart-GatedCT_To_USD/0-download_and_convert_4d_to_3d.ipynb.
"""

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from monai_physio.data_download_tools import DataDownloadTools


class TestDataDownloadTools:
    """Synthetic tests for dataset verification helpers."""

    def test_verify_slicer_heart_ct_data(self, tmp_path: Path) -> None:
        """Verify Slicer data by expected `TruncalValve_4DCT.seq.nrrd` filename."""
        data_file = tmp_path / DataDownloadTools.SLICER_HEART_CT_FILENAME

        assert not DataDownloadTools.VerifySlicerHeartCTData(tmp_path)

        data_file.write_bytes(b"nrrd")

        assert DataDownloadTools.VerifySlicerHeartCTData(tmp_path)

    def test_verify_kcl_heart_model_data(self, tmp_path: Path) -> None:
        """Verify KCL data by expected average mesh and input mesh filenames."""
        (tmp_path / "average_mesh.vtk").write_text("# vtk\n")
        input_meshes = tmp_path / "input_meshes"
        input_meshes.mkdir()

        assert not DataDownloadTools.VerifyKCLHeartModelData(tmp_path)

        (input_meshes / "01.vtk").write_text("# vtk\n")

        assert DataDownloadTools.VerifyKCLHeartModelData(tmp_path)

    def test_verify_dirlab_4dct_data(self, tmp_path: Path) -> None:
        """Verify DirLab data requires both the header and its backing data."""
        case1_dir = tmp_path / "Case1"
        case1_dir.mkdir()

        assert not DataDownloadTools.VerifyDirLab4DCTData(tmp_path)

        # A header alone (as committed to the repo) must not verify --
        # the raw pixel data it points to has not been downloaded yet.
        (case1_dir / "case1_T00.mhd").write_text(
            "ObjectType = Image\nElementDataFile = case1_T00.img\n"
        )
        assert not DataDownloadTools.VerifyDirLab4DCTData(tmp_path)

        # Once the backing pixel data the header points to exists, it verifies.
        (case1_dir / "case1_T00.img").write_bytes(b"raw")
        assert DataDownloadTools.VerifyDirLab4DCTData(tmp_path)

    def test_verify_dirlab_4dct_pack_layout_requires_backing_data(
        self, tmp_path: Path
    ) -> None:
        """A committed Case1Pack_T*.mhd header alone must not verify as downloaded.

        Regression test: these headers are committed to the repo to
        document the expected layout, so glob-matching the header filename
        alone would always report the dataset as present.
        """
        mhd_file = tmp_path / "Case1Pack_T00.mhd"
        mhd_file.write_text(
            "ObjectType = Image\nElementDataFile = Case1Pack/Images/case1_T00_s.img\n"
        )
        assert not DataDownloadTools.VerifyDirLab4DCTData(tmp_path)

        backing_file = tmp_path / "Case1Pack" / "Images" / "case1_T00_s.img"
        backing_file.parent.mkdir(parents=True)
        backing_file.write_bytes(b"raw")
        assert DataDownloadTools.VerifyDirLab4DCTData(tmp_path)

    def test_verify_chop_valve_4d_data(self, tmp_path: Path) -> None:
        """Verify CHOP data by expected CT or valve time-series paths."""
        assert not DataDownloadTools.VerifyCHOPValve4DData(tmp_path)

        ct_dir = tmp_path / "CT"
        ct_dir.mkdir()
        (ct_dir / "RVOT28-Dias.nii.gz").write_bytes(b"nii")

        assert DataDownloadTools.VerifyCHOPValve4DData(tmp_path)


class TestDownloadHeartData:
    """Test suite for downloading and converting Slicer-Heart-CT data."""

    def test_directories_created(self, test_directories: dict[str, Path]) -> None:
        """Test that directories are created successfully."""
        data_dir = test_directories["data"]
        slicer_heart_dir = test_directories["slicer_heart_data"]
        slicer_heart_small_dir = test_directories["slicer_heart_small_data"]
        output_dir = test_directories["output"]

        assert data_dir.exists(), f"Data directory not created: {data_dir}"
        assert slicer_heart_dir.exists(), (
            f"Slicer-Heart directory not created: {slicer_heart_dir}"
        )
        assert slicer_heart_small_dir.exists(), (
            f"Small Slicer-Heart directory not created: {slicer_heart_small_dir}"
        )
        assert output_dir.exists(), f"Output directory not created: {output_dir}"
        assert data_dir.is_dir(), f"Data path is not a directory: {data_dir}"
        assert slicer_heart_dir.is_dir(), (
            f"Slicer-Heart path is not a directory: {slicer_heart_dir}"
        )
        assert slicer_heart_small_dir.is_dir(), (
            f"Small Slicer-Heart path is not a directory: {slicer_heart_small_dir}"
        )
        assert output_dir.is_dir(), f"Output path is not a directory: {output_dir}"

    def test_data_downloaded(
        self,
        download_test_data: Path,
        test_directories: dict[str, Path],
    ) -> None:
        """Test that the TruncalValve 4D CT data file is downloaded."""
        data_file = download_test_data

        assert DataDownloadTools.VerifySlicerHeartCTData(
            test_directories["slicer_heart_data"]
        )
        assert data_file.exists(), f"Data file not found: {data_file}"
        assert data_file.is_file(), f"Data path is not a file: {data_file}"

        # Check file size is reasonable (should be > 1MB)
        file_size = data_file.stat().st_size
        assert file_size > 1_000_000, (
            f"Downloaded file seems too small: {file_size} bytes"
        )

        print(f"\nData file downloaded successfully: {data_file}")
        print(f"  File size: {file_size / 1_000_000:.2f} MB")

    def test_download_kcl_heart_model_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each per-model and average archive is downloaded and unpacked."""
        archives_dir = tmp_path / "archives"
        archives_dir.mkdir()

        def make_archive(member_name: str, content: bytes) -> Path:
            archive_path = archives_dir / f"{member_name}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                info = tarfile.TarInfo(name=member_name)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
            return archive_path

        urls_to_archives = {}
        for index in range(1, DataDownloadTools.KCL_HEART_MODEL_MESH_COUNT + 1):
            url = DataDownloadTools.KCL_HEART_MODEL_INDIVIDUAL_URL_TEMPLATE.format(
                index=index
            )
            urls_to_archives[url] = make_archive(
                f"{index:02d}.vtk", f"# vtk {index}\n".encode()
            )
        urls_to_archives[DataDownloadTools.KCL_HEART_MODEL_AVERAGE_URL] = make_archive(
            "average.vtk", b"# vtk average\n"
        )

        def fake_urlopen(url: str, timeout: float) -> object:
            return open(urls_to_archives[url], "rb")

        monkeypatch.setattr(
            "monai_physio.data_download_tools.urllib.request.urlopen", fake_urlopen
        )

        output_dir = tmp_path / "KCL-Heart-Model"
        result_dir = DataDownloadTools.DownloadKCLHeartModelData(output_dir)

        assert result_dir == output_dir
        assert (output_dir / "average_mesh.vtk").read_text() == "# vtk average\n"
        for index in range(1, DataDownloadTools.KCL_HEART_MODEL_MESH_COUNT + 1):
            mesh_file = output_dir / "input_meshes" / f"{index:02d}.vtk"
            assert mesh_file.read_text() == f"# vtk {index}\n"
        assert DataDownloadTools.VerifyKCLHeartModelData(output_dir)

    def test_download_chop_valve4d_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each subdirectory's zip archive is downloaded and extracted."""
        archives_dir = tmp_path / "archives"
        archives_dir.mkdir()

        def make_archive(subdir_name: str, member_name: str, content: bytes) -> Path:
            archive_path = archives_dir / f"{subdir_name}.zip"
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr(member_name, content)
            return archive_path

        urls_to_archives = {}
        for subdir_name, asset_name in DataDownloadTools.CHOP_VALVE4D_ASSETS.items():
            url = DataDownloadTools.CHOP_VALVE4D_RELEASE_URL + asset_name
            urls_to_archives[url] = make_archive(
                subdir_name,
                f"{subdir_name}/{subdir_name}.txt",
                f"# {subdir_name}\n".encode(),
            )

        def fake_urlopen(url: str, timeout: float) -> object:
            return open(urls_to_archives[url], "rb")

        monkeypatch.setattr(
            "monai_physio.data_download_tools.urllib.request.urlopen", fake_urlopen
        )

        output_dir = tmp_path / "CHOP-Valve4D"
        result_dir = DataDownloadTools.DownloadCHOPValve4DData(output_dir)

        assert result_dir == output_dir
        for subdir_name in DataDownloadTools.CHOP_VALVE4D_ASSETS:
            extracted_file = output_dir / subdir_name / f"{subdir_name}.txt"
            assert extracted_file.read_text() == f"# {subdir_name}\n"

    @staticmethod
    def _write_chop_valve4d_expected_marker(target_dir: Path, subdir_name: str) -> None:
        """Create the marker file/dir _CHOPValve4DSubdirIsPopulated looks for."""
        target_dir.mkdir(parents=True, exist_ok=True)
        if subdir_name == "CT":
            (target_dir / "RVOT28-Dias.mha").write_bytes(b"mha")
        else:
            (target_dir / "frame_0000.vtk").write_text("# vtk\n")

    def test_download_chop_valve4d_data_skips_populated_subdirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Subdirectories with their expected marker files are not re-downloaded."""
        output_dir = tmp_path / "CHOP-Valve4D"
        for subdir_name in DataDownloadTools.CHOP_VALVE4D_ASSETS:
            self._write_chop_valve4d_expected_marker(
                output_dir / subdir_name, subdir_name
            )

        def fake_urlopen(url: str, timeout: float) -> object:
            raise AssertionError(f"Should not download populated subdir: {url}")

        monkeypatch.setattr(
            "monai_physio.data_download_tools.urllib.request.urlopen", fake_urlopen
        )

        result_dir = DataDownloadTools.DownloadCHOPValve4DData(output_dir)

        assert result_dir == output_dir

    def test_download_chop_valve4d_data_redownloads_partial_extraction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A subdir left behind by an interrupted extraction is re-downloaded.

        Regression test: an earlier version skipped any subdirectory that
        merely had *some* file in it, so an interrupted extraction that only
        wrote a partial/temp file would be treated as complete forever.
        """
        output_dir = tmp_path / "CHOP-Valve4D"
        partial_dir = output_dir / "Alterra"
        partial_dir.mkdir(parents=True)
        (partial_dir / "partial_download.tmp").write_text("incomplete\n")

        archives_dir = tmp_path / "archives"
        archives_dir.mkdir()

        def make_archive(subdir_name: str, member_name: str, content: bytes) -> Path:
            archive_path = archives_dir / f"{subdir_name}.zip"
            with zipfile.ZipFile(archive_path, "w") as zf:
                zf.writestr(member_name, content)
            return archive_path

        urls_to_archives = {}
        for subdir_name, asset_name in DataDownloadTools.CHOP_VALVE4D_ASSETS.items():
            url = DataDownloadTools.CHOP_VALVE4D_RELEASE_URL + asset_name
            leaf = "RVOT28-Dias.mha" if subdir_name == "CT" else "frame_0000.vtk"
            urls_to_archives[url] = make_archive(
                subdir_name, f"{subdir_name}/{leaf}", f"# {subdir_name}\n".encode()
            )

        def fake_urlopen(url: str, timeout: float) -> object:
            return open(urls_to_archives[url], "rb")

        monkeypatch.setattr(
            "monai_physio.data_download_tools.urllib.request.urlopen", fake_urlopen
        )

        DataDownloadTools.DownloadCHOPValve4DData(output_dir)

        # The stray leftover file did not prevent Alterra from re-downloading.
        assert (partial_dir / "frame_0000.vtk").read_text() == "# Alterra\n"
        assert (partial_dir / "partial_download.tmp").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
