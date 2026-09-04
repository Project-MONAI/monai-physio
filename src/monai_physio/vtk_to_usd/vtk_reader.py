"""VTK file readers for various VTK formats (VTK, VTP, VTU).

Reads VTK files and extracts geometry, topology, and data arrays.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import vtk
from numpy.typing import NDArray
from vtk.util import numpy_support

from .data_structures import DataType, GenericArray, MeshData

logger = logging.getLogger(__name__)


class VTKReader:
    """Base class for VTK file readers."""

    @staticmethod
    def _vtk_to_numpy_type(vtk_type: int) -> DataType:
        """Convert VTK data type to our DataType enum."""
        type_map = {
            vtk.VTK_UNSIGNED_CHAR: DataType.UCHAR,
            vtk.VTK_CHAR: DataType.CHAR,
            vtk.VTK_UNSIGNED_SHORT: DataType.USHORT,
            vtk.VTK_SHORT: DataType.SHORT,
            vtk.VTK_UNSIGNED_INT: DataType.UINT,
            vtk.VTK_INT: DataType.INT,
            vtk.VTK_UNSIGNED_LONG: DataType.ULONG,
            vtk.VTK_LONG: DataType.LONG,
            vtk.VTK_FLOAT: DataType.FLOAT,
            vtk.VTK_DOUBLE: DataType.DOUBLE,
        }
        return type_map.get(vtk_type, DataType.FLOAT)

    @staticmethod
    def _extract_point_data_arrays(polydata: vtk.vtkPolyData) -> list[GenericArray]:
        """Extract all point data arrays as GenericArray objects."""
        arrays = []
        point_data = polydata.GetPointData()
        num_arrays = point_data.GetNumberOfArrays()

        # Get expected number of points for validation
        num_points = polydata.GetNumberOfPoints()

        for i in range(num_arrays):
            vtk_array = point_data.GetArray(i)
            name = vtk_array.GetName()
            if name is None or name == "":
                continue

            # Convert to numpy
            numpy_array = numpy_support.vtk_to_numpy(vtk_array)
            num_components = vtk_array.GetNumberOfComponents()

            # Determine data type
            vtk_type = vtk_array.GetDataType()
            data_type = VTKReader._vtk_to_numpy_type(vtk_type)

            # Reshape if multi-component
            if num_components > 1 and numpy_array.ndim == 1:
                numpy_array = numpy_array.reshape(-1, num_components)

            # Validate array size matches number of points
            array_size = len(numpy_array)
            if array_size != num_points:
                logger.warning(
                    f"Point data array '{name}' size mismatch: "
                    f"got {array_size} values, expected {num_points} points. "
                    f"This array will be skipped to avoid USD corruption."
                )
                continue  # Skip this array

            arrays.append(
                GenericArray(
                    name=name,
                    data=numpy_array,
                    num_components=num_components,
                    data_type=data_type,
                    interpolation="vertex",
                )
            )

        return arrays

    @staticmethod
    def _extract_cell_data_arrays(polydata: vtk.vtkPolyData) -> list[GenericArray]:
        """Extract all cell data arrays as GenericArray objects."""
        arrays = []
        cell_data = polydata.GetCellData()
        num_arrays = cell_data.GetNumberOfArrays()

        # Get expected number of cells for validation
        num_cells = polydata.GetNumberOfCells()

        for i in range(num_arrays):
            vtk_array = cell_data.GetArray(i)
            name = vtk_array.GetName()
            if name is None or name == "":
                continue

            # Convert to numpy
            numpy_array = numpy_support.vtk_to_numpy(vtk_array)
            num_components = vtk_array.GetNumberOfComponents()

            # Determine data type
            vtk_type = vtk_array.GetDataType()
            data_type = VTKReader._vtk_to_numpy_type(vtk_type)

            # Reshape if multi-component
            if num_components > 1 and numpy_array.ndim == 1:
                numpy_array = numpy_array.reshape(-1, num_components)

            # Validate array size matches number of cells
            array_size = len(numpy_array)
            if array_size != num_cells:
                logger.warning(
                    f"Cell data array '{name}' size mismatch: "
                    f"got {array_size} values, expected {num_cells} cells. "
                    f"This array will be skipped to avoid USD corruption."
                )
                continue  # Skip this array

            arrays.append(
                GenericArray(
                    name=name,
                    data=numpy_array,
                    num_components=num_components,
                    data_type=data_type,
                    interpolation="uniform",  # Cell data -> uniform interpolation
                )
            )

        return arrays

    @staticmethod
    def _extract_geometry_from_polydata(polydata: vtk.vtkPolyData) -> tuple:
        """Extract points, face counts, and face indices from vtkPolyData."""
        # Get points
        vtk_points = polydata.GetPoints()
        if vtk_points is None:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )
        num_points = vtk_points.GetNumberOfPoints()
        points: NDArray[np.float64]
        if num_points == 0:
            points = np.empty((0, 3), dtype=np.float64)
        else:
            points = np.asarray(
                [vtk_points.GetPoint(i) for i in range(num_points)],
                dtype=np.float64,
            ).reshape(-1, 3)

        # Get cells (faces)
        polys = polydata.GetPolys()
        num_polys = polys.GetNumberOfCells()

        face_vertex_counts = []
        face_vertex_indices = []

        polys.InitTraversal()
        id_list = vtk.vtkIdList()
        for _ in range(num_polys):
            polys.GetNextCell(id_list)
            num_pts = id_list.GetNumberOfIds()
            face_vertex_counts.append(num_pts)
            face_vertex_indices.extend([id_list.GetId(j) for j in range(num_pts)])

        return (
            points,
            np.array(face_vertex_counts, dtype=np.int32),
            np.array(face_vertex_indices, dtype=np.int32),
        )

    @staticmethod
    def _extract_normals(polydata: vtk.vtkPolyData) -> Optional[NDArray]:
        """Extract normals if they exist, or compute them."""
        from typing import cast

        # Check if normals exist in point data
        point_data = polydata.GetPointData()
        normals_array = point_data.GetNormals()

        if normals_array is not None:
            return cast(NDArray, numpy_support.vtk_to_numpy(normals_array))

        # Compute normals if they don't exist
        normal_generator = vtk.vtkPolyDataNormals()
        normal_generator.SetInputData(polydata)
        normal_generator.ComputePointNormalsOn()
        normal_generator.ComputeCellNormalsOff()
        normal_generator.SplittingOff()
        normal_generator.ConsistencyOn()
        normal_generator.AutoOrientNormalsOn()
        normal_generator.Update()

        output = normal_generator.GetOutput()
        normals_array = output.GetPointData().GetNormals()

        if normals_array is not None:
            return cast(NDArray, numpy_support.vtk_to_numpy(normals_array))

        return None

    @staticmethod
    def _extract_colors(polydata: vtk.vtkPolyData) -> Optional[NDArray]:
        """Extract vertex colors if they exist."""
        from typing import cast

        point_data = polydata.GetPointData()
        scalars = point_data.GetScalars()

        if scalars is not None:
            colors = cast(NDArray, numpy_support.vtk_to_numpy(scalars))
            num_components = scalars.GetNumberOfComponents()

            # Handle different color formats
            if num_components == 3:  # RGB
                return cast(NDArray, colors.reshape(-1, 3))
            elif num_components == 4:  # RGBA
                return cast(NDArray, colors.reshape(-1, 4))
            elif num_components == 1:  # Scalar - could be mapped to color later
                return None

        return None


class PolyDataReader(VTKReader):
    """Reader for VTK PolyData files (.vtp)."""

    @staticmethod
    def read(filename: str | Path) -> MeshData:
        """Read a VTP file and return MeshData.

        Args:
            filename: Path to .vtp file

        Returns:
            MeshData: Extracted mesh data
        """
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        logger.info(f"Reading PolyData file: {filename}")

        # Read VTP file
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(filename))
        reader.Update()
        polydata = reader.GetOutput()

        # Extract geometry
        points, face_counts, face_indices = VTKReader._extract_geometry_from_polydata(
            polydata
        )

        # Extract normals
        normals = VTKReader._extract_normals(polydata)

        # Extract colors
        colors = VTKReader._extract_colors(polydata)

        # Extract point and cell data arrays
        point_arrays = VTKReader._extract_point_data_arrays(polydata)
        cell_arrays = VTKReader._extract_cell_data_arrays(polydata)

        # Combine arrays
        generic_arrays = point_arrays + cell_arrays

        mesh_data = MeshData(
            points=points,
            face_vertex_counts=face_counts,
            face_vertex_indices=face_indices,
            normals=normals,
            colors=colors,
            generic_arrays=generic_arrays,
        )

        logger.info(
            f"Loaded mesh: {len(points)} points, {len(face_counts)} faces, "
            f"{len(generic_arrays)} data arrays"
        )

        return mesh_data


class LegacyVTKReader(VTKReader):
    """Reader for legacy VTK files (.vtk).

    Handles all legacy VTK dataset types:
    - POLYDATA
    - UNSTRUCTURED_GRID
    - STRUCTURED_GRID
    - STRUCTURED_POINTS
    - RECTILINEAR_GRID
    """

    @staticmethod
    def read(filename: str | Path, extract_surface: bool = True) -> MeshData:
        """Read a legacy VTK file and return MeshData.

        Args:
            filename: Path to .vtk file
            extract_surface: If True, extract surface from volumetric data

        Returns:
            MeshData: Extracted mesh data
        """
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        logger.info(f"Reading legacy VTK file: {filename}")

        # Use generic reader to auto-detect dataset type
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(str(filename))
        reader.Update()

        output = reader.GetOutput()

        if output is None:
            raise ValueError(f"Failed to read VTK file: {filename}")

        # Check what type of dataset we got
        if reader.IsFilePolyData():
            logger.debug("Detected POLYDATA format")
            polydata = reader.GetPolyDataOutput()
        elif reader.IsFileUnstructuredGrid():
            logger.debug("Detected UNSTRUCTURED_GRID format")
            ugrid = reader.GetUnstructuredGridOutput()
            if extract_surface:
                # Convert cell data to point data before surface extraction
                # This preserves cell-based arrays (like stress, strain) as point data
                cell_to_point = vtk.vtkCellDataToPointData()
                cell_to_point.SetInputData(ugrid)
                cell_to_point.PassCellDataOn()  # Keep cell data temporarily
                cell_to_point.Update()

                # Extract surface
                surface_filter = vtk.vtkDataSetSurfaceFilter()
                surface_filter.SetInputConnection(cell_to_point.GetOutputPort())
                surface_filter.Update()
                polydata = surface_filter.GetOutput()

                # Clear cell data after surface extraction
                # (Cell data from volume is invalid for surface topology)
                polydata.GetCellData().Initialize()

                logger.debug(
                    "Extracted surface from UnstructuredGrid (with cell->point data conversion)"
                )
            else:
                raise ValueError(
                    "UnstructuredGrid without surface extraction not supported. "
                    "Set extract_surface=True"
                )
        elif reader.IsFileStructuredGrid():
            logger.debug("Detected STRUCTURED_GRID format")
            sgrid = reader.GetStructuredGridOutput()
            # Convert cell data to point data before surface extraction
            cell_to_point = vtk.vtkCellDataToPointData()
            cell_to_point.SetInputData(sgrid)
            cell_to_point.PassCellDataOn()  # Keep cell data temporarily
            cell_to_point.Update()
            # Extract surface from structured grid
            surface_filter = vtk.vtkDataSetSurfaceFilter()
            surface_filter.SetInputConnection(cell_to_point.GetOutputPort())
            surface_filter.Update()
            polydata = surface_filter.GetOutput()

            # Clear cell data after surface extraction
            polydata.GetCellData().Initialize()

            logger.debug(
                "Extracted surface from StructuredGrid (with cell->point data conversion)"
            )
        elif reader.IsFileStructuredPoints():
            logger.debug("Detected STRUCTURED_POINTS format")
            spoints = reader.GetStructuredPointsOutput()
            # Convert cell data to point data before surface extraction
            cell_to_point = vtk.vtkCellDataToPointData()
            cell_to_point.SetInputData(spoints)
            cell_to_point.PassCellDataOn()  # Keep cell data temporarily
            cell_to_point.Update()
            # Extract surface from structured points (image data)
            surface_filter = vtk.vtkDataSetSurfaceFilter()
            surface_filter.SetInputConnection(cell_to_point.GetOutputPort())
            surface_filter.Update()
            polydata = surface_filter.GetOutput()

            # Clear cell data after surface extraction
            polydata.GetCellData().Initialize()

            logger.debug(
                "Extracted surface from StructuredPoints (with cell->point data conversion)"
            )
        elif reader.IsFileRectilinearGrid():
            logger.debug("Detected RECTILINEAR_GRID format")
            rgrid = reader.GetRectilinearGridOutput()
            # Convert cell data to point data before surface extraction
            cell_to_point = vtk.vtkCellDataToPointData()
            cell_to_point.SetInputData(rgrid)
            cell_to_point.PassCellDataOn()  # Keep cell data temporarily
            cell_to_point.Update()
            # Extract surface from rectilinear grid
            surface_filter = vtk.vtkDataSetSurfaceFilter()
            surface_filter.SetInputConnection(cell_to_point.GetOutputPort())
            surface_filter.Update()
            polydata = surface_filter.GetOutput()

            # Clear cell data after surface extraction
            polydata.GetCellData().Initialize()

            logger.debug(
                "Extracted surface from RectilinearGrid (with cell->point data conversion)"
            )
        else:
            raise ValueError(f"Unsupported VTK dataset type in file: {filename}")

        # Verify we have valid polydata
        if polydata is None or polydata.GetPoints() is None:
            raise ValueError(f"No valid geometry found in file: {filename}")

        # Extract geometry
        points, face_counts, face_indices = VTKReader._extract_geometry_from_polydata(
            polydata
        )

        # Extract normals
        normals = VTKReader._extract_normals(polydata)

        # Extract colors
        colors = VTKReader._extract_colors(polydata)

        # Extract point and cell data arrays
        point_arrays = VTKReader._extract_point_data_arrays(polydata)
        cell_arrays = VTKReader._extract_cell_data_arrays(polydata)

        # Combine arrays
        generic_arrays = point_arrays + cell_arrays

        mesh_data = MeshData(
            points=points,
            face_vertex_counts=face_counts,
            face_vertex_indices=face_indices,
            normals=normals,
            colors=colors,
            generic_arrays=generic_arrays,
        )

        logger.info(
            f"Loaded mesh: {len(points)} points, {len(face_counts)} faces, "
            f"{len(generic_arrays)} data arrays"
        )

        return mesh_data


class UnstructuredGridReader(VTKReader):
    """Reader for VTK UnstructuredGrid files (.vtu)."""

    @staticmethod
    def read(filename: str | Path, extract_surface: bool = True) -> MeshData:
        """Read a VTU file and return MeshData.

        Args:
            filename: Path to .vtu file
            extract_surface: If True, extract surface as PolyData

        Returns:
            MeshData: Extracted mesh data
        """
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"File not found: {filename}")

        logger.info(f"Reading UnstructuredGrid file: {filename}")

        # Read VTU file
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(filename))
        reader.Update()
        ugrid = reader.GetOutput()

        if extract_surface:
            # Convert cell data to point data before surface extraction
            # This preserves cell-based arrays (like stress, strain) as point data
            cell_to_point = vtk.vtkCellDataToPointData()
            cell_to_point.SetInputData(ugrid)
            cell_to_point.PassCellDataOn()  # Keep cell data temporarily
            cell_to_point.Update()

            # Extract surface
            surface_filter = vtk.vtkDataSetSurfaceFilter()
            surface_filter.SetInputConnection(cell_to_point.GetOutputPort())
            surface_filter.Update()
            polydata = surface_filter.GetOutput()

            # Clear cell data after surface extraction
            # (Cell data from volume is invalid for surface topology)
            polydata.GetCellData().Initialize()
        else:
            # Convert directly to polydata (may not work for all cell types)
            logger.warning("Converting UnstructuredGrid directly to PolyData")
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(ugrid.GetPoints())

        # Extract geometry
        points, face_counts, face_indices = VTKReader._extract_geometry_from_polydata(
            polydata
        )

        # Extract normals
        normals = VTKReader._extract_normals(polydata)

        # Extract colors
        colors = VTKReader._extract_colors(polydata)

        # Extract point and cell data arrays (from original ugrid to preserve all data)
        point_arrays = []
        point_data = ugrid.GetPointData()
        num_arrays = point_data.GetNumberOfArrays()

        for i in range(num_arrays):
            vtk_array = point_data.GetArray(i)
            name = vtk_array.GetName()
            if name is None or name == "":
                continue

            numpy_array = numpy_support.vtk_to_numpy(vtk_array)
            num_components = vtk_array.GetNumberOfComponents()
            vtk_type = vtk_array.GetDataType()
            data_type = VTKReader._vtk_to_numpy_type(vtk_type)

            if num_components > 1 and numpy_array.ndim == 1:
                numpy_array = numpy_array.reshape(-1, num_components)

            point_arrays.append(
                GenericArray(
                    name=name,
                    data=numpy_array,
                    num_components=num_components,
                    data_type=data_type,
                    interpolation="vertex",
                )
            )

        # Note: cell arrays from ugrid may not map correctly to surface cells
        # We'll extract from the surface polydata instead
        cell_arrays = VTKReader._extract_cell_data_arrays(polydata)

        # Combine arrays
        generic_arrays = point_arrays + cell_arrays

        mesh_data = MeshData(
            points=points,
            face_vertex_counts=face_counts,
            face_vertex_indices=face_indices,
            normals=normals,
            colors=colors,
            generic_arrays=generic_arrays,
        )

        logger.info(
            f"Loaded mesh: {len(points)} points, {len(face_counts)} faces, "
            f"{len(generic_arrays)} data arrays"
        )

        return mesh_data


def read_vtk_file(filename: str | Path, extract_surface: bool = True) -> MeshData:
    """Auto-detect VTK file format and read appropriately.

    Args:
        filename: Path to VTK file (.vtk, .vtp, or .vtu)
        extract_surface: For .vtu and .vtk files, whether to extract surface from volumetric data

    Returns:
        MeshData: Extracted mesh data

    Raises:
        ValueError: If file format is not supported
    """
    filename = Path(filename)
    suffix = filename.suffix.lower()

    if suffix == ".vtp":
        return PolyDataReader.read(filename)
    elif suffix == ".vtk":
        return LegacyVTKReader.read(filename, extract_surface=extract_surface)
    elif suffix == ".vtu":
        return UnstructuredGridReader.read(filename, extract_surface=extract_surface)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. Supported formats: .vtk, .vtp, .vtu"
        )


def validate_time_series_topology(
    mesh_data_sequence: list[MeshData], filenames: list[str | Path] | None = None
) -> dict:
    """Validate topology consistency across a time series of meshes.

    Checks for:
    - Changes in number of points/cells over time
    - Inconsistent primvar sizes

    Args:
        mesh_data_sequence: List of MeshData objects
        filenames: Optional list of filenames for better error messages

    Returns:
        dict: Validation report with warnings and statistics
    """
    if not mesh_data_sequence:
        return {"warnings": [], "is_consistent": True}

    warnings = []
    first_mesh = mesh_data_sequence[0]
    first_n_points = len(first_mesh.points)
    first_n_faces = len(first_mesh.face_vertex_counts)

    # Track topology changes
    topology_changes = []

    for idx, mesh_data in enumerate(mesh_data_sequence):
        frame_label = f"frame {idx}"
        if filenames and idx < len(filenames):
            frame_label = f"{Path(filenames[idx]).name}"

        n_points = len(mesh_data.points)
        n_faces = len(mesh_data.face_vertex_counts)

        # Check for topology changes
        if n_points != first_n_points or n_faces != first_n_faces:
            topology_changes.append(
                {
                    "frame": idx,
                    "label": frame_label,
                    "points": n_points,
                    "faces": n_faces,
                }
            )
            warnings.append(
                f"{frame_label}: Topology change detected - "
                f"{n_points} points (expected {first_n_points}), "
                f"{n_faces} faces (expected {first_n_faces})"
            )

        # Validate primvar sizes
        for array in mesh_data.generic_arrays:
            array_size = len(array.data)

            if array.interpolation == "vertex":
                if array_size != n_points:
                    warnings.append(
                        f"{frame_label}: Point data '{array.name}' size mismatch - "
                        f"got {array_size}, expected {n_points} points"
                    )

            elif array.interpolation == "uniform":
                if array_size != n_faces:
                    warnings.append(
                        f"{frame_label}: Cell data '{array.name}' size mismatch - "
                        f"got {array_size}, expected {n_faces} faces"
                    )

    # Log summary
    if topology_changes:
        logger.warning(
            f"Topology changes detected in {len(topology_changes)}/{len(mesh_data_sequence)} frames"
        )
        logger.warning(f"First frame: {first_n_points} points, {first_n_faces} faces")
        for change in topology_changes[:5]:  # Show first 5
            logger.warning(
                f"  {change['label']}: {change['points']} points, {change['faces']} faces"
            )
        if len(topology_changes) > 5:
            logger.warning(f"  ... and {len(topology_changes) - 5} more frames")

    if warnings and not topology_changes:
        logger.warning(
            f"Found {len(warnings)} primvar size mismatches across time series"
        )

    return {
        "warnings": warnings,
        "is_consistent": len(warnings) == 0,
        "topology_changes": topology_changes,
        "first_topology": {"points": first_n_points, "faces": first_n_faces},
    }
