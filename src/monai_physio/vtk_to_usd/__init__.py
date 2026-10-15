"""Public advanced VTK-to-USD conversion layer for MONAI Physio.

This subpackage is a stable low-level API for advanced external users. Code in
MONAI Physio experiments, workflows, and CLIs should use ConvertVTKToUSD from
monai_physio.convert_vtk_to_usd instead of importing this package directly.

Provides:
- File facade: convert_vtk_file
- Data containers: MeshData, ConversionSettings, MaterialData, etc.
- VTK file readers (.vtk, .vtp, .vtu)
- USD primitive writers: UsdMeshConverter, MaterialManager
- Coordinate helpers (LPS to USD Y-up) and mesh splitting utilities
"""

from .converter import convert_vtk_file
from .data_structures import (
    ConversionSettings,
    DataType,
    GenericArray,
    MaterialData,
    MeshData,
    TimeStepData,
    VolumeData,
)
from .material_manager import MaterialManager
from .mesh_utils import (
    cell_type_name_for_vertex_count,
    split_mesh_data_by_cell_type,
    split_mesh_data_by_connectivity,
)
from .usd_mesh_converter import UsdMeshConverter
from .usd_utils import (
    add_framing_camera,
    compute_mesh_extent,
    create_primvar,
    lps_normals_to_usd,
    lps_points_to_usd,
    lps_to_usd,
    sanitize_primvar_name,
    triangulate_face,
)
from .vtk_reader import (
    LegacyVTKReader,
    PolyDataReader,
    UnstructuredGridReader,
    VTKReader,
    read_vtk_file,
    validate_time_series_topology,
)

__all__ = [
    # Data structures
    "ConversionSettings",
    "DataType",
    "GenericArray",
    "LegacyVTKReader",
    "MaterialData",
    # Managers
    "MaterialManager",
    "MeshData",
    "PolyDataReader",
    "TimeStepData",
    "UnstructuredGridReader",
    "UsdMeshConverter",
    # Readers
    "VTKReader",
    "VolumeData",
    "add_framing_camera",
    # Mesh utils (cell type split)
    "cell_type_name_for_vertex_count",
    "compute_mesh_extent",
    # File facade
    "convert_vtk_file",
    "create_primvar",
    "lps_normals_to_usd",
    "lps_points_to_usd",
    # Utilities
    "lps_to_usd",
    "read_vtk_file",
    "sanitize_primvar_name",
    "split_mesh_data_by_cell_type",
    "split_mesh_data_by_connectivity",
    "triangulate_face",
    "validate_time_series_topology",
]

__version__ = "0.1.0"
