"""Data structures for VTK to USD conversion.

Based on OmniConnectData from ParaViewConnector but simplified for file-based conversion.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from numpy.typing import NDArray


class DataType(Enum):
    """Data type enumeration for generic arrays."""

    UCHAR = "uchar"
    CHAR = "char"
    USHORT = "ushort"
    SHORT = "short"
    UINT = "uint"
    INT = "int"
    ULONG = "ulong"
    LONG = "long"
    FLOAT = "float"
    DOUBLE = "double"


@dataclass
class GenericArray:
    """Generic data array that can be converted to USD primvar.

    Represents a named array of data with a specific type and number of components.
    Mimics OmniConnectGenericArray from ParaViewConnector.
    """

    name: str
    data: NDArray
    num_components: int
    data_type: DataType
    interpolation: str = "vertex"  # 'vertex', 'uniform' (per-face), 'constant'

    def __post_init__(self) -> None:
        """Validate and normalize data shape.

        Handles three cases:
        1. 1D array with num_components=1: kept as-is (scalar)
        2. 1D array with num_components>1: reshaped to 2D if length is divisible
        3. 2D array: validated that shape[1] matches num_components
        """
        if self.data.ndim == 1:
            if self.num_components == 1:
                # Scalar array - keep as 1D
                pass
            elif self.num_components > 1:
                # Flat multi-component array - reshape to 2D
                if len(self.data) % self.num_components != 0:
                    raise ValueError(
                        f"Data length {len(self.data)} not divisible by "
                        f"num_components={self.num_components}"
                    )
                self.data = self.data.reshape(-1, self.num_components)
            else:
                raise ValueError(
                    f"num_components must be >= 1, got {self.num_components}"
                )
        elif self.data.ndim == 2:
            if self.data.shape[1] != self.num_components:
                raise ValueError(
                    f"Data shape {self.data.shape} incompatible with "
                    f"num_components={self.num_components}"
                )
        else:
            raise ValueError(
                f"Data must be 1D or 2D array, got shape {self.data.shape}"
            )


@dataclass
class MaterialData:
    """Material properties for USD conversion.

    Mimics OmniConnectMaterialData from ParaViewConnector.
    """

    name: str = "default_material"
    diffuse_color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    specular_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    opacity: float = 1.0
    roughness: float = 0.5
    metallic: float = 0.0
    ior: float = 1.5
    use_vertex_colors: bool = False


@dataclass
class MeshData:
    """Mesh geometry data for USD conversion.

    Mimics OmniConnectMeshData from ParaViewConnector.
    """

    points: NDArray  # Shape: (N, 3)
    face_vertex_counts: NDArray  # Shape: (F,)
    face_vertex_indices: NDArray  # Shape: (sum(face_vertex_counts),)
    normals: Optional[NDArray] = None  # Shape: (N, 3) or (sum(face_vertex_counts), 3)
    uvs: Optional[NDArray] = None  # Shape: (N, 2) or (sum(face_vertex_counts), 2)
    colors: Optional[NDArray] = None  # Shape: (N, 3) or (N, 4)
    generic_arrays: list[GenericArray] = field(default_factory=list)
    material_id: str = "default_material"

    def __post_init__(self) -> None:
        """Validate mesh data."""
        if self.points.shape[1] != 3:
            raise ValueError(f"Points must have shape (N, 3), got {self.points.shape}")


@dataclass
class VolumeData:
    """Volume data for USD conversion.

    Mimics OmniConnectVolumeData from ParaViewConnector.
    """

    image_data: NDArray  # 3D array
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scalar_range: Optional[tuple[float, float]] = None
    generic_arrays: list[GenericArray] = field(default_factory=list)


@dataclass
class TimeStepData:
    """Data for a single time step."""

    time_code: float
    meshes: dict[str, MeshData] = field(default_factory=dict)
    volumes: dict[str, VolumeData] = field(default_factory=dict)
    materials: dict[str, MaterialData] = field(default_factory=dict)


@dataclass
class ConversionSettings:
    """Settings for VTK to USD conversion."""

    # Output settings
    output_binary: bool = False
    meters_per_unit: float = 1.0
    up_axis: str = "Y"  # 'Y' or 'Z'

    # Mesh processing
    triangulate_meshes: bool = True
    compute_normals: bool = True
    preserve_point_arrays: bool = True
    preserve_cell_arrays: bool = True
    separate_objects_by_cell_type: bool = False  # Split into separate USD meshes by cell type (triangle/quad/tetra/hex etc.)
    separate_objects_by_connectivity: bool = False  # Split into separate USD meshes by connected components (object1, object2, ...). Mutually exclusive with separate_objects_by_cell_type.

    # Material settings
    use_preview_surface: bool = True
    default_color: tuple[float, float, float] = (0.8, 0.8, 0.8)

    # Time settings
    frames_per_second: float = 24.0
    use_time_samples: bool = True

    # Array prefixes
    point_array_prefix: str = "vtk_point_"
    cell_array_prefix: str = "vtk_cell_"

    def __post_init__(self) -> None:
        """Validate that at most one of the split options is enabled."""
        if self.separate_objects_by_cell_type and self.separate_objects_by_connectivity:
            raise ValueError(
                "separate_objects_by_cell_type and separate_objects_by_connectivity "
                "cannot both be True; enable only one."
            )
