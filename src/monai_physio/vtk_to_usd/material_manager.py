"""Material management for USD export.

Creates and manages UsdPreviewSurface materials for mesh rendering.
"""

import logging
from typing import Optional

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from .data_structures import MaterialData

logger = logging.getLogger(__name__)


class MaterialManager:
    """Manages creation and binding of USD materials.

    Creates UsdPreviewSurface materials based on MaterialData specifications.
    Handles material caching to avoid duplicate creation.
    """

    def __init__(self, stage: Usd.Stage, materials_scope_path: str = "/World/Looks"):
        """Initialize material manager.

        Args:
            stage: USD stage
            materials_scope_path: Path where materials will be created
        """
        self.stage = stage
        self.materials_scope_path = materials_scope_path
        self.material_cache: dict[str, UsdShade.Material] = {}

        # Create materials scope
        UsdGeom.Scope.Define(stage, materials_scope_path)

    def create_material(
        self, mat_data: MaterialData, time_code: Optional[float] = None
    ) -> UsdShade.Material:
        """Create a UsdPreviewSurface material.

        Args:
            mat_data: Material data specification
            time_code: Optional time code for time-varying materials

        Returns:
            UsdShade.Material: Created material
        """
        # Check cache
        if mat_data.name in self.material_cache:
            logger.debug(f"Returning cached material: {mat_data.name}")
            return self.material_cache[mat_data.name]

        logger.info(f"Creating material: {mat_data.name}")

        # Create material path
        mat_path = f"{self.materials_scope_path}/{mat_data.name}"

        # Create material
        material = UsdShade.Material.Define(self.stage, mat_path)

        # Create shader
        shader_path = f"{mat_path}/PreviewSurface"
        shader = UsdShade.Shader.Define(self.stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")

        # Set shader inputs
        # NOTE (Omniverse/ParaViewConnector compatibility):
        # When a material is bound, many viewers (including Omniverse Kit) will NOT
        # automatically use the mesh's `displayColor` primvar. ParaViewConnector
        # explicitly wires `UsdPrimvarReader_float3(varname=displayColor)` into
        # `UsdPreviewSurface.inputs:diffuseColor`. We mirror that behavior here.
        diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        if mat_data.use_vertex_colors:
            vc_reader_path = f"{mat_path}/PrimvarReader_displayColor"
            vc_reader = UsdShade.Shader.Define(self.stage, vc_reader_path)
            vc_reader.CreateIdAttr("UsdPrimvarReader_float3")
            vc_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(
                "displayColor"
            )
            vc_out = vc_reader.CreateOutput("result", Sdf.ValueTypeNames.Color3f)
            diffuse_input.ConnectToSource(vc_out)
        else:
            diffuse_color = Gf.Vec3f(*mat_data.diffuse_color)
            if time_code is not None:
                diffuse_input.Set(diffuse_color, time_code)
            else:
                diffuse_input.Set(diffuse_color)

        # Specular color
        if mat_data.specular_color != (0.0, 0.0, 0.0):
            specular_input = shader.CreateInput(
                "specularColor", Sdf.ValueTypeNames.Color3f
            )
            specular_color = Gf.Vec3f(*mat_data.specular_color)
            specular_input.Set(specular_color)

        # Emissive color
        if mat_data.emissive_color != (0.0, 0.0, 0.0):
            emissive_input = shader.CreateInput(
                "emissiveColor", Sdf.ValueTypeNames.Color3f
            )
            emissive_color = Gf.Vec3f(*mat_data.emissive_color)
            emissive_input.Set(emissive_color)

        # Opacity
        opacity_input = shader.CreateInput("opacity", Sdf.ValueTypeNames.Float)
        opacity_input.Set(mat_data.opacity)

        # Roughness
        roughness_input = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
        roughness_input.Set(mat_data.roughness)

        # Metallic
        metallic_input = shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
        metallic_input.Set(mat_data.metallic)

        # IOR
        ior_input = shader.CreateInput("ior", Sdf.ValueTypeNames.Float)
        ior_input.Set(mat_data.ior)

        # Connect shader to material surface output
        surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(surface_output)

        # Cache material
        self.material_cache[mat_data.name] = material

        logger.debug(f"Created material '{mat_data.name}' at {mat_path}")

        return material

    def bind_material(
        self, geom_prim: UsdGeom.Gprim, material: UsdShade.Material
    ) -> None:
        """Bind a material to a geometry prim.

        Args:
            geom_prim: Geometry prim (Mesh, Points, etc.)
            material: Material to bind
        """
        # Use Apply() so MaterialBindingAPI ends up in the prim's apiSchemas.
        # Without this, Omniverse Kit's RTX renderer ignores the binding and
        # renders the prim as white.
        binding_api = UsdShade.MaterialBindingAPI.Apply(geom_prim.GetPrim())
        binding_api.Bind(material)

        logger.debug(
            f"Bound material '{material.GetPath()}' to '{geom_prim.GetPath()}'"
        )

    def get_or_create_material(
        self, mat_data: MaterialData, time_code: Optional[float] = None
    ) -> UsdShade.Material:
        """Get existing material from cache or create new one.

        Args:
            mat_data: Material data specification
            time_code: Optional time code for time-varying materials

        Returns:
            UsdShade.Material: Material (cached or newly created)
        """
        if mat_data.name in self.material_cache:
            return self.material_cache[mat_data.name]
        return self.create_material(mat_data, time_code)

    def create_default_material(
        self, name: str = "default", color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    ) -> UsdShade.Material:
        """Create a simple default material.

        Args:
            name: Material name
            color: RGB color

        Returns:
            UsdShade.Material: Created default material
        """
        mat_data = MaterialData(
            name=name, diffuse_color=color, roughness=0.5, metallic=0.0
        )
        return self.create_material(mat_data)
