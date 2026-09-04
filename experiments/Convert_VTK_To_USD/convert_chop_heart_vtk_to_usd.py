# %%
from pathlib import Path

import pyvista as pv

from monai_physio.workflow_convert_vtk_to_usd import WorkflowConvertVTKToUSD

_HERE = Path(__file__).parent

# %%
vtknames = ["a", "la", "lca", "lv", "myo", "pa", "ra", "rv"]
usdnames = [
    "Aorta",
    "LeftAtrium",
    "LeftCoronaryArtery",
    "LeftVentricle",
    "Myocardium",
    "PulmonaryArtery",
    "RightAtrium",
    "RightVentricle",
]

input_dir = _HERE / "../../data/CHOP-Valve4D/CT/Simpleware/parts"
output_dir = _HERE / "results" / "heart"

output_dir.mkdir(parents=True, exist_ok=True)

all_meshes = []
for vtkname, usdname in zip(vtknames, usdnames):
    mesh = pv.read(str(input_dir / f"{vtkname}.vtk"))
    all_meshes.append(mesh)

    workflow = WorkflowConvertVTKToUSD(
        input_meshes=[mesh],
        usd_project_name=f"RVOT28-Dias-{usdname}",
        output_directory=output_dir,
        separate_by_connectivity=False,
        separate_by_cell_type=False,
        appearance="anatomy",
        anatomy_type="heart",
    )
    workflow.process()

workflow = WorkflowConvertVTKToUSD(
    input_meshes=all_meshes,
    usd_project_name="RVOT28-Dias-WholeHeart",
    output_directory=output_dir,
    separate_by_connectivity=False,
    separate_by_cell_type=False,
    static_merge=True,
    appearance="anatomy",
    anatomy_type="heart",
)
workflow.process()
