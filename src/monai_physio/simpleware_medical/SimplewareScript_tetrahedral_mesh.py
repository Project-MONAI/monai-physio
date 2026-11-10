"""Simpleware Medical Python script for +FE Free tetrahedral meshing.

This script is designed to run within the Synopsys Simpleware Medical
environment. The active document is a labelmap image (opened via
``--input-file``) whose labels are numbered 1..``part_count``; the working
directory passed via ``--input-value`` holds ``params.json`` and receives the
meshes: ``tetrahedra.vtk``, holding every label as a part tagged by its
``PartId`` cell array, when ``shared_interfaces`` is set, otherwise one
``tetrahedra_<part>.vtk`` per label. It is called as an external process by
``ProcessVolumetricMesh.extract_label_tetrahedra_simpleware``.
"""

import json
import os

import simpleware.scripting as sw

app = sw.App.GetInstance()
doc = sw.App.GetDocument()

work_dir = app.GetInputValue()
with open(os.path.join(work_dir, "params.json"), encoding="utf-8") as fh:
    params = json.load(fh)

# A mask carries no sub-voxel boundary position, so +FE Free meshes its voxel
# staircase; blurring each label's mask gives the mesher a smooth boundary.
smoothing_sigma_mm = params["smoothing_sigma_mm"]
masks = []
for part in range(1, params["part_count"] + 1):
    existing = {mask.GetName() for mask in doc.GetMasks()}
    doc.CreateMasksFromPresegmentedData(part, part)
    (mask,) = [mask for mask in doc.GetMasks() if mask.GetName() not in existing]
    if smoothing_sigma_mm > 0.0:
        mask.Activate()
        doc.ApplyRecursiveGaussianFilter(
            sw.Doc.TargetActiveMask,
            True,
            sw.Sigma(sw.Length(smoothing_sigma_mm, sw.Length.Millimetre)),
        )
    masks.append(mask)


def export_mesh(name: str, parts: list[sw.Mask]) -> None:
    """Mesh *parts* as one +FE Free model and export it as ``<name>.vtk``.

    The parts of one model share their nodes where they meet, and the export
    numbers them, in its ``PartId`` cell array, in the order they were added.
    """
    model = doc.CreateFeModel(name, True)
    doc.SetActiveModel(model)
    for part_mask in parts:
        model.GetPartsContainer().AddMask(part_mask)
    model.SetMeshingAlgorithmType(sw.Model.Free)
    model.SetExportType(sw.Model.VtkVolume)

    element_size_mm = params["element_size_mm"]
    if element_size_mm is not None:
        model.SetMaximumEdgeLength(element_size_mm)
        model.SetTargetMinimumEdgeLength(0.5 * element_size_mm)

    doc.GenerateMesh()
    doc.ExportVtkVolume(os.path.join(work_dir, f"{name}.vtk"), False)


if params["shared_interfaces"]:
    export_mesh("tetrahedra", masks)
else:
    for part, mask in enumerate(masks, start=1):
        export_mesh(f"tetrahedra_{part}", [mask])
