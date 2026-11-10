# Simpleware Medical Integration for MONAI Physio

This directory contains the code MONAI Physio uses to drive Synopsys Simpleware Medical for heart segmentation (ASCardio) and for tetrahedral meshing (+FE Free).

## Overview

Each integration has two parts: a MONAI Physio method that runs in your Python environment, and a script that runs inside Simpleware Medical's own Python environment. Both MONAI Physio methods launch Simpleware through one shared function.

| Task | MONAI Physio entry point | Script run inside Simpleware | Simpleware module |
|------|--------------------------|------------------------------|-------------------|
| Heart segmentation | `SegmentHeartSimpleware.segment()` (`segment_heart_simpleware.py`) and `SegmentHeartSimplewareTrimmedBranches` | `SimplewareScript_heart_segmentation.py` | ASCardio |
| Tetrahedral meshing | `ProcessVolumetricMesh.extract_label_tetrahedra_simpleware()` (`process_volumetric_mesh.py`) | `SimplewareScript_tetrahedral_mesh.py` | +FE Free |

`__init__.py` provides the shared pieces:

- `DEFAULT_SIMPLEWARE_EXE_PATH` - `C:/Program Files/Synopsys/Simpleware Medical/Y-2026.03/ConsoleSimplewareMedical.exe`
- `run_simpleware_script(exe_path, input_file, input_value, script_path, timeout_s=600)` - checks that the executable and script exist, runs Simpleware, answers its start-up prompt, and returns `(stdout, stderr)`. On timeout it kills the whole process tree (`taskkill /T` on Windows, the process group elsewhere) so no child keeps holding the GPU, then raises `RuntimeError`; a non-zero exit also raises `RuntimeError`.

## Requirements

- Synopsys Simpleware Medical with console mode and Python scripting: X-2025.06 or later for segmentation; meshing has been tested with Y-2026.03, the default path
- ASCardio license for segmentation; +FE Free license for meshing
- `ConsoleSimplewareMedical.exe` (the command-line version)
- No additional Python packages: the scripts use Simpleware's own Python environment

## How It Works

Both integrations exchange files through a temporary directory:

```bash
ConsoleSimplewareMedical.exe \
    --input-file <input image> \       # Becomes the active document
    --input-value <working dir> \       # Read by the script with app.GetInputValue()
    --run-script <script> \
    --exit-after-script \
    --no-progress
```

### Heart segmentation

1. MONAI Physio preprocesses the CT image (1 mm isotropic, intensity scaling) and writes it as `input_image.nii.gz`.
2. `SimplewareScript_heart_segmentation.py` runs ASCardio on the active document and exports each mask as `mask_<name>.mhd`, plus `landmarks.csv` when ASCardio produces landmarks.
3. `SegmentHeartSimpleware` reads the masks, assembles the labelmap (deriving the heart exterior from the interior regions), and returns the result.

### Tetrahedral meshing

1. `extract_label_tetrahedra_simpleware` crops the labelmap to its nonzero voxels, pads it so the structures and their blur are off the image border, renumbers its labels 1..N in ascending order, and writes it as `labelmap.mha` with the labelmap's spacing but an identity direction and a zero origin. It writes `params.json` with `element_size_mm`, `smoothing_sigma_mm`, `part_count` (N) and `shared_interfaces`.
2. `SimplewareScript_tetrahedral_mesh.py` creates one mask per label value and blurs each with the recursive Gaussian filter (`smoothing_sigma_mm`; a 0/1 mask otherwise meshes as a voxel staircase). With `shared_interfaces` it builds one +FE Free FE model holding every mask as a part, so neighboring parts share their interface nodes, and exports `tetrahedra.vtk`, whose `PartId` cell array numbers the parts in label order; otherwise it builds one model per mask and exports `tetrahedra_<part>.vtk` for each. Every model's maximum edge length is `element_size_mm` and its target minimum half of it (Simpleware's defaults when `element_size_mm` is `None`).
3. MONAI Physio reads the linear tetrahedra, splits them by part back into one mesh per original label, and maps the points back through the cropped labelmap's origin and direction. Because Simpleware only ever sees an identity frame, the result does not depend on how Simpleware interprets image orientation. Any inverted cell has two corners swapped, so every tetrahedron has positive volume.

## Usage

### Heart segmentation

```python
import itk

from monai_physio import SegmentHeartSimpleware

segmenter = SegmentHeartSimpleware()
result = segmenter.segment(itk.imread("heart_ct.nii.gz"))

labelmap = result["labelmap"]
heart_mask = result["heart"]
vessel_mask = result["major_vessels"]
```

### Tetrahedral meshing

```python
from monai_physio import ProcessVolumetricMesh

volume_tools = ProcessVolumetricMesh()
meshes = volume_tools.extract_label_tetrahedra_simpleware(labelmap, element_size_mm=2.0)
for label_id, mesh in meshes.items():
    mesh.save(f"heart_tetrahedra_{label_id}.vtu")
```

### Custom Simpleware path

Both classes accept a different install location:

```python
segmenter.set_simpleware_executable_path(
    "D:/CustomPath/Simpleware/ConsoleSimplewareMedical.exe"
)
volume_tools.set_simpleware_executable_path(
    "D:/CustomPath/Simpleware/ConsoleSimplewareMedical.exe"
)
```

### Segmentation output

The ASCardio module segments the following cardiac structures (label IDs match `segment_heart_simpleware.py`):

**Heart Structures (IDs 1-6):**
- 1: Left Ventricle
- 2: Right Ventricle
- 3: Left Atrium
- 4: Right Atrium
- 5: Myocardium
- 6: Heart (combined heart mask; derived from interior regions in postprocessing)

**Major Vessels (IDs 7-10):**
- 7: Aorta
- 8: Pulmonary Artery
- 9: Right Coronary Artery
- 10: Left Coronary Artery

## Testing

Tests that launch Simpleware are marked `requires_simpleware` and run only with `--run-simpleware` (or `--run-all`):

```bash
# Tetrahedral meshing (fast)
py -m pytest tests/test_contour_mesh_extraction.py -k TestExtractTetrahedraExternal --run-gpu --run-simpleware

# Heart segmentation (slow; needs the CHOP-Valve4D data)
py -m pytest tests/test_segment_heart_simpleware.py --run-all
```

## Troubleshooting

**Issue**: `FileNotFoundError: Simpleware Medical executable not found`
- **Solution**:
  - Verify the Simpleware installation path
  - Use `ConsoleSimplewareMedical.exe`, not `SimplewareMedical.exe`
  - Call `set_simpleware_executable_path()` with the right location

**Issue**: `unrecognised option '-python'`
- **Solution**: The GUI version `SimplewareMedical.exe` does not support command-line scripting. Use `ConsoleSimplewareMedical.exe`.

**Issue**: `ImportError: Failed to import Simpleware modules`
- **Solution**: The scripts only run inside Simpleware, via `--run-script` through `ConsoleSimplewareMedical.exe`; do not import them from your own Python.

**Issue**: Missing `mask_*.mhd` files, or `Simpleware Medical produced no segmentation output`
- **Solution**: Check that the input image is passed via `--input-file` so the document is loaded, and that the heart is clearly visible with adequate contrast and field of view.

**Issue**: `Simpleware Medical exported no mesh` or `exported no linear tetrahedra`
- **Solution**: Check the +FE Free license and the Simpleware stderr that MONAI Physio logs as a warning. Run with `log_level=logging.DEBUG` to see Simpleware's stdout.

**Issue**: `... timed out after 600 seconds`
- **Solution**: The image or mesh is too large for the timeout. For segmentation, adjust preprocessing; for meshing, use a coarser `element_size_mm` or pass a larger `timeout_s`.

### Logging

```python
import logging

segmenter = SegmentHeartSimpleware(log_level=logging.DEBUG)
volume_tools = ProcessVolumetricMesh(log_level=logging.DEBUG)
```

## Customization

Edit the script for the task:

- `SimplewareScript_heart_segmentation.py` - which ASCardio parts to segment (the `sw.HeartParts(...)` call) and what is exported.
- `SimplewareScript_tetrahedral_mesh.py` - +FE Free model settings such as edge lengths, target maximum error, and export type. The method expects linear tetrahedra in `tetrahedra.vtk`; other cell types are dropped with a warning.

## Reference Documentation

- Simpleware Medical reference guide
- Simpleware Python API reference (`ScriptingAPI.chm`), with type stubs in `Python3/simpleware/scripting.pyi`
- Console mode: `ConsoleSimplewareMedical.exe --help`

These are installed under `C:\Program Files\Synopsys\Simpleware Medical\<version>\Documentation\`.

### Console mode options used here

```bash
--input-file <file>         # Open input image; becomes the active document
--input-value <value>       # Single string passed to the script via app.GetInputValue()
--run-script <script>       # Execute a Python script
--exit-after-script         # Close after the script completes
--no-progress               # Disable progress messages
```

## License

This integration code is part of MONAI Physio and follows the same license.
Simpleware Medical, ASCardio, and +FE Free are commercial products requiring separate licenses from Synopsys.
