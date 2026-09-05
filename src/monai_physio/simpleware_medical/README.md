# Simpleware Medical Integration for MONAI Physio

This directory contains integration code for using Synopsys Simpleware Medical with MONAI Physio for heart segmentation.

## Overview

The integration enables MONAI Physio to leverage Simpleware Medical's ASCardio module for automated cardiac segmentation. The implementation uses a two-component architecture:

1. **segment_heart_simpleware.py** (in parent directory): A Python class that inherits from `SegmentAnatomyBase` and manages the external Simpleware Medical process
2. **SimplewareScript_heart_segmentation.py** (this directory): A Python script that runs within the Simpleware Medical environment and performs the actual segmentation using ASCardio

## Requirements

- Synopsys Simpleware Medical X-2025.06 or later
- ASCardio module license
- Valid Simpleware Medical installation with console mode and Python scripting support
- ConsoleSimplewareMedical.exe (command-line version)

## Current Status

**FUNCTIONAL**: This integration works as expected.

### How It Works

The integration passes the input CT image and output directory directly to Simpleware Medical:

1. **Input**: The preprocessed NIfTI image is written to a temporary file; its path is passed via `--input-file` so Simpleware opens it as the active document.
2. **Output directory**: The temporary output directory is passed via `--input-value`; the script reads it with `app.GetInputValue()`.
3. The script runs ASCardio on the current document (the loaded NIfTI), then exports each mask as `mask_<name>.mhd` into that directory.
4. `SegmentHeartSimpleware` reads the MHD mask files, assembles the labelmap, and returns the result.

```bash
ConsoleSimplewareMedical.exe \
    --input-file input_image.nii.gz \      # Input CT (becomes active document)
    --input-value <output_dir> \            # Where to write mask_*.mhd files
    --run-script SimplewareScript_heart_segmentation.py \
    --exit-after-script \
    --no-progress
```

The Python script then:
```python
output_dir = app.GetInputValue()  # Output directory from --input-value
doc = sw.App.GetDocument()  # Active document (loaded NIfTI)
as_cardio = doc.GetAutoSegmenters().GetASCardio()
# ... run ASCardio, then export each mask to mask_<name>.mhd in output_dir ...
```

## Installation

1. Install Simpleware Medical (default path: `C:\Program Files\Synopsys\Simpleware Medical\X-2025.06\`)
2. Ensure the ASCardio module is licensed and available
3. No additional Python packages are required (Simpleware has its own Python environment)

## Usage

### Basic Usage

```python
from monai_physio.segment_heart_simpleware import SegmentHeartSimpleware
import itk

# Create segmenter instance
segmenter = SegmentHeartSimpleware()

# Load CT image
ct_image = itk.imread("heart_ct.nii.gz")

# Perform segmentation
result = segmenter.segment(ct_image)

# Access results
labelmap = result["labelmap"]
heart_mask = result["heart"]
vessel_mask = result["major_vessels"]

# Save results
itk.imwrite(labelmap, "heart_segmentation.nii.gz")
```

### Custom Simpleware Path

If Simpleware Medical is installed in a non-default location:

```python
segmenter = SegmentHeartSimpleware()
segmenter.set_simpleware_executable_path(
    "D:/CustomPath/Simpleware/ConsoleSimplewareMedical.exe"
)
```

### Segmentation Output

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

## Architecture

### Process Flow

1. MONAI Physio preprocesses the CT image (resampling to 1 mm isotropic, intensity scaling).
2. Preprocessed image is saved to a temporary NIfTI file (e.g. `input_image.nii.gz`) in a temporary directory.
3. `ConsoleSimplewareMedical.exe` is launched with:
   - `--input-file <path_to_input.nii.gz>` - the preprocessed CT (Simpleware opens it as the active document)
   - `--input-value <tmp_dir>` - directory where the script will write mask files
   - `--run-script SimplewareScript_heart_segmentation.py`
   - `--exit-after-script` and `--no-progress`
4. The script runs inside Simpleware:
   - Gets the output directory from `app.GetInputValue()`
   - Uses the current document (the loaded NIfTI) and ASCardio to segment heart and vessels
   - Exports each mask as `mask_<name>.mhd` into the output directory
5. MONAI Physio reads the `mask_*.mhd` files, builds the labelmap (including heart exterior from interior regions), and returns the result.

### Communication

- **Executable**: `ConsoleSimplewareMedical.exe` (command-line version)
- **Script**: `--run-script SimplewareScript_heart_segmentation.py`
- **Input**: NIfTI image path via `--input-file` (becomes the active document)
- **Output**: Directory path via `--input-value`; script writes `mask_<name>.mhd` per structure
- **Protocol**: File-based I/O via temporary directory
- **Timeout**: 10 minutes (configurable in `segment_heart_simpleware.py`)

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: Simpleware Medical executable not found`
- **Solution**:
  - Verify Simpleware installation path
  - Ensure you're using `ConsoleSimplewareMedical.exe` not `SimplewareMedical.exe`
  - Use `set_simpleware_executable_path()` to specify custom location

**Issue**: `unrecognised option '-python'`
- **Solution**: The GUI version `SimplewareMedical.exe` doesn't support command-line scripting. Use `ConsoleSimplewareMedical.exe` instead.

**Issue**: `ImportError: Failed to import Simpleware modules`
- **Solution**: Ensure the script is being called with `--run-script` flag through `ConsoleSimplewareMedical.exe`

**Issue**: `WARNING: No segmentation masks were created` or missing mask files
- **Solution**: Check that the input NIfTI is passed via `--input-file` so the document is loaded. Ensure input image quality, contrast, and field of view; the heart should be clearly visible.

**Issue**: Segmentation timeout after 600 seconds
- **Solution**: Image may be too large or high resolution. Consider adjusting preprocessing parameters.

### Logging

Enable detailed logging to troubleshoot issues:

```python
import logging

segmenter = SegmentHeartSimpleware(log_level=logging.DEBUG)
```

## Customization

### Modifying ASCardio Parameters

To customize ASCardio segmentation parameters, edit `SimplewareScript_heart_segmentation.py`:

```python
cardio.auto_segment(
    image=image,
    segment_chambers=True,
    segment_myocardium=True,
    segment_vessels=True,
    # Add custom parameters here
)
```

## Reference Documentation

For more information on Simpleware Medical and ASCardio:
- Simpleware Medical User Guide
- ASCardio Module Documentation
- Simpleware Python API Reference (ScriptingAPI.chm)
- Console Mode Documentation

Located in: `C:\Program Files\Synopsys\Simpleware Medical\X-2025.06\Documentation\`

### Console Mode Command Reference

```bash
# View all command-line options
ConsoleSimplewareMedical.exe --help

# Key options used by this integration:
--input-file <file>         # Open input (NIfTI image); becomes active document
--input-value <value>       # Single string passed to script via app.GetInputValue() (e.g. output dir)
--run-script <script>       # Execute a Python script
--exit-after-script         # Close after script completes
--no-progress               # Disable progress messages
```

### Example Command (as used by SegmentHeartSimpleware)

```bash
ConsoleSimplewareMedical.exe \
    --input-file /tmp/.../input_image.nii.gz \
    --input-value /tmp/.../output_dir \
    --run-script SimplewareScript_heart_segmentation.py \
    --exit-after-script \
    --no-progress
```

### Simpleware Python API Usage

Within the script, the output directory is passed via `--input-value` and read as:

```python
import simpleware.scripting as sw

app = sw.App.GetInstance()
output_dir = app.GetInputValue()  # Value from --input-value (output directory)
doc = sw.App.GetDocument()  # Active document (NIfTI loaded via --input-file)
```

## License

This integration code is part of MONAI Physio and follows the same license.
Simpleware Medical and ASCardio are commercial products requiring separate licenses from Synopsys.
