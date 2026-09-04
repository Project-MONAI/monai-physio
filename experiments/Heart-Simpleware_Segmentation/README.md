# Heart Segmentation using Simpleware Medical

This experiment demonstrates cardiac segmentation using Synopsys Simpleware Medical's ASCardio module integrated with MONAI Physio.

## Overview

The `SegmentHeartSimpleware` class provides integration between MONAI Physio and Synopsys Simpleware Medical for automated heart segmentation. This experiment shows how to:

1. Load cardiac CT images
2. Segment heart structures using ASCardio
3. Visualize and analyze the results
4. Export data for further processing

## Requirements

### Software Requirements

- **Synopsys Simpleware Medical X-2025.06 or later**
  - Default installation path: `C:\Program Files\Synopsys\Simpleware Medical\X-2025.06\`
  - Console mode: `ConsoleSimplewareMedical.exe` (command-line version)
  - ASCardio module license required

- **Python packages** (included in MONAI Physio environment):
  - `itk`
  - `numpy`
  - `matplotlib`
  - `pyvista` (optional, for 3D visualization)

### Data Requirements

- Cardiac CT image (3D volume)
- Recommended: Gated cardiac CT or high-resolution heart scan
- Format: NIfTI (.nii, .nii.gz) or any ITK-readable format
- Image should include complete heart anatomy

## Files

- **`simpleware_heart_segmentation.py`**: Main demonstration script (percent-cell format)
- **`README.md`**: This file
- **`results/`**: Output directory (created automatically)

## Usage

### Quick Start

1. Open `simpleware_heart_segmentation.py` in VS Code, Cursor, or any editor
   with `# %%` cell support — or run it top-to-bottom with
   `python simpleware_heart_segmentation.py`.
2. Update the `input_image_path` near the top of the script to point to your
   cardiac CT image.
3. Run the cells sequentially or execute the file.

### Configuration

Before running, configure these parameters in the script:

```python
# Set input image path
input_image_path = "/path/to/your/cardiac_ct.nii.gz"

# Set custom Simpleware path (if not default)
custom_simpleware_path = "D:/CustomPath/Simpleware/ConsoleSimplewareMedical.exe"
```

### Expected Output

The script generates:

1. **Segmentation files** (in `results/`):
   - `heart_labelmap_simpleware.nii.gz` - Complete labelmap with all structures
   - `heart_mask_simpleware.nii.gz` - Binary mask of heart structures
   - `vessels_mask_simpleware.nii.gz` - Binary mask of major vessels
   - `contrast_mask_simpleware.nii.gz` - Contrast-enhanced regions

2. **Visualizations**:
   - 2D slice views with segmentation overlays
   - 3D surface renderings (if PyVista available)
   - Statistical analysis tables

## Segmented Structures

### Heart Structures (Label IDs 1-6)

- **1**: Left Ventricle (LV)
- **2**: Right Ventricle (RV)
- **3**: Left Atrium (LA)
- **4**: Right Atrium (RA)
- **5**: Myocardium
- **6**: Heart (combined heart mask)

### Major Vessels (Label IDs 7-10)

- **7**: Aorta
- **8**: Pulmonary Artery
- **9**: Right Coronary Artery
- **10**: Left Coronary Artery

## Workflow Details

### Step-by-Step Process

1. **Preprocessing** (automatic):
   - Resampling to 1.0mm isotropic spacing
   - Intensity normalization if needed

2. **Simpleware Integration**:
   - Input image saved to a temporary NIfTI file
   - ConsoleSimplewareMedical.exe is launched with `--input-file` (NIfTI) and `--input-value` (output directory)
   - The Simpleware script runs ASCardio on the loaded image and exports per-structure masks as MHD
   - MONAI Physio assembles the labelmap from the mask files and returns results

3. **Postprocessing** (automatic):
   - Labelmap resampled to original image space
   - Anatomical masks created (heart, vessels, etc.)
   - Optional contrast agent detection

4. **Analysis & Visualization**:
   - Volume calculations for each structure
   - 2D slice visualization
   - 3D surface rendering

### Processing Time

Typical processing times:
- Small CT (256³ voxels): 2-5 minutes
- Medium CT (512³ voxels): 5-10 minutes
- Large CT (1024³ voxels): 10-20 minutes

Times depend on image size, system performance, and Simpleware configuration.

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: Simpleware Medical executable not found`

**Solution**:
- Verify Simpleware installation
- Ensure you're using `ConsoleSimplewareMedical.exe` (not `SimplewareMedical.exe`)
- Update `custom_simpleware_path` in the script
- Check default path: `C:\Program Files\Synopsys\Simpleware Medical\X-2025.06\ConsoleSimplewareMedical.exe`; use `set_simpleware_executable_path()` if installed elsewhere

---

**Issue**: `unrecognised option '-python'` or similar command-line errors

**Solution**:
- Use `ConsoleSimplewareMedical.exe` (console version), not `SimplewareMedical.exe` (GUI version)
- The script is automatically called with `--run-script` flag

---

**Issue**: `ImportError: Failed to import Simpleware modules`

**Solution**:
- Ensure ASCardio module is licensed
- Verify Simpleware version is X-2025.06 or later
- Check that console mode is properly installed

---

**Issue**: `WARNING: No segmentation masks were created` or missing masks

**Solution**:
- Ensure the input NIfTI is passed correctly via `--input-file` so Simpleware has an active document
- Check that ASCardio completed successfully (inspect Simpleware stdout in debug logging)
- Verify input image contains clear heart anatomy and adequate contrast

---

**Issue**: Input image quality issues

**Solution**:
- Verify input image contains heart anatomy
- Check image contrast and quality
- Ensure proper field of view (complete heart visible)
- Try adjusting image preprocessing parameters

---

**Issue**: Segmentation timeout after 600 seconds

**Solution**:
- Image may be too large; consider downsampling
- Check system resources (CPU, RAM)
- Increase timeout in `segment_heart_simpleware.py` if needed

---

**Issue**: Poor segmentation quality

**Solution**:
- Ensure image has adequate contrast
- Use contrast-enhanced CT if available
- Check that heart is centered in field of view
- Verify image orientation is correct

### Debug Mode

Enable detailed logging:

```python
import logging
segmenter = SegmentHeartSimpleware(log_level=logging.DEBUG)
```

This provides:
- Detailed subprocess output
- Simpleware script messages
- File I/O operations
- Timing information

## Integration with Other Workflows

This experiment can be combined with other MONAI Physio workflows:

### 4D Heart Animation
Use segmentation results with `Heart-GatedCT_To_USD` workflow:
```python
# After segmentation
from monai_physio.workflow_convert_image_to_usd import WorkflowConvertImageToUSD

workflow = WorkflowConvertImageToUSD()
workflow.set_static_labelmap(result["labelmap"])
# Continue with 4D USD generation
```

### Statistical Model Registration
Register segmentation with heart model using `Heart-Statistical_Model_To_Patient`:
```python
from monai_physio.workflow_fit_statistical_model_to_patient import WorkflowFitStatisticalModelToPatient

workflow = WorkflowFitStatisticalModelToPatient()
workflow.set_patient_segmentation(result["labelmap"])
# Perform model-to-patient registration
```

### Custom Analysis
Extract specific structures for analysis:
```python
# Get left ventricle only
lv_mask = np.where(labelmap_array == 1, 1, 0)

# Calculate LV volume
lv_volume_ml = np.sum(lv_mask) * voxel_volume / 1000

# Create mesh for computational modeling
from monai_physio.convert_vtk_to_usd import create_mesh_from_mask
lv_mesh = create_mesh_from_mask(lv_mask)
```

## Performance Optimization

### For Faster Processing

1. **Reduce image resolution**:
```python
# Before segmentation
segmenter.set_target_spacing(2.0)  # Use 2mm instead of 1mm
```

2. **Use region of interest**:
```python
# Crop image to heart region before segmentation
from monai_physio.image_tools import crop_to_roi
cropped_image = crop_to_roi(input_image, roi_bounds)
```

## References

- **Simpleware Medical Documentation**: See Synopsys user manual
- **ASCardio Module**: Refer to ASCardio technical documentation
- **MONAI Physio**: Main repository documentation

## Citation

If using this integration in research, please cite:
- Synopsys Simpleware Medical
- MONAI Physio framework
- Any relevant papers using ASCardio segmentation

## License

This integration code follows the MONAI Physio license. Simpleware Medical and ASCardio are commercial products requiring separate licenses from Synopsys.

## Support

For issues with:
- **MONAI Physio integration**: Submit issue to MONAI Physio repository
- **Simpleware Medical/ASCardio**: Contact Synopsys support
- **This experiment**: Check troubleshooting section above

## Version History

- **v0.2.0** (2026-02-06): Documentation and alignment with current implementation
  - README reflects actual script name (`simpleware_heart_segmentation.py`) and correct label IDs (heart 1–6, vessels 7–10)
  - Workflow description updated for `--input-file` (NIfTI) and `--input-value` (output dir) invocation
  - Removed obsolete “placeholder output” limitation; integration works as expected
- **v0.1.0** (2026-02-04): Initial implementation
  - Basic ASCardio integration
  - Heart and vessel segmentation
  - 2D/3D visualization
