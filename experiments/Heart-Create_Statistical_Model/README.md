# Heart-Create_Statistical_Model Experiment

This experiment demonstrates how to create a Principal Component Analysis (PCA) statistical shape model of a heart from a population of meshes using the KCL Heart Model dataset.

**IMPORTANT:** Users should complete this experiment BEFORE attempting the `Heart-Statistical_Model_To_Patient` experiment, as it generates the PCA model data required for patient-specific registration.

## Overview

Statistical shape models (SSMs) capture the natural variation in anatomy across a population. By using PCA on a set of aligned heart meshes, we can:
- Identify the principal modes of shape variation
- Generate new plausible heart shapes by varying PCA coefficients
- Register these models to patient-specific data for personalized digital twins

This experiment uses the **King's College London (KCL) four-chamber heart model dataset**, which provides 20 four-chamber heart models derived from CT images.

## Dataset

The experiment requires the **KCL-Heart-Model** dataset located in `data/KCL-Heart-Model/`.

### Download

Download it automatically with:

```bash
monai-physio-download-data KCL-Heart-Model --directory data/KCL-Heart-Model
```

The data comes from
**[Virtual cohort of adult healthy four-chamber heart meshes from CT images](https://zenodo.org/records/4590294)**.
See `data/KCL-Heart-Model/README.md` for complete details.

### Required Files

- 20 tetrahedral heart meshes (`.vtk` format)
- Mesh labels (chamber, valve, vessel identifiers)
- Optional: existing PCA results from the dataset authors

## Pipeline Overview

This experiment follows a fully automated multi-step process. Each step is a
`# %%` percent-cell Python script that can be run top-to-bottom with
`python <script>.py` or stepped through cell-by-cell in VS Code / Cursor.

### Automated Steps

1. **`1-input_meshes_to_input_surfaces.py`**
   - Converts volumetric tetrahedral meshes to surface representations
   - Extracts relevant anatomical surfaces for shape analysis
   - Prepares data for alignment

2. **`2-input_surfaces_to_surfaces_aligned.py`**
   - Performs initial rigid alignment of surfaces using ICP
   - Centers and orients meshes for consistent positioning
   - Computes and saves the average surface from aligned meshes
   - Prepares aligned data for correspondence computation

3. **`3-registration_based_correspondence.py`**
   - Establishes point correspondences across the population using Greedy affine + ICON deformable registration
   - Uses mask-based distance map registration via `RegisterModelsDistanceMaps`
   - Greedy affine pre-aligns masks; ICON deep learning refines with a deformable field
   - Critical step for meaningful PCA analysis

4. **`4-surfaces_aligned_correspond_to_pca_inputs.py`**
   - Converts corresponded surfaces to PCA input format
   - Prepares data matrices for statistical analysis
   - Validates correspondence quality

5. **`5-compute_pca_model.py`**
   - Computes PCA on the corresponded point sets using sklearn
   - Generates eigenvectors, eigenvalues, and explained variance
   - Exports complete PCA model to `pca_model.json`
   - Produces statistical shape model outputs

## Technical Approach

This experiment uses a fully automated approach combining:

### Registration-Based Correspondence

Instead of traditional mesh parameterization methods (e.g., SPHARM-PDM), this pipeline uses **deformable image registration** to establish correspondences:

- **Greedy affine** (PICSL Greedy) performs fast CPU-based affine pre-alignment
- **ICON deformable** applies deep learning registration on the affine-pre-aligned masks
- Distance maps from surface meshes create continuous fields for registration
- Mask-based approach focuses registration on anatomical structures

**Advantages:**
- Fully automated (no manual parameter tuning)
- Handles complex topologies naturally
- Composed Greedy + ICON transforms provide smooth, invertible deformation fields
- Integrates seamlessly with medical imaging pipelines

### PCA Computation

Statistical shape modeling uses **sklearn's PCA implementation**:
- Computes eigenvectors and eigenvalues from point correspondences
- Calculates explained variance for each mode
- Exports model in JSON format compatible with `RegisterModelsPCA` class

## Workflow

### 1. Download Data

See `data/KCL-Heart-Model/README.md` for download instructions. Place downloaded mesh files in `data/KCL-Heart-Model/input_meshes/`.

### 2. Run All Scripts in Sequence

```bash
cd experiments/Heart-Create_Statistical_Model/

# Execute scripts in order (each can also be stepped through cell-by-cell
# in VS Code or Cursor via the `# %%` cell markers):
python 1-input_meshes_to_input_surfaces.py     # Extract surfaces from volumetric meshes
python 2-input_surfaces_to_surfaces_aligned.py # Rigid ICP alignment + compute average
python 3-registration_based_correspondence.py  # Greedy affine + ICON deformable correspondence
python 4-surfaces_aligned_correspond_to_pca_inputs.py  # Prepare PCA input matrices
python 5-compute_pca_model.py                  # Compute PCA and export JSON model
```

**Total Runtime:** Approximately 1-3 hours depending on hardware (20 heart meshes; Greedy affine is fast on CPU, ICON requires a GPU for reasonable speed).

## Outputs

After completing this experiment, you will have generated files in `kcl-heart-model/`:

### Primary Output: PCA Model JSON
- **`pca_model.json`** - Complete PCA statistical model containing:
  - Mean shape (template surface)
  - PCA eigenvectors (shape modes)
  - PCA eigenvalues (variance per mode)
  - Explained variance ratios
  - Number of components
  - Ready for use with `RegisterModelsPCA` class

### Intermediate Outputs
- **`surfaces/`** - Extracted surface meshes from tetrahedral volumes
- **`surfaces_aligned/`** - ICP-aligned surfaces + `average_surface.vtp`
- **`surfaces_aligned_corresponded/`** - Deformably registered surfaces with correspondence
- **`pca_inputs/`** - Point coordinate matrices ready for PCA computation

### Visualizations
- Various visualization plots generated within the scripts showing:
  - Alignment quality
  - Deformation magnitude maps
  - PCA mode variations
  - Explained variance plots

## Usage in Patient Registration

The outputs from this experiment are used in the `Heart-Statistical_Model_To_Patient` experiment:

```python
from monai_physio import WorkflowFitStatisticalModelToPatient

# Use PCA model from this experiment
workflow = WorkflowFitStatisticalModelToPatient(
    moving_mesh=mean_shape,
    fixed_meshes=patient_surfaces,
    fixed_image=patient_ct,
    pca_json="pca_model.json",  # From this experiment
    pca_number_of_modes=10
)

registered_mesh = workflow.process()["registered_template_model_surface"]
```

## Requirements

### Software
- Python 3.11+ with MONAI Physio installed
- VS Code or Cursor with the Python extension for cell-by-cell execution
  (optional; scripts also run end-to-end as plain Python)
- ITK, VTK, PyVista (included with MONAI Physio)
- picsl-greedy and ICON (included with MONAI Physio)
- scikit-learn for PCA computation

### Data
- KCL Heart Model dataset (20 heart meshes)
- ~5GB disk space for intermediate files
- ~2GB for final outputs

### Compute
- CPU: Multi-core processor (4+ cores recommended for Greedy affine registration)
- RAM: 16GB minimum (32GB recommended)
- GPU: Recommended for ICON deformable registration (CUDA-capable GPU)
- Time: ~1-3 hours total (Greedy is fast; ICON speed depends on GPU availability)

## Citation

If you use this experiment or the KCL dataset, please cite:

> Rodero et al. (2021), "Linking statistical shape models and simulated function in the healthy adult human heart". *PLOS Computational Biology*. DOI: [10.1371/journal.pcbi.1008851](https://doi.org/10.1371/journal.pcbi.1008851)

For ICON registration:
> Greer et al. (2021). "ICON: Learning Regular Maps Through Inverse Consistency". *ICCV*. DOI: [10.1109/ICCV48922.2021.00129](https://doi.org/10.1109/ICCV48922.2021.00129)

## Related Experiments

- **`Heart-Statistical_Model_To_Patient`** - Uses PCA model from this experiment to register to patient data
- **`Heart-VTKSeries_To_USD`** - Converts VTK meshes to USD for Omniverse visualization
- **`Convert_VTK_To_USD`** - General VTK to USD conversion utilities

## Support and Resources

- **KCL Dataset**: [https://zenodo.org/records/4590294](https://zenodo.org/records/4590294)
- **Greedy Documentation**: [https://greedy.readthedocs.io/](https://greedy.readthedocs.io/)
- **MONAI Physio Documentation**: See main repository README and API documentation
- **Issues**: Report bugs or request features on the MONAI Physio GitHub repository

## Troubleshooting

### Missing Data Files
- Ensure KCL dataset is downloaded to `data/KCL-Heart-Model/input_meshes/`
- Check `data/KCL-Heart-Model/README.md` for download instructions
- Verify all 20 heart mesh files (`.vtk` format) are present

### Registration Taking Too Long
- Greedy affine is fast (< 1 minute per subject on CPU)
- ICON deformable is GPU-accelerated; without a GPU it falls back to CPU and will be significantly slower
- Total time for 20 subjects: 1-3 hours depending on GPU availability

### Memory Issues
- Close other applications to free RAM
- ICON can use 4-8GB GPU VRAM; reduce batch size or iterations if needed
- Process fewer meshes initially to test pipeline

### Correspondence Quality Issues
- Check alignment quality from step 2 (ICP should produce good initial alignment)
- Verify average surface looks reasonable before step 3
- If Greedy affine fails, check input mesh quality and topology
- If ICON deformable quality is poor, increase the iteration count on the
  registrar's ICON instance: `registrar.registrar_ICON.set_number_of_iterations(...)`

### Import Errors
- Ensure all MONAI Physio dependencies are installed
- Check Greedy is available: `python -c "from picsl_greedy import Greedy3D; print('ok')"`
- Reinstall environment if needed: `pip install -e .` in repository root

---

**Next Steps:** After completing this experiment, proceed to `experiments/Heart-Statistical_Model_To_Patient/` to learn how to register the statistical model to patient-specific imaging data.
