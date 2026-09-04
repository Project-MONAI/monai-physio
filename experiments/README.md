# MONAI Physio Experiments

This directory contains research and design experiments that informed the development
of the MONAI Physio library.

**These are not examples of how to use the library**.

## ⭐ For Production Workflows: Use the CLI Commands and Library API

**The CLI commands and library API are the definitive resources for:**
- Production-ready workflows and usage patterns
- Proper class instantiation and method usage
- Library capabilities and best practices
- Command-line tools and parameter specifications

See:
- **CLI Commands**: Run `monai-physio-convert-image-to-usd --help`, `monai-physio-create-statistical-model --help`, and `monai-physio-fit-statistical-model-to-patient --help`
- **CLI Implementation**: `src/monai_physio/cli/` for Python API examples
- **Library Classes**: `src/monai_physio/` for all workflow and utility classes

The experiments here serve as **conceptual references** showing the research process
and design explorations that can inform adaptation to new digital twin models and tasks.

## Purpose

The code in this folder and its subfolders represents:

- **Research prototypes** - Early explorations of algorithms and approaches
- **Design experiments** - Testing different implementation strategies
- **Development iterations** - Code that evolved into the final library components
- **Proof-of-concept work** - Demonstrations that guided architectural decisions

## Experiment Highlights

These experiments demonstrate key digital twin workflows that can be adapted to new
anatomical regions, imaging modalities, and physiological motion tasks.

> **Note:** For production implementations of these workflows, use the CLI commands
> (`monai-physio-convert-image-to-usd`, `monai-physio-create-statistical-model`, `monai-physio-fit-statistical-model-to-patient`) or consult
> the CLI implementation in `src/monai_physio/cli/` for proper class usage and parameter specifications.

### `Reconstruct4DCT` - High-Resolution 4D Reconstruction

**What it demonstrates:** Generates high-resolution 4D sequences by registering a
high-resolution 3D scan with a low-resolution 4D sequence and upsampling. Critical for
obtaining finer details in 4D data used to train AI motion simulations.

**Key technologies:**
- GPU-accelerated ICON registration (inverse consistent)
- Traditional ANTs registration library
- CT-appropriate tissue mass-preserving metric (optional)

**Adaptation potential:** This approach can be finetuned for:
- Different anatomical regions (abdomen, pelvis, extremities)
- Alternative imaging modalities (MRI, ultrasound time series)
- Custom registration metrics for specific tissue types
- Multi-phase imaging protocols beyond respiratory/cardiac gating

### `Lung-GatedCT_To_USD` and `Heart-GatedCT_To_USD` - Complete 4D CT Pipelines

**What they demonstrate:** End-to-end pipelines from 4D CT reconstruction through
segmentation, meshing, USD conversion, and tissue property assignment for Omniverse
visualization and manipulation.

**Pipeline stages:**
1. 4D CT reconstruction (using methods from `Reconstruct4DCT`)
2. AI segmentation (TotalSegmentator)
3. Per-organ mesh generation
4. OpenUSD conversion with animation
5. Tissue property mapping (subsurface scatter, color, etc.)

**Adaptation potential:** Highly modular design enables:
- New anatomical regions (brain, spine, joints, abdominal organs)
- Custom segmentation models and label schemas
- Alternative meshing strategies (higher/lower resolution, topology optimization)
- Domain-specific tissue property mappings for different visualization needs
- Integration with Isaac for Healthcare "Bring-Your-Own-Data" workflows
- Multi-organ system digital twins for surgical planning

### `Heart-VTKSeries_To_USD` - VTK to USD Conversion

**What it demonstrates:** Direct conversion from VTK time series representations to
OpenUSD, preserving colormaps and visualization properties where possible.

**Use cases:**
- Leveraging existing VTK-based medical imaging applications in Omniverse
- Accessing Omniverse connectors (PyAnsys, ROS/ROS2, Isaac for Healthcare)
- Enhanced AR/VR visualization beyond traditional medical imaging viewers
- Integration with physics simulation engines
- Surgical robot training environments

**Adaptation potential:**
- Custom VTK property to USD material mappings
- Support for additional VTK data structures (polydata, unstructured grids)
- Pipeline integration for real-time or near-real-time visualization
- Domain-specific colormap standards (ultrasound Doppler, functional imaging)

**Note:** VTK-to-USD mapping is complex and may require refinement for specific use cases.
This experiment provides a foundation to build upon.

### `Heart-Create_Statistical_Model` - Statistical Shape Model Generation with PCA

**What it demonstrates:** Creating a Principal Component Analysis (PCA) statistical shape
model from a population of anatomical meshes using the KCL Heart Model dataset and SlicerSALT.

**Key benefit:** Statistical shape models capture natural anatomical variation and enable:
- Patient-specific model fitting with anatomically plausible constraints
- Population-based shape analysis and hypothesis testing
- Efficient parameterization of complex 3D anatomy
- Data augmentation for AI training with realistic shape variations

**Technologies:**
- Population mesh alignment and preprocessing
- **SlicerSALT** for interactive correspondence computation (manual steps)
- **SlicerSALT** for PCA analysis (manual steps)
- VTK/ITK for mesh and image processing
- Template mask generation for registration

**Pipeline stages:**
1. Convert volumetric meshes to surfaces (automated)
2. Align population meshes (automated)
3. Compute point correspondences (manual - SlicerSALT)
4. Prepare PCA inputs (automated)
5. Perform PCA analysis (manual - SlicerSALT)
6. Generate template mask (automated)

**Adaptation potential:** This workflow can be adapted to:
- Any anatomical structure (liver, brain, bones, organs)
- Multi-structure shape models (combined organ systems)
- Time-varying shape models (4D motion patterns)
- Disease-specific shape analysis (pathology vs. normal)
- Surgical planning with shape prediction

**Required first**: Complete this experiment BEFORE `Heart-Statistical_Model_To_Patient`
as it generates the PCA model files needed for patient-specific registration.

**Output:** PCA model files (eigenvectors, eigenvalues, mean shape, template mask) for
use in patient registration workflows.

### `Heart-Statistical_Model_To_Patient` - Patient-Specific Digital Twin Registration

**What it demonstrates:** Fitting a simulation-ready or AI-ready anatomical model to
patient-specific imaging data, creating a consistent architecture across subjects. Uses
PCA statistical shape models to constrain deformation to anatomically plausible variations.

**Key benefit:** Using consistent model topology across patients dramatically improves
cross-patient robustness of AI models for physiological motion, enabling a single model
to handle diverse cases.

**Technologies:**
- ICP (Iterative Closest Point) registration for initial alignment
- Labelmap-based deformable registration for anatomical fitting
- PCA (Principal Component Analysis) shape modeling for shape constraints
- Three-stage registration pipeline (ICP → Labelmap-to-Labelmap → Labelmap-to-Image)
- Computationally intensive (>1 hour on typical PC)

**Prerequisites:**
- Complete `Heart-Create_Statistical_Model` experiment first to generate PCA model
- Or use pre-computed PCA model from KCL dataset

**Adaptation potential:** This foundational approach can be extended to:
- Any anatomical structure with consistent topology (liver, brain, vessels)
- Multi-organ system models with coupled motion patterns
- Patient-specific biomechanics simulation preparation
- AI training datasets with normalized anatomical coordinates
- Population-based statistical shape modeling
- Real-time deformable registration for interventional guidance
- Integration with simulation engines (PyAnsys, FEniCS, etc.)

**Output:** Patient-specific USD models ready for physics simulation or AI training.

## Adapting to New Digital Twin Models and Tasks

These experiments illustrate architectural patterns and workflows that can be adapted
to new anatomical regions, physiological processes, and digital twin applications.

### General Adaptation Strategy

1. **Study the experiment conceptually** - Understand the workflow stages and data transformations
2. **Consult the CLI implementation** - See `src/monai_physio/cli/` to identify production classes and methods that implement similar functionality
3. **Identify customization points:**
   - Registration parameters (metrics, transforms, optimization)
   - Segmentation models (custom training, finetuning, label mappings)
   - Meshing parameters (resolution, smoothing, topology)
   - Tissue property mappings (visualization, physics parameters)
4. **Leverage modular design** - Swap components (registration algorithms, segmentation models) as needed
5. **Validate on your data** - Test with representative samples before production deployment

### Key Takeaway

These experiments are **starting points for exploration**, not copy-paste solutions.
The **CLI commands and implementations in `src/monai_physio/cli/`** are the production-quality
code you should use and extend for real-world digital twin projects.

## Testing

Experiment scripts are **not** run as tests. They are exploratory research code:
they assume interactive display, full-resolution parameters, and data layouts
that only exist on the author's machine.

The optional end-to-end test suite runs the `tutorials/` scripts instead:

```bash
pytest tests/test_tutorials.py -v --run-tutorials
```

Run experiment scripts manually, in the order their filename prefixes imply
(`0-`, `1-`, `2-`, ...), from within their own subdirectory:

```bash
cd experiments/Heart-GatedCT_To_USD
py 0-download_and_convert_4d_to_3d.py
py 1-register_images.py
```

Each script requires all dependencies installed (see `pyproject.toml`),
GPU/CUDA support in most cases, large amounts of disk space and memory, and
external data downloads.

## Structure

Each subdirectory represents a different experimental domain:

### Cardiac Imaging
- `Heart-GatedCT_To_USD/` - Complete cardiac 4D CT pipeline from images to animated USD models
- `Heart-VTKSeries_To_USD/` - Direct VTK time series to USD conversion for cardiac data
- `Heart-Create_Statistical_Model/` - PCA statistical shape model creation from population meshes using SlicerSALT
- `Heart-Statistical_Model_To_Patient/` - Advanced model-to-patient registration with ICP, mask-based, and PCA methods

### Pulmonary Imaging
- `Lung-GatedCT_To_USD/` - Respiratory motion analysis using DirLab 4D-CT benchmark data
- `Lung-VesselsAirways/` - Specialized vessel and airway segmentation with deep learning models

### Advanced Visualization
- `Colormap-VTK_To_USD/` - Time-varying colormap rendering for scalar field visualization in Omniverse
- `DisplacementField_To_USD/` - Convert registration displacement fields to USD for PhysicsNeMo visualization

### Data Processing
- `Reconstruct4DCT/` - 4D CT reconstruction from sparse temporal samples using deformable registration

## Important Notes

**These experiments are not production code** - They may contain:
- Hardcoded paths and parameters
- Incomplete error handling
- Experimental APIs that have since changed
- Code that doesn't follow the final library conventions
- Dataset-specific assumptions and configurations

**Do not use these as usage examples** - These are research artifacts, not tutorials.

### For Production Use, Consult:

1. **Tutorials in `tutorials/`** - the primary resource
   - End-to-end runnable scripts on downloadable data
   - Show the workflow classes in context, with the constants to change for
     your own scans
   - Documented at https://project-monai.github.io/monai-physio/tutorials.html

2. **CLI Commands** - the same workflows packaged for unattended runs
   - `monai-physio-convert-image-to-usd` - Complete heart-gated CT workflow
   - `monai-physio-create-statistical-model` - Create PCA statistical shape model from sample meshes
   - `monai-physio-fit-statistical-model-to-patient` - Model-to-patient registration
   - Run with `--help` for all options and parameter specifications
   - Tested on diverse datasets

3. **CLI Implementation in `src/monai_physio/cli/`**
   - Production-ready workflow code
   - Proper class usage patterns and parameter specifications
   - Complete error handling and validation
   - Python API usage examples

4. **Main library documentation**
   - API references and class documentation
   - Architecture explanations
   - Performance considerations

4. **`tests/` directory**
   - Unit tests demonstrating correct API usage
   - Edge case handling
   - Expected input/output formats

### Using Experiments as Inspiration

**DO:** Study these experiments to understand:
- Overall workflow architecture and data flow
- Problem decomposition strategies
- Technology choices and trade-offs
- Potential adaptation approaches for your domain

**DON'T:** Copy experiment code directly into production:
- Experiments may use outdated APIs
- Error handling may be minimal or missing
- Paths and parameters are often hardcoded for specific datasets
- Performance may not be optimized

**INSTEAD:** Use the CLI commands or the production implementations in `src/monai_physio/cli/` and extend them for your needs.

## Development History

This experimental code was instrumental in:
1. Defining the final library architecture
2. Testing registration algorithms (ICON, SyN, LDDMM)
3. Evaluating segmentation approaches (TotalSegmentator)
4. Developing the USD export pipeline
5. Optimizing the complete 4D CT → USD workflow
6. Identifying modular extension points for new anatomical regions and tasks

The lessons learned from these experiments led to the creation of production components
like the unified `HeartGatedCTProcessor` class and other reusable classes in the main
library.

### Evolution from Experiment to Production

The typical evolution path was:
1. **Experiment** - Proof of concept with specific dataset (this directory)
2. **Refinement** - Generalization and parameterization
3. **Testing** - Validation across diverse datasets
4. **Production** - Documented, tested CLI commands and library API ⭐

When exploring new digital twin applications, you can follow a similar path:
- Start by understanding relevant experiments here as conceptual references
- Examine CLI implementations in `src/monai_physio/cli/` for proper library usage
- Use CLI commands (`monai-physio-convert-image-to-usd`, `monai-physio-create-statistical-model`, `monai-physio-fit-statistical-model-to-patient`) as starting points
- Extend and adapt production code with your domain-specific requirements
- Contribute back improvements and new capabilities to the community

**Remember:** The CLI commands and their implementations in `src/monai_physio/cli/` are your
source of truth for how to properly use MONAI Physio classes, workflows, and capabilities in production environments.
