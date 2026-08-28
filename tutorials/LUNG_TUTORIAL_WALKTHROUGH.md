# Dockerized Lung Tutorial Walkthrough

This guide covers the featured lung course only: Tutorials 1, 4, and 6-11.
It records the validated commands, outputs, timings, visualization paths,
troubleshooting, and learning objectives. Heart tutorials, Tutorial 2
finetuning, full-dataset regeneration during class, and Tutorial 12 are outside
this deployment profile.

## Start the course container

From the repository root on the GPU host:

```bash
docker/check-host.sh
docker build -t monai-physio:tutorials .
HF_TOKEN=... docker/download-lung-bundles.sh
docker/tutorial-shell.sh
```

The interactive shell bind-mounts the repository and persistent model caches,
runs as the host UID/GID, enables the GPU, and allocates 8 GiB of shared
memory. Inside it, each lesson is reduced to its Python command. Type `exit`
when finished; data, checkpoints, caches, and outputs remain on the host.
The download command installs both private bundle profiles and the shell uses
their segmentation caches without contacting Hugging Face.
The image uses CUDA 12.6.3 and requires NVIDIA Linux driver 560.35.05 or newer.

### Brev VM launchable

For a Brev Launchable, select **VM Mode** and paste `docker/brev-deploy.sh` into
the setup-script field. A separate source checkout is not required. Define
these launch parameters:

| Parameter | Required | Value |
|---|---|---|
| `NGC_API_KEY` | Yes | Private NGC registry key; leave the default empty |
| `HF_TOKEN` | Yes | Read token for the private workshop dataset; leave the default empty |

The setup script logs in to NGC through standard input, pulls and checks the
fixed `nvcr.io/0569033758414229/physiomotion:v0.2-cu126` image, seeds a
persistent workspace at `$HOME/monai-physio`, and downloads the `course` and
`offline-segmentation` profiles from the private
`maximilianofir/physioMotionWorkshop` Hugging Face dataset at pinned revision
`459c538385d36eb2ebb7a92bb0086494ee2ebdcf`. Brev makes launch parameters
available only during setup, and the script removes both credentials when it
finishes. Do not embed either credential in the script or use one as a
parameter default. See the [Brev Launchables
documentation][brev-launchables] for the current VM-mode configuration steps.

To download or repair the bundles later, export a read token and rerun:

```bash
HF_TOKEN=... docker/download-lung-bundles.sh
```

After the VM starts, use the same host commands as above:

```bash
cd "$HOME/monai-physio"
docker/check-host.sh
docker/tutorial-shell.sh
```

[brev-launchables]: https://docs.nvidia.com/brev/concepts/launchables

## Reusable mesh viewer

Run viewer commands from a second host terminal, not from inside the tutorial
shell:

```bash
./docker/view-meshes.sh --port 8080 INPUT_FILE [INPUT_FILE ...]
```

The viewer plays animated USD and overlays one or more static VTP surfaces.
It does not display CT or labelmap voxels; use 3D Slicer for MHA/NIfTI review.
Stop the viewer with `Ctrl+C` before starting another one on the same port.

For a remote GPU host, forward the viewer port from local PowerShell:

```powershell
ssh -N -L 8080:127.0.0.1:8080 USER@REMOTE_HOST
```

Then open `http://127.0.0.1:8080/index.html`. Use the same replacement port in
the Docker command, tunnel, and URL when 8080 is occupied.

## Compact one-patient deployment

The Docker image contains the Python, CUDA, PhysicsNeMo, segmentation,
registration, ITK, VTK/PyVista, OpenUSD, and Trame dependencies. Model weights
and datasets remain mounted because their licenses and update mechanisms differ
from the source image.

The rebuilt Docker image is approximately 12.46 GB (11.61 GiB). The private
course archive is 1.11 GB (1.04 GiB) compressed and installs 3.13 GB
(2.91 GiB). The offline-segmentation archive is 2.58 GB (2.40 GiB) compressed
and installs 2.80 GB (2.61 GiB). Allow at least 20 GiB for the image, both
installed profiles, the Git checkout, Docker layer overhead, and regenerated
outputs.

| Required content | Size | Purpose |
|---|---:|---|
| `data/DirLab-4DCT/Case1Pack_T??.mha` | 74.19 MiB | Ten acquired phases for one held-out patient |
| `output/tutorial_01_lung/lung_model.all_painted.usd` | 203.25 MiB | Prepared Tutorial 1 animation |
| Tutorial 4 lung labelmap and VTP subset | 24.40 MiB | Label and surface inspection |
| Tutorial 6 mean, PCA JSON, and leading mode subset | 180.05 MiB | Prepared population-model inspection |
| `output/tutorial_07_lung/` | 60.36 MiB | Prepared static patient fit |
| Tutorial 8 Case 1 reference and ten forward transforms | 1,426.60 MiB | Registration provenance used by the compatibility fit |
| `output/tutorial_08_lung_checkpoint/Case1Pack/` | 179.05 MiB | Checkpoint-compatible reference, coefficients, and phases |
| Tutorial 9 three-case smoke inputs | 608.64 MiB | Real manifests, references, coefficients, and targets for one no-save batch |
| Final MGN checkpoint and domain files | 226.60 MiB | Frozen surrogate, graph, PCA domain, and summaries |
| Tutorial 11 `ground_truth/` cache | 1.73 MiB | Independent lobe labelmaps |

Retain these Tutorial 4 files:

```text
tutorials/output/tutorial_04_lung/patient_nvsegmentctmri_lung_labelmap.mha
tutorials/output/tutorial_04_lung/patient_nvsegmentctmri_lung.vtp
tutorials/output/tutorial_04_lung/patient_nvsegmentctmri_surfaces.vtp
tutorials/output/tutorial_04_lung/patient_nvsegmentctmri_lung_*_lobe_*.vtp
```

Retain these Tutorial 6 files:

```text
tutorials/output/tutorial_06_lung/pca_model.json
tutorials/output/tutorial_06_lung/pca_mean_surface.vtp
tutorials/output/tutorial_06_lung/pca_mode_01_minus_2sigma.vtp
tutorials/output/tutorial_06_lung/pca_mode_01_plus_2sigma.vtp
```

Under `tutorials/output/tutorial_08_lung/Case1Pack/`, retain only
`Case1Pack_ssm_surface.vtp`, `Case1Pack_ssm_pca_coefficients.json`, and the ten
`Case1Pack_T??_forward_tfm.hdf` files. Preserve timestamps with `rsync -a` or a
tar archive because the compatibility cache records transform provenance.

The downloadable course profile also includes the reference surface, PCA
coefficients, manifest, and ten training targets for Case 2 and Case 3. Together
with Case 1, these provide the three real cases required by Tutorial 9's
`--smoke-test` without shipping the complete ten-case training cache.

Under `tutorials/network_weights/physicsnemo_mgn_lung_motion/`, retain:

```text
mgn_stage_model.pt
mgn_stage_model_metadata.json
pca_mean_surface.vtp
pca_mean_template.vtp
pca_model.json
shared_edge_features.pt
shared_edge_index.pt
training_losses.json
training_validation_rmse.csv
training_validation_rmse.json
```

Intermediate epoch checkpoints and the complete 6.900 GiB weights tree are not
needed when training and resume are skipped. Also retain:

```text
tutorials/output/tutorial_11_lung/Case1Pack/ground_truth/
```

One patient cannot rebuild a population PCA model or a training/validation
split. In this compact profile, Tutorials 6-9 inspect prepared population and
training artifacts, while Tutorials 10 and 11 perform genuine held-out
inference and evaluation on Case 1.

The `offline-segmentation` profile additionally installs the pinned
NV-Segment-CTMR Hugging Face cache and the TotalSegmentator task 117, 291-295,
297, and 299 weights used by the full and fast lung paths.

Optional additions not included in either profile:

| Capability | Additional content | Size |
|---|---|---:|
| Rerun Tutorial 7 | `data/Chest-CT/`, final distance-map ICON checkpoint, and stock uniGradICON checkpoint | 624.41 MiB |

The host needs the NVIDIA Container Toolkit and driver 560.35.05 or newer.

## Course sequence

| Time | Tutorial | Course action |
|---:|---|---|
| 0-10 min | 1: gated CT to animated USD | Inspect prepared animation and pipeline |
| 10-20 min | 4: CT to VTK | Run or inspect segmentation surfaces |
| 20-42 min | 6: PCA shape model | Build or inspect prepared PCA outputs |
| 42-55 min | 7: patient fit | Fit one static patient or inspect cache |
| 55-68 min | 8: propagate through 4D | Inspect prepared registration targets |
| 68-88 min | 9: train the surrogate | Run one no-save smoke batch |
| 88-103 min | 10: predict motion | Run supplied checkpoint inference |
| 103-113 min | 11: evaluate motion | Run image-backed scores |
| 113-120 min | Design canvas | Map the pattern to another organ |

## Tutorial 1: gated CT to animated USD

Source: [`tutorial_01_lung_gated_ct_to_usd.py`](tutorial_01_lung_gated_ct_to_usd.py)

### Why and command

This tutorial converts ten respiratory CT phases into one time-sampled anatomy
scene. It separates anatomical identification from motion: one reference phase
is segmented, and image-registration transforms carry those surfaces through
the remaining phases.

```bash
python tutorials/tutorial_01_lung_gated_ct_to_usd.py
```

### What executes

1. Read `Case1Pack_T00.mha` through `Case1Pack_T90.mha`.
2. Select `T70` as the reference phase.
3. Segment permitted chest structures with TotalSegmentator.
4. Register all phases to the reference with Greedy.
5. Extract reference surfaces and propagate them with the transforms.
6. Write and paint a ten-frame OpenUSD scene.

Fine heart chambers are absent because separately licensed segmentation tasks
are disabled by default. Greedy registration is CPU work; segmentation uses
the GPU.

### Expected outputs

Under `tutorials/output/tutorial_01_lung/`:

| Output | Expected result |
|---|---|
| `reference_labelmap.mha` | T70 segmentation |
| `slice_*_all_registered.mha` | Ten CTs in the reference frame |
| `slice_*_all_{forward,inverse}.hdf` | Registration transforms |
| `lung_model.all_painted.usd` | About 203 MiB, ten animated phases |
| `slice_007_registered_test.png` | Registered CT check |
| `lung_model_test.png` | Anatomy render check |

Visualize at three frames per second:

```bash
./docker/view-meshes.sh --port 8080 --fps 3 \
  tutorials/output/tutorial_01_lung/lung_model.all_painted.usd
```

The validated viewer preloaded ten fixed-topology point frames and replayed the
cycle in about 3.4 seconds. File creation alone is not validation: inspect the
CT for tearing or doubling and confirm coherent lung motion.

### Troubleshooting and objectives

- Missing frames: verify all ten Case 1 MHA files and the repository mount.
- License error: keep `use_totalsegmentator_licensed_tasks = False` unless the
  relevant weights are licensed and configured.
- Slow playback: confirm the startup log reports preloaded point frames; lower
  `--fps` if remote rendering cannot sustain the requested rate.
- Memory failure: run this tutorial alone with the configured 8 GiB shared
  memory and adequate host RAM.

Learning objectives: explain the reference phase, distinguish segmentation
from registration, and relate CTs, transforms, surfaces, and animated USD.

## Tutorial 4: CT segmentation to VTK surfaces

Source: [`tutorial_04_lung_ct_to_vtk.py`](tutorial_04_lung_ct_to_vtk.py)

### Why and command

Later models operate on geometry rather than raw CT intensities. This lesson
exposes the replaceable boundary `CT -> labelmap -> lung surfaces`.

```bash
python tutorials/tutorial_04_lung_ct_to_vtk.py
```

### What executes

1. Read `Case1Pack_T00.mha`.
2. Segment with NV-Segment-CTMR in CT body mode.
3. Preserve the published label IDs in the full labelmap.
4. Clear non-lung labels for a lung/lobe labelmap.
5. Extract a combined lung mesh and per-label surfaces.

NV-Segment-CTMR weights use the NVIDIA OneWay Non-Commercial License. The
first run can download the model into the persistent Hugging Face cache.

### Expected outputs

The validated run produced five lobe labels (`28`-`32`) and a combined mesh
with 271,393 points and 545,114 triangles.

| Output | Inspect with |
|---|---|
| `patient_nvsegmentctmri_labelmap.mha` | 3D Slicer |
| `patient_nvsegmentctmri_lung_labelmap.mha` | 3D Slicer |
| `patient_nvsegmentctmri_lung.vtp` | Trame or 3D Slicer |
| `patient_nvsegmentctmri_lung_*_lobe_*.vtp` | Trame or 3D Slicer |
| `tutorial_04_lung_*png` | Image viewer |

```bash
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_04_lung/patient_nvsegmentctmri_lung_*_lobe_*.vtp
```

The Trame view confirms five surface labels but cannot display CT or labelmap
voxels. A browser-accessible 3D Slicer container remains a future improvement.

Troubleshooting: check the model cache on download failures, expect the
non-commercial license warning, and use Slicer to confirm CT/labelmap/surface
alignment before trusting the mesh. Learning objectives: identify labels,
coordinates, surface resolution, and the three definitions to replace for
another organ: segmenter, anatomy group, and reduction rate.

## Tutorial 6: create a PCA lung shape model

Source:
[`tutorial_06_lung_create_statistical_model.py`](tutorial_06_lung_create_statistical_model.py)

### Why and command

This lesson converts population anatomy into a mean surface plus six
corresponding modes of variation. It models inter-patient shape, not breathing
motion.

```bash
python tutorials/tutorial_06_lung_create_statistical_model.py
```

### What executes

1. Segment the ten DIR-Lab T70 reference phases or reuse cached surfaces.
2. Build an unbiased mean atlas over three iterations or reuse it.
3. Register all surfaces to a shared vertex correspondence.
4. Compute six PCA components and explained variance.
5. Write the PCA JSON, mean surface, and plus/minus mode surfaces.

| Processing step | Cold | Cached | Hardware |
|---|---:|---:|---|
| Segment and contour ten lungs | 9:45 | skipped | GPU model + CPU |
| Build mean atlas | 1:08:08 | skipped | mixed |
| ICP alignment | 2:26 | 2:26 | CPU |
| Deformable correspondence | 16:34 | 16:34 | CPU/Greedy + GPU/ICON |
| PCA and outputs | 0:26 | 0:26 | CPU + rendering |
| **Total** | **1:37:19** | **19:26** | mixed |

GPU use is intermittent because only segmentation and ICON run on CUDA. ICP,
distance-map preparation, Greedy, PCA, and file writing are CPU work.

### Expected result and visualization

The validated 282,764-point model explains 86.70% of cohort variation with six
modes. Mode 1 explains 22.64%; all six explain 22.64%, 43.12%, 58.29%, 70.15%,
79.47%, and 86.70% cumulatively.

```bash
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_06_lung/pca_mode_01_minus_2sigma.vtp \
  tutorials/output/tutorial_06_lung/pca_mean_surface.vtp \
  tutorials/output/tutorial_06_lung/pca_mode_01_plus_2sigma.vtp
```

![PCA mode 1 at minus two sigma, mean, and plus two sigma](../docs/assets/tutorial_06_lung_mode_01_overlay.png)

Troubleshooting: a cold run is intentionally long; fewer modes indicate a
cohort-rank limit; low GPU use outside ICON is expected. Learning objectives:
define mean, correspondence, mode, eigenvalue, coefficient, and cohort limits.

## Tutorial 7: fit the model to one patient

Source:
[`tutorial_07_lung_fit_statistical_model_to_patient.py`](tutorial_07_lung_fit_statistical_model_to_patient.py)

### Why and command

Tutorial 7 converts the population model into one patient-specific anatomical
state: six PCA coefficients plus a final fitted surface.

```bash
python tutorials/tutorial_07_lung_fit_statistical_model_to_patient.py
```

It reads the static `data/Chest-CT/Chest-CT.mha` patient, fits the Tutorial 6
mean with ICP and PCA optimization, then uses Greedy plus ICON for remaining
local differences.

| Processing step | Cached time | Hardware |
|---|---:|---|
| Read CT, PCA, and cached anatomy | 0:11.5 | CPU + I/O |
| ICP | 0:17.7 | CPU |
| PCA optimization | 0:31.4 | CPU |
| Distance maps and masks | 0:55.0 | CPU |
| Greedy | 1:45.3 | CPU |
| ICON and composition | 2:17.1 | CUDA + CPU |
| Apply transforms and write | 0:05.1 | CPU |
| **Total** | **6:04.2** | mixed |

The validated coefficients were
`[-0.940, -0.509, 0.033, -0.903, -0.612, 0.082]`.

```bash
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_07_lung/tutorial_07_lung_lung_surface.vtp \
  tutorials/output/tutorial_07_lung/tutorial_07_lung_template_surface_registered.vtp
```

![Patient target and fitted model in Trame](../docs/assets/tutorial_07_lung_trame_viewer.png)

Key outputs are the patient labelmap and surface, registered coefficients JSON,
PCA-fitted template, final registered template, and before/after PNGs.
Troubleshooting: restore Chest-CT or PCA files when missing; intermittent GPU
use is expected; inspect target/fitted overlap before trusting coefficients.
Learning objectives: interpret standardized coefficients and distinguish the
PCA fit from residual deformation.

## Tutorial 8: propagate the patient model through 4D

Source:
[`tutorial_08_lung_fit_model_to_4d_patients.py`](tutorial_08_lung_fit_model_to_4d_patients.py)

### Why and course mode

This is the expensive supervision-generation step. It fits each patient's T70
reference anatomy, registers the remaining breathing phases, and applies those
transforms to the corresponding reference mesh. Inspect prepared results during
the course; regenerate offline only when needed:

```bash
python tutorials/tutorial_08_lung_fit_model_to_4d_patients.py
```

Each `*_T??_ssm_surface.vtp` stores all 282,764 vertex coordinates at one
phase with unchanged connectivity. Vertex `i` therefore identifies the same
anatomical location in every phase.

### Why it is expensive

- Ten patients with ten phases require 90 non-identity 3D registrations.
- Each solve repeatedly samples full CT volumes and optimizes affine and dense
  deformation parameters across multiple resolution levels.
- Reference fitting adds segmentation, ICP, PCA optimization, distance maps,
  Greedy, and ICON.
- Resampling CTs and writing dense forward/inverse transforms adds CPU and I/O.

For cached Case 1, reference fitting took about 4:47, phase registration 4:47,
and applying/writing the meshes only 0:29. Optimization, not vertex storage,
dominates runtime. Across all cases, dense transforms account for most of the
approximately 97 GiB full output.

Inspect the prepared reference and one phase:

```bash
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_08_lung/Case2Pack/Case2Pack_ssm_surface.vtp \
  tutorials/output/tutorial_08_lung/Case2Pack/Case2Pack_T00_ssm_surface.vtp
```

Tutorial 9 converts the phase coordinates into displacement targets. Tutorial
10 later supplies the lean whole-cycle animated viewer. Troubleshooting:
missing phases usually mean an interrupted registration; run one case at a
time under memory pressure and preserve transform timestamps when copying
caches. Learning objectives: explain correspondence, supervision generation,
and why offline registration can be replaced by a learned forward pass.

## Tutorial 9: check the PhysicsNeMo surrogate

Source:
[`tutorial_09_lung_train_physicsnemo_mgn.py`](tutorial_09_lung_train_physicsnemo_mgn.py)

### Why and command

MeshGraphNet learns a map from patient shape and respiratory stage to
per-vertex displacement. The course verifies the training wiring without
creating a new model:

```bash
python tutorials/tutorial_09_lung_train_physicsnemo_mgn.py --smoke-test
```

Node features are mean X/Y/Z, six patient PCA coefficients, and stage. Edge
features are relative XYZ and distance. The target is displacement XYZ at each
vertex.

Smoke mode loads prepared manifests and the supplied checkpoint, builds one
real graph batch, performs forward and backward propagation, verifies the
checkpoint SHA-256, and exits before optimizer creation, saving, or held-out
evaluation.

| Result | Validated value |
|---|---:|
| Training graphs available | 90 |
| Batch size | 1 graph |
| Forward/backward time | 6.73 s |
| Loss | 0.015115 |
| Peak allocated GPU memory | 16.88 GiB |
| Checkpoint modified | No |

Inspect a reference and one displacement target:

```bash
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_08_lung/Case2Pack/Case2Pack_ssm_surface.vtp \
  tutorials/output/tutorial_09_lung_mgn/manifests_mgn/Case2Pack_T00_ssm_surface_target.vtp
```

Troubleshooting: restore manifests and checkpoint before smoke mode; use the
Docker image for PhysicsNeMo dependencies; expect about 17 GiB for the measured
batch; stop and add `--smoke-test` if epoch training begins. Learning
objectives: formulate graph inputs/targets, explain patient-level holdout, and
distinguish a smoke test from convergence.

## Tutorial 10: predict motion

Source:
[`tutorial_10_lung_infer_physicsnemo_mgn.py`](tutorial_10_lung_infer_physicsnemo_mgn.py)

### Why and command

The frozen surrogate predicts ten breathing stages for `Case1Pack`, which was
excluded from training:

```bash
python tutorials/tutorial_10_lung_infer_physicsnemo_mgn.py
```

The checkpoint expects 282,782 vertices, while the regenerated Tutorial 8 mesh
has 282,764 and a different ordering and PCA basis. The script therefore fits
the held-out anatomy to the checkpoint-bundled PCA domain and propagates the
existing Tutorial 8 transforms onto that compatible mesh. The result is cached
under `tutorials/output/tutorial_08_lung_checkpoint/Case1Pack/`. The topology
guard must not be weakened or bypassed.

### Execution and timing

1. Validate or build the compatible patient cache.
2. Load the final checkpoint, graph, PCA assets, and normalization.
3. Predict displacement for stages 0.0 through 0.9.
4. Score predictions against registered phase surfaces.
5. Rasterize and smooth motion to warp the T70 CT.
6. Write predicted VTPs, warped CTs, metrics, screenshots, and animated USD.

| Cached step | Time | Hardware |
|---|---:|---|
| Cache/model/graph load | about 4 s | CPU I/O + CUDA |
| Predict, score, and warp ten stages | about 31 s | CUDA bursts + CPU/ITK |
| Build animated USD | about 7 s | CPU + I/O |
| **Total** | **about 42 s** | mixed |

Mean vertex error averaged 1.043 mm across ten stages; per-stage means ranged
from 0.241 mm at T70 to 1.328 mm at T30.

```bash
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_10_lung_mgn/Case1Pack/Case1Pack_mgn_motion.usd
```

### Prediction versus ground truth

Build a synchronized red/green comparison inside the tutorial shell:

```bash
python utils/create_motion_comparison_usd.py \
  --predicted tutorials/output/tutorial_10_lung_mgn/Case1Pack/Case1Pack_ssm_pca_coefficients_s???_pred.vtp \
  --ground-truth tutorials/output/tutorial_08_lung_checkpoint/Case1Pack/Case1Pack_T??_ssm_surface.vtp \
  --output tutorials/output/tutorial_10_lung_mgn/Case1Pack/Case1Pack_prediction_vs_ground_truth.usd \
  --fps 3
```

Keep the composite USD and its two referenced layers together. From a second
host terminal, launch it with:

```bash
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_10_lung_mgn/Case1Pack/Case1Pack_prediction_vs_ground_truth.usd
```

Red is prediction; green is registered ground truth.

Troubleshooting: a topology error means an incompatible regenerated mesh was
passed directly; a long fit means the compatible cache is missing; low average
GPU use is expected because warping and I/O are CPU work. Learning objectives:
distinguish inference, registration-derived targets, image warping, acquired
stage scoring, and continuous-stage interpolation.

## Tutorial 11: evaluate against images

Source:
[`tutorial_11_lung_evaluate_physicsnemo.py`](tutorial_11_lung_evaluate_physicsnemo.py)

### Why and command

Tutorial 11 compares predicted lobe motion with independently segmented CT
phases rather than only registration-derived vertices:

```bash
python tutorials/tutorial_11_lung_evaluate_physicsnemo.py
```

The script reuses the compatible Case 1 fit, predicts ten stages, carries the
T70 reference labels through the predicted deformation, resamples acquired and
predicted labels to a 2 mm grid, and scores five lobes at ten stages.

| Cached step | Time | Hardware |
|---|---:|---|
| Startup, cache, and model load | about 19 s | CPU I/O + CUDA load |
| Predict and warp ten stages | about 24 s | CUDA bursts + CPU/ITK |
| Extract surfaces and compute 50 rows | about 68 s | CPU/VTK |
| Reports and screenshots | about 1 s | CPU |
| **Total** | **1:52** | mixed |

Across nine non-reference phases and five lobes, the validated mean surface
RMSE was 1.065 mm and mean absolute lobe-volume error was 1.661%. T70 can score
zero on the 2 mm label grid even though Tutorial 10 reports a small vertex
error; that is grid quantization, not proof of perfect prediction.

Key outputs under `tutorials/output/tutorial_11_lung/Case1Pack/` are
`evaluation_report.md`, `evaluation_metrics.csv`, `volume_vs_stage.png`, ten
predicted VTPs, ten warped labelmaps, and the cached independent labelmaps.

```bash
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_11_lung/Case1Pack/Case1Pack_ssm_pca_coefficients_s020_pred.vtp
```

Use 3D Slicer to compare a warped MHA labelmap with its independently segmented
NIfTI labelmap. Troubleshooting: a missing ground-truth cache triggers ten
segmentations; tiny unrelated labels can be ignored if all five lobes are
scored; changing grid spacing or smoothing changes the reported metrics.
Learning objectives: distinguish image-backed validation from surface-target
checks and interpret signed volume bias and symmetric surface RMSE.

## Course conclusion

The workflow turns gated CT into corresponding surfaces, learns a
patient-conditioned graph surrogate, predicts held-out breathing motion, and
tests it against images. An animated model demonstrates pipeline output; it is
not evidence of physiological validity without the Tutorial 11 measurements
and task-specific validation design.
