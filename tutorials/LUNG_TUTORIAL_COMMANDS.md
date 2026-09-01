# Lung Tutorial Commands

## Course setup

Run the setup commands from the repository root on the GPU host. They check the
host, build the tutorial image, install the prepared lung bundles, and open the
tutorial shell.

```bash
docker/check-host.sh
docker build -t monai-physio:tutorials .
(
  read -rsp "Hugging Face token: " hf_token
  printf "\n"
  HF_TOKEN="$hf_token" docker/download-lung-bundles.sh
)
docker/tutorial-shell.sh
```

Run each processing command inside the tutorial shell. Run its visualization
command from a second terminal on the GPU host, and stop the viewer with
`Ctrl+C` before reusing port 8080.

## Tutorial 1: gated CT to animated USD

Segments one reference CT, registers all ten respiratory phases, and produces
registered images, transforms, surfaces, and an animated USD.

### Process

```bash
python tutorials/tutorial_01_lung_gated_ct_to_usd.py
```

### Visualize the animated registered anatomy

```bash
./docker/view-meshes.sh --port 8080 --fps 3 \
  tutorials/output/tutorial_01_lung/lung_model.all_painted.usd
```

## Tutorial 4: CT to lung surfaces

Segments one CT and produces lung/lobe labelmaps, VTP surfaces, and validation
images.

### Process

```bash
python tutorials/tutorial_04_lung_ct_to_vtk.py
```

### Visualize the lung lobe surfaces

```bash
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_04_lung/patient_nvsegmentctmri_lung_*_lobe_*.vtp
```

## Tutorial 6: PCA lung shape model

Uses the full DIR-Lab population to produce the PCA model, mean surface, and
plus/minus mode surfaces. This is a long, full-data regeneration command.

### Process

```bash
python tutorials/tutorial_06_lung_create_statistical_model.py
```

### Visualize the leading PCA mode

```bash
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_06_lung/pca_mode_01_minus_2sigma.vtp \
  tutorials/output/tutorial_06_lung/pca_mean_surface.vtp \
  tutorials/output/tutorial_06_lung/pca_mode_01_plus_2sigma.vtp
```

## Tutorial 7: fit the model to one patient

Fits the PCA model to the Chest-CT patient and produces fitted surfaces,
coefficients, transforms, and validation images. Rerunning requires the optional
Chest-CT inputs.

### Process

```bash
python tutorials/tutorial_07_lung_fit_statistical_model_to_patient.py
```

### Visualize the patient surface and fitted model

```bash
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_07_lung/tutorial_07_lung_lung_surface.vtp \
  tutorials/output/tutorial_07_lung/tutorial_07_lung_template_surface_registered.vtp
```

## Tutorial 8: propagate the model through 4D

Fits each reference anatomy, registers every respiratory phase, and produces
corresponding phase surfaces and transforms. This is a long, full-data command.

### Process

```bash
python tutorials/tutorial_08_lung_fit_model_to_4d_patients.py
```

### Visualize a reference and propagated phase

```bash
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_08_lung/Case2Pack/Case2Pack_ssm_surface.vtp \
  tutorials/output/tutorial_08_lung/Case2Pack/Case2Pack_T00_ssm_surface.vtp
```

## Tutorial 9: check the PhysicsNeMo surrogate

Loads the prepared training graphs and checkpoint, runs one forward/backward
batch, and reports a pass without saving or modifying the checkpoint.

### Process

```bash
python tutorials/tutorial_09_lung_train_physicsnemo_mgn.py --smoke-test
```

### Visualize a reference and displacement target

```bash
manifest_dir=tutorials/output/tutorial_09_lung_mgn/manifests_mgn
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_08_lung/Case2Pack/Case2Pack_ssm_surface.vtp \
  "$manifest_dir/Case2Pack_T00_ssm_surface_target.vtp"
```

## Tutorial 10: predict lung motion

Runs the frozen surrogate for ten respiratory stages and produces predicted
surfaces, warped CTs, images, and an animated USD.

### Process

```bash
python tutorials/tutorial_10_lung_infer_physicsnemo_mgn.py
```

### Visualize the predicted motion

```bash
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_10_lung_mgn/Case1Pack/Case1Pack_mgn_motion.usd
```

### Build and visualize prediction versus ground truth

Build the comparison USD inside the tutorial shell.

```bash
comparison_dir=tutorials/output/tutorial_10_lung_mgn/Case1Pack
ground_truth_dir=tutorials/output/tutorial_08_lung_checkpoint/Case1Pack
python utils/create_motion_comparison_usd.py \
  --predicted "$comparison_dir"/Case1Pack_ssm_pca_coefficients_s???_pred.vtp \
  --ground-truth "$ground_truth_dir"/Case1Pack_T??_ssm_surface.vtp \
  --output "$comparison_dir/Case1Pack_prediction_vs_ground_truth.usd" \
  --fps 3
```

View the translucent red/green comparison from the host terminal.

```bash
comparison_dir=tutorials/output/tutorial_10_lung_mgn/Case1Pack
./docker/view-meshes.sh --port 8080 \
  "$comparison_dir/Case1Pack_prediction_vs_ground_truth.usd"
```

## Tutorial 11: evaluate against images

Compares predicted lobe motion with independently segmented CT phases and
produces per-lobe metrics, a report, plots, predicted surfaces, and labelmaps.

### Process

```bash
python tutorials/tutorial_11_lung_evaluate_physicsnemo.py
```

### Visualize a predicted evaluation surface

```bash
evaluation_dir=tutorials/output/tutorial_11_lung/Case1Pack
./docker/view-meshes.sh --port 8080 \
  "$evaluation_dir/Case1Pack_ssm_pca_coefficients_s020_pred.vtp"
```

## Optional: forward a remote viewer port

Run from the local machine, then open `http://127.0.0.1:8080/index.html`.

```powershell
ssh -N -L 8080:127.0.0.1:8080 USER@REMOTE_HOST
```
