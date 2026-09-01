# Lung Tutorial Commands

Run setup commands from the repository root. Run Python commands inside
`docker/tutorial-shell.sh`; run viewer commands from a second host terminal.
Stop a viewer with `Ctrl+C` before reusing port 8080.

## Setup

```bash
docker build -t monai-physio:tutorials .
read -rsp "Hugging Face token: " HF_TOKEN && echo
export HF_TOKEN
./docker/download-lung-bundles.sh
unset HF_TOKEN
./docker/tutorial-shell.sh
```

## Tutorial 1: gated CT to animated USD

Segments one phase, registers the breathing cycle, and writes animated USD.

```bash
# Tutorial shell
python tutorials/tutorial_01_lung_gated_ct_to_usd.py

# Host
./docker/view-meshes.sh --port 8080 --fps 3 \
  tutorials/output/tutorial_01_lung/lung_model.all_painted.usd
```

## Tutorial 4: CT to lung surfaces

Segments one CT and writes lung labelmaps and per-lobe VTP surfaces.

```bash
# Tutorial shell
python tutorials/tutorial_04_lung_ct_to_vtk.py

# Host
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_04_lung/patient_nvsegmentctmri_lung_*_lobe_*.vtp
```

## Tutorial 6: PCA shape model

Builds the population lung model. A full regeneration is intentionally slow.

```bash
# Tutorial shell
python tutorials/tutorial_06_lung_create_statistical_model.py

# Host
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_06_lung/pca_mode_01_minus_2sigma.vtp \
  tutorials/output/tutorial_06_lung/pca_mean_surface.vtp \
  tutorials/output/tutorial_06_lung/pca_mode_01_plus_2sigma.vtp
```

## Tutorial 7: fit one patient

Fits the PCA model to the Chest-CT patient and writes the fitted surface.

```bash
# Tutorial shell
python tutorials/tutorial_07_lung_fit_statistical_model_to_patient.py

# Host
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_07_lung/tutorial_07_lung_lung_surface.vtp \
  tutorials/output/tutorial_07_lung/tutorial_07_lung_template_surface_registered.vtp
```

## Tutorial 8: propagate through 4D

Creates corresponding phase surfaces used as surrogate targets. Prefer the
prepared outputs during the workshop; full regeneration is expensive.

```bash
# Tutorial shell
python tutorials/tutorial_08_lung_fit_model_to_4d_patients.py

# Host
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_08_lung/Case2Pack/Case2Pack_ssm_surface.vtp \
  tutorials/output/tutorial_08_lung/Case2Pack/Case2Pack_T00_ssm_surface.vtp
```

## Tutorial 9: train the surrogate

The workshop uses the supplied checkpoint. Run this only to train a replacement
model; the full run takes roughly four hours on the reference GPU.

```bash
# Tutorial shell
python tutorials/tutorial_09_lung_train_physicsnemo_mgn.py
```

## Tutorial 10: predict motion

Predicts ten stages for held-out Case 1 and writes surfaces, warped CTs, and
animated USD.

```bash
# Tutorial shell
python tutorials/tutorial_10_lung_infer_physicsnemo_mgn.py

# Host
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_10_lung_mgn/Case1Pack/Case1Pack_mgn_motion.usd
```

Create the optional prediction-versus-ground-truth animation inside the shell:

```bash
python utils/create_motion_comparison_usd.py \
  --predicted tutorials/output/tutorial_10_lung_mgn/Case1Pack/Case1Pack_ssm_pca_coefficients_s???_pred.vtp \
  --ground-truth tutorials/output/tutorial_08_lung/Case1Pack/Case1Pack_T??_ssm_surface.vtp \
  --output tutorials/output/tutorial_10_lung_mgn/Case1Pack/Case1Pack_prediction_vs_ground_truth.usd \
  --fps 3
```

```bash
# Host
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_10_lung_mgn/Case1Pack/Case1Pack_prediction_vs_ground_truth.usd
```

## Tutorial 11: evaluate motion

Scores predicted lobe motion against independently segmented CT phases.

```bash
# Tutorial shell
python tutorials/tutorial_11_lung_evaluate_physicsnemo.py

# Host
./docker/view-meshes.sh --port 8080 \
  tutorials/output/tutorial_11_lung/Case1Pack/Case1Pack_ssm_pca_coefficients_s020_pred.vtp
```

## Remote viewer

Leave the viewer running, forward its port locally, then open
`http://127.0.0.1:8080/index.html`.

```bash
brev port-forward INSTANCE_NAME -p 8080:8080
```

Without Brev CLI:

```bash
ssh -N -L 8080:127.0.0.1:8080 USER@REMOTE_HOST
```
