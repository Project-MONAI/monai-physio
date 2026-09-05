# %% [markdown]
# # Finetune uniGradICON on Duke 4D Gated CT Data
#
# Discovers per-patient gated CT images and their precomputed
# SegmentHeartSimpleware labelmaps and applies the project-wide fixed 80/20
# train/test split (sort patients in ``ref_data_dir`` by filename; the first
# 80% are train, the last 20% are test).  The train cohort is handed to
# :class:`WorkflowFinetuneICONRegistration`, which builds the paired dataset
# JSON, YAML config, and derived loss-function masks, then launches
# ``unigradicon.finetuning.finetune`` as a subprocess.
#
# ``2-recon_4d_icon_eval.py`` re-derives the same split from the same sorted
# patient list - no cached split file is needed.
#
# Each patient directory under ``src_data_dir_base`` is one ``subject_id``;
# all of that patient's gated time-point frames form a paired training group.
# Frames whose labelmap is missing on disk are dropped from the dataset.
#
# In addition to the original ``gated_nii`` frames, each patient's training
# group is augmented with that patient's ANTS- and Greedy-init frames
# written by ``1-initial_registration.py`` (init image + labelmap per gated
# frame, under ``output_dir / <method> / <patient_id>``).  Because the init
# frames are merged into the *same* ``subject_id`` group, uniGradICON pairs the
# original gated frames and both backends' pre-registered frames together.

# %%
import os
from pathlib import Path
from typing import Optional

from monai_physio import WorkflowFinetuneICONRegistration

# %% [markdown]
# ## 1. Configure data, output locations, and the train/test split

# %%
ref_data_dir = Path("d:/MONAI-Physio/duke_data/ref_images")
src_data_dir_base = Path("d:/MONAI-Physio/duke_data/gated_nii")
labelmap_dir_base = Path("d:/MONAI-Physio/duke_data/simple_ascardio")

# Where the workflow writes the dataset JSON, YAML config, derived masks, and
# the uniGradICON ``checkpoints/`` tree.  experiment_dir resolves to
# ``output_dir / finetune_name``.
_HERE = Path(__file__).parent
output_dir = _HERE / "results_finetuning"
finetune_name = "icon_finetuning"

# Pre-registration augmentation: ``1-initial_registration.py`` warps every gated
# moving frame into reference space with these backends and writes the init
# image + labelmap under ``initial_registration_dir / <method>.lower() /
# <patient_id>``.  Those init frames are merged into each patient's training
# group below (section 4b).
initial_registration_dirs = [
    Path("d:/MONAI-Physio/duke_data/greedy_registrations/results_l/greedy_40.40.10"),
    Path("d:/MONAI-Physio/duke_data/greedy_registrations/results_raw/greedy_80.40.5"),
]

# Fixed train/test split: sort patients in ``ref_data_dir`` by filename;
# first 80% are train, last 20% are test.  ``2-recon_4d_icon_eval.py`` applies
# the same rule so the two scripts agree without a cached split record.
train_fraction = 0.8

# Local clone of uniGradICON (feat-add-finetuning branch) - prepended to
# PYTHONPATH so the subprocess picks up the local source instead of the
# installed package.  Set to ``None`` to use the pip-installed unigradicon.
unigradicon_src_path: Optional[Path] = Path(__file__).parent / "uniGradICON" / "src"

# %% [markdown]
# ## 2. Enumerate patients and apply the fixed 80/20 split
#
# Sort ``ref_data_dir`` by filename to produce the canonical patient order.
# The first 80% become the train cohort; the last 20% are the held-out test
# cohort that ``2-recon_4d_icon_eval.py`` will evaluate.

# %%
ref_files = sorted(
    p
    for p in ref_data_dir.iterdir()
    if p.name.startswith("pm00") and p.suffixes[-2:] == [".nii", ".gz"]
)
all_patient_ids = [p.name[:6] for p in ref_files]
print(f"Found {len(all_patient_ids)} patients under {ref_data_dir}")

if len(all_patient_ids) < 2:
    raise FileNotFoundError(
        f"Need at least 2 patients to form a train/test split; "
        f"discovered {len(all_patient_ids)} under {ref_data_dir}"
    )

n_train = max(
    1,
    min(len(all_patient_ids) - 1, round(train_fraction * len(all_patient_ids))),
)
train_subjects = all_patient_ids[:n_train]
test_subjects = all_patient_ids[n_train:]
print(f"  Train (first {n_train}): {train_subjects}")
print(f"  Test  (last {len(test_subjects)}): {test_subjects}")

# %% [markdown]
# ## 3. Gather the train cohort's gated frames and labelmaps and masks
#
# For each train-cohort patient, list gated frames in
# ``src_data_dir_base / <patient_id>`` (excluding ``"nop"`` non-gated
# references) and pair each frame with its
# ``<stem>_labelmap.nii.gz`` and ``<stem>_mask.nii.gz`` under ``labelmap_dir_base / <patient_id>``.
# Patients with no source directory or no valid frames are skipped here only
# - they remain part of the canonical train list above, but contribute no
# training data.  Missing labelmaps are recorded as ``None`` so the workflow
# skips just that frame.

# %%
train_image_files: list[list[Path]] = []
train_labelmap_files: list[list[Optional[Path]]] = []
train_mask_files: list[list[Optional[Path]]] = []
valid_train_subjects: list[str] = []


# %%
def gather_init_frames(
    initial_registration_dir: Path,
    patient_id: str,
) -> tuple[list[Path], list[Optional[Path]], list[Optional[Path]]]:
    """Return ``(init_image_paths, init_labelmap_paths, init_mask_paths)`` for one
    ``initial_registration_dir / <patient_id>`` directory.

    Enumerates the init moving images (``<stem>.mha``), excluding the
    ``_labelmap.mha`` and ``_mask.mha``
    companions, and pairs each with its ``<stem>_labelmap.mha`` (``None`` when
    that labelmap is absent).  Returns empty lists when
    ``initial_registration_dir`` does not exist.
    """
    if not initial_registration_dir.is_dir():
        print(f"  {patient_id}: registration dir {initial_registration_dir} not found")
        return [], [], []
    companion_suffixes = (
        "_labelmap.mha",
        "_mask.mha",
    )
    image_paths: list[Path] = []
    labelmap_paths: list[Optional[Path]] = []
    mask_paths: list[Optional[Path]] = []
    for image in sorted(initial_registration_dir.glob("*.mha")):
        if image.name.endswith(companion_suffixes):
            continue
        stem = image.name[:-4]
        labelmap = initial_registration_dir / f"{stem}_labelmap.mha"
        mask = initial_registration_dir / f"{stem}_mask.mha"
        if not image.exists() or not labelmap.exists() or not mask.exists():
            print(
                f"  {patient_id}: image {image} or labelmap {labelmap} or mask {mask} not found"
            )
            continue
        image_paths.append(image)
        labelmap_paths.append(labelmap)
        mask_paths.append(mask)
    print(
        f"  {patient_id}: {len(image_paths)} init frames, {len(labelmap_paths)} with labelmap, {len(mask_paths)} with mask"
    )
    return image_paths, labelmap_paths, mask_paths


# %%
for patient_id in train_subjects:
    src_dir = src_data_dir_base / patient_id
    seg_dir = labelmap_dir_base / patient_id

    if not src_dir.is_dir():
        print(f"  Skipping {patient_id}: source dir {src_dir} not found")
        continue

    frame_names = sorted(
        f for f in os.listdir(src_dir) if "nop" not in f and f.endswith(".nii.gz")
    )
    if not frame_names:
        print(f"  Skipping {patient_id}: no valid frames in {src_dir}")
        continue

    image_paths = [src_dir / f for f in frame_names]
    labelmap_paths: list[Optional[Path]] = []
    mask_paths: list[Optional[Path]] = []
    for f in frame_names:
        labelmap = seg_dir / f.replace(".nii.gz", "_labelmap.nii.gz")
        labelmap_paths.append(labelmap if labelmap.exists() else None)
        mask = seg_dir / f.replace(".nii.gz", "_mask.nii.gz")
        mask_paths.append(mask)

    train_image_files.append(image_paths)
    train_labelmap_files.append(labelmap_paths)
    train_mask_files.append(mask_paths)
    valid_train_subjects.append(patient_id)

    n_seg = sum(1 for s in labelmap_paths if s is not None)
    print(f"  {patient_id}: {len(image_paths)} frames, {n_seg} with labelmap")


for subject_index, patient_id in enumerate(valid_train_subjects):
    for initial_registration_dir in initial_registration_dirs:
        initial_registration_patient_dir = initial_registration_dir / patient_id

        init_images, init_labelmaps, init_masks = gather_init_frames(
            initial_registration_patient_dir, patient_id
        )
        if not init_images:
            print(
                f"  {patient_id}: no initial-registered frames "
                f"in {initial_registration_dir}"
            )
            continue
        train_image_files[subject_index].extend(init_images)
        train_labelmap_files[subject_index].extend(init_labelmaps)
        train_mask_files[subject_index].extend(init_masks)

# %%
workflow = WorkflowFinetuneICONRegistration(
    subject_image_files=[
        [str(image_path) for image_path in image_paths]
        for image_paths in train_image_files
    ],
    output_dir=output_dir,
    finetune_name=finetune_name,
    subject_ids=valid_train_subjects,
    subject_labelmap_files=[
        [
            str(labelmap_path) if labelmap_path is not None else None
            for labelmap_path in labelmap_paths
        ]
        for labelmap_paths in train_labelmap_files
    ],
    subject_mask_files=[
        [str(mask_path) if mask_path is not None else None for mask_path in mask_paths]
        for mask_paths in train_mask_files
    ],
    unigradicon_src_path=unigradicon_src_path,
    epochs=500,
)

weights_path = workflow.process()
print(f"\nFinetuning complete. Expected weights at: {weights_path}")
print(f"Held-out test cohort (for 2-recon_4d_icon_eval.py): {test_subjects}")
