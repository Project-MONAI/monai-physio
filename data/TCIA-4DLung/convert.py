"""Convert 4D-Lung DICOM CT series into gated-percentage-named nii.gz files.

Reads DICOM series from download/???_HM10395/**/CT_* and writes
???_HM10395/???_HM10395_gYYY.nii.gz alongside the download directory,
where YYY is the zero-padded gated percentage parsed from the DICOM
series description (e.g. "Gated, 70.0%" -> g070).

Does not modify or add anything inside the download directory.
Does not keep any .json sidecar files.
"""

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(ROOT, "download")
DCM2NIIX = r"C:\src\dcm2niix.exe"

GATED_RE = re.compile(r"Gated,?[_\s]*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)


def find_ct_series_dirs(patient_dir):
    series_dirs = []
    for dirpath, dirnames, _filenames in os.walk(patient_dir):
        for d in dirnames:
            if d.startswith("CT_"):
                series_dirs.append(os.path.join(dirpath, d))
    return sorted(series_dirs)


def variant_patient_name(patient_name, index):
    """index 0 -> unchanged name, 1 -> 'a' suffix on the number, 2 -> 'b', ..."""
    if index == 0:
        return patient_name
    num_part, rest = patient_name.split("_", 1)
    letter = chr(ord("a") + index - 1)
    return f"{num_part}{letter}_{rest}"


def convert_series(series_dir, root, patient_name, pct_counts):
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = subprocess.run(
            [DCM2NIIX, "-f", "%d", "-b", "n", "-z", "y", "-o", tmp_dir, series_dir],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  FAILED: {series_dir}\n{result.stderr}")
            return

        produced = glob.glob(os.path.join(tmp_dir, "*.nii.gz"))
        if not produced:
            print(f"  WARNING: no nii.gz produced for {series_dir}")
            return

        for nii_path in produced:
            base = os.path.basename(nii_path)
            match = GATED_RE.search(base)
            if not match:
                print(
                    f"  WARNING: could not parse gated percentage from '{base}' ({series_dir})"
                )
                continue
            pct = round(float(match.group(1)))

            index = pct_counts.get(pct, 0)
            pct_counts[pct] = index + 1
            variant_name = variant_patient_name(patient_name, index)

            out_dir = os.path.join(root, variant_name)
            os.makedirs(out_dir, exist_ok=True)

            dest_name = f"{variant_name}_g{pct:03d}.nii.gz"
            dest_path = os.path.join(out_dir, dest_name)
            shutil.move(nii_path, dest_path)
            print(f"  {series_dir} -> {variant_name}/{dest_name}")

        # tmp_dir cleaned up automatically, dropping any stray .json sidecars


def main():
    if not os.path.isfile(DCM2NIIX):
        print(f"dcm2niix not found at {DCM2NIIX}")
        sys.exit(1)

    patient_dirs = sorted(
        d
        for d in glob.glob(os.path.join(DOWNLOAD_DIR, "*_HM10395"))
        if os.path.isdir(d)
    )

    for patient_dir in patient_dirs:
        patient_name = os.path.basename(patient_dir)

        series_dirs = find_ct_series_dirs(patient_dir)
        print(f"{patient_name}: {len(series_dirs)} CT series found")

        pct_counts = {}
        for series_dir in series_dirs:
            convert_series(series_dir, ROOT, patient_name, pct_counts)

    print("Done.")


if __name__ == "__main__":
    main()
