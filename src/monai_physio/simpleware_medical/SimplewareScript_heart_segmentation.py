"""Simpleware Medical Python script for heart segmentation using ASCardio.

This script is designed to run within the Synopsys Simpleware Medical
environment. It uses the ASCardio module to perform automated heart
segmentation on CT images. The script is called as an external process
from the MONAI Physio SegmentHeartSimpleware class.
"""

import os

import simpleware.scripting as sw

app = sw.App.GetInstance()
doc = sw.App.GetDocument()

output_dir = app.GetInputValue()

as_cardio = doc.GetAutoSegmenters().GetASCardio()

parts = sw.HeartParts(
    as_cardio.RightAtrium,
    as_cardio.LeftAtrium,
    as_cardio.RightVentricle,
    as_cardio.LeftVentricle,
    as_cardio.Myocardium,
    as_cardio.Aorta,
    as_cardio.LeftCoronaryArtery,
    as_cardio.RightCoronaryArtery,
    as_cardio.PulmonaryArtery,
)
bounds = as_cardio.CalculateHeartCTRegionOfInterest(parts)

as_cardio.ApplyHeartCTTool(bounds, parts, True)

for mask in doc.GetMasks():
    mask_name = mask.GetName()
    fixed_name = mask_name.replace(" ", "_").lower()
    mask.MetaImageExport(os.path.join(output_dir, f"mask_{fixed_name}.mhd"))

landmarks = doc.GetMeasurements()
landmarks.Export(os.path.join(output_dir, "landmarks.csv"))
