"""
MONAI Physio - Methods, workflows, tutorials, and CLI for creating
personalized physiological digital twins.

Starting from a 3D medical image of a subject, this package extracts anatomic
models and then uses AI surrogates to estimate the subject's physiological
processes, initially cardiac and respiratory motion and expanding to
electrophysiology, blood flow, and organ perfusion. It provides methods for
forming those AI surrogates and for finetuning the segmentation and
registration AI methods that power them, with statistical shape models used
to capture subject-specific characteristics and establish correspondence
across subjects.

Main Components:
    - WorkflowConvertImageToUSD: 4D CT image to USD workflow
    - Segmentation classes: Multiple AI-based chest segmentation implementations
    - Registration tools: Deep learning-based image registration
    - Transform utilities: Tools for image and contour transformations
    - USD tools: Utilities for Omniverse integration
    - MONAIPhysioBase: Base class with standardized logging and debug settings
"""

__version__ = "2026.08.0"

import importlib.metadata as _importlib_metadata
import importlib.util as _importlib_util
import warnings as _warnings

if _importlib_util.find_spec("cupy") is None:
    _warnings.warn(
        "CuPy is not installed - GPU-accelerated mesh operations will fall "
        "back to NumPy and run significantly slower. Every workflow still "
        "runs. Re-install with uv to get CuPy and CUDA-enabled PyTorch in one "
        "step (pip alone will not select the correct CUDA wheel):\n"
        '  uv pip install "monai-physio[cuda12]"  # CUDA 12.6\n'
        '  uv pip install "monai-physio[cuda13]"  # CUDA 13',
        UserWarning,
        stacklevel=2,
    )

try:
    _installer = _importlib_metadata.distribution("monai-physio").read_text("INSTALLER")
except _importlib_metadata.PackageNotFoundError:
    _installer = None
if _installer is not None and _installer.strip() == "pip":
    _warnings.warn(
        "monai-physio was installed with pip, which selects the PyTorch CUDA "
        "wheel manually and needs a separate PyTorch install step (see the "
        "installation guide). uv does it in one command:\n"
        '  uv pip install "monai-physio[cuda12]"',
        UserWarning,
        stacklevel=2,
    )

from .anatomy_taxonomy import AnatomyGroup, AnatomyTaxonomy
from .contour_tools import ContourTools

# Data processing utilities
from .convert_image_4d_to_3d import ConvertImage4DTo3D
from .convert_vtk_to_usd import ConvertVTKToUSD
from .data_download_tools import DataDownloadTools
from .evaluate_movement_base import EvaluateMovementBase, MovementGroundTruth
from .evaluate_movement_duke_heart import EvaluateMovementDukeHeart
from .evaluate_movement_lung import EvaluateMovementLung

# Utility classes
from .image_tools import ImageTools
from .infer_physicsnemo_base import InferPhysicsNeMoBase
from .infer_physicsnemo_mgn import InferPhysicsNeMoMGN
from .infer_physicsnemo_mlp import InferPhysicsNeMoMLP
from .labelmap_tools import LabelmapTools
from .landmark_tools import LandmarkTools

# Base classes
from .monai_physio_base import MONAIPhysioBase
from .physicsnemo_tools import DistributedContext, distributed_context
from .register_images_ants import RegisterImagesANTS

# Registration classes
from .register_images_base import RegisterImagesBase
from .register_images_chain import RegisterImagesChain
from .register_images_greedy import RegisterImagesGreedy
from .register_images_greedy_icon import RegisterImagesGreedyICON
from .register_images_icon import RegisterImagesICON
from .register_models_distance_maps import RegisterModelsDistanceMaps
from .register_models_icp import RegisterModelsICP
from .register_models_icp_itk import RegisterModelsICPITK
from .register_models_pca import RegisterModelsPCA
from .register_time_series_images import RegisterTimeSeriesImages
from .report_evaluate_movement import ReportEvaluateMovement

# Segmentation classes
from .segment_anatomy_base import SegmentAnatomyBase
from .segment_chest_total_segmentator import SegmentChestTotalSegmentator
from .segment_chest_total_segmentator_with_contrast import (
    SegmentChestTotalSegmentatorWithContrast,
)
from .segment_heart_simpleware import SegmentHeartSimpleware
from .segment_heart_simpleware_trimmed_branches import (
    SegmentHeartSimplewareTrimmedBranches,
)
from .segment_nv_segment_ct_mri import SegmentNVSegmentCTMRI
from .test_tools import TestTools
from .train_physicsnemo_base import TrainPhysicsNeMoBase
from .train_physicsnemo_mgn import TrainPhysicsNeMoMGN
from .train_physicsnemo_mlp import TrainPhysicsNeMoMLP
from .train_physicsnemo_physics_informed_motion import (
    TrainPhysicsNeMoPhysicsInformedMotion,
)
from .transform_tools import TransformTools
from .usd_anatomy_tools import USDAnatomyTools
from .usd_tools import USDTools
from .workflow_convert_image_to_usd import WorkflowConvertImageToUSD

# Core workflow processor
from .workflow_convert_image_to_vtk import WorkflowConvertImageToVTK
from .workflow_convert_vtk_to_usd import WorkflowConvertVTKToUSD
from .workflow_create_mean_surface import WorkflowCreateMeanSurface
from .workflow_create_statistical_model import WorkflowCreateStatisticalModel
from .workflow_evaluate_movement import WorkflowEvaluateMovement
from .workflow_finetune_icon_registration import WorkflowFinetuneICONRegistration
from .workflow_fit_statistical_model_to_patient import (
    WorkflowFitStatisticalModelToPatient,
)
from .workflow_infer_movement import WorkflowInferMovement
from .workflow_infer_physicsnemo import WorkflowInferPhysicsNeMo
from .workflow_reconstruct_highres_4d_ct import WorkflowReconstructHighres4DCT
from .workflow_train_physicsnemo import WorkflowTrainPhysicsNeMo

__all__ = [
    "AnatomyGroup",
    # Anatomy taxonomy (shared between segmenters and USD renderer)
    "AnatomyTaxonomy",
    "ContourTools",
    # Data processing utilities
    "ConvertImage4DTo3D",
    "ConvertVTKToUSD",
    "DataDownloadTools",
    # Distributed execution
    "DistributedContext",
    "EvaluateMovementBase",
    "EvaluateMovementDukeHeart",
    "EvaluateMovementLung",
    # Utility classes
    "ImageTools",
    # Inference method classes
    "InferPhysicsNeMoBase",
    "InferPhysicsNeMoMGN",
    "InferPhysicsNeMoMLP",
    "LabelmapTools",
    "LandmarkTools",
    # Base classes
    "MONAIPhysioBase",
    "MovementGroundTruth",
    "RegisterImagesANTS",
    # Registration classes
    "RegisterImagesBase",
    "RegisterImagesChain",
    "RegisterImagesGreedy",
    "RegisterImagesGreedyICON",
    "RegisterImagesICON",
    "RegisterModelsDistanceMaps",
    "RegisterModelsICP",
    "RegisterModelsICPITK",
    "RegisterModelsPCA",
    "RegisterTimeSeriesImages",
    "ReportEvaluateMovement",
    # Segmentation classes
    "SegmentAnatomyBase",
    "SegmentChestTotalSegmentator",
    "SegmentChestTotalSegmentatorWithContrast",
    "SegmentHeartSimpleware",
    "SegmentHeartSimplewareTrimmedBranches",
    "SegmentNVSegmentCTMRI",
    "TestTools",
    # Training method classes
    "TrainPhysicsNeMoBase",
    "TrainPhysicsNeMoMGN",
    "TrainPhysicsNeMoMLP",
    "TrainPhysicsNeMoPhysicsInformedMotion",
    "TransformTools",
    "USDAnatomyTools",
    "USDTools",
    "WorkflowConvertImageToUSD",
    # Workflow classes
    "WorkflowConvertImageToVTK",
    "WorkflowConvertVTKToUSD",
    "WorkflowCreateMeanSurface",
    "WorkflowCreateStatisticalModel",
    "WorkflowEvaluateMovement",
    "WorkflowFinetuneICONRegistration",
    "WorkflowFitStatisticalModelToPatient",
    "WorkflowInferMovement",
    "WorkflowInferPhysicsNeMo",
    "WorkflowReconstructHighres4DCT",
    "WorkflowTrainPhysicsNeMo",
    "distributed_context",
]
