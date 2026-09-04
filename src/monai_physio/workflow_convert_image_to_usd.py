"""
Image-to-USD workflow implementing the complete 3D/4D image to USD pipeline.

This module implements the complete pipeline for processing 3D or 4D medical
images (e.g. cardiac and respiratory gated CT studies) into dynamic USD
models.  4D image arrays follow the (X, Y, Z, T) axis convention used
throughout MONAI Physio.
"""

import logging
import os
from typing import Optional, Union

import itk
import numpy as np
import pyvista as pv

from .contour_tools import ContourTools
from .convert_vtk_to_usd import ConvertVTKToUSD
from .image_tools import ImageTools
from .monai_physio_base import MONAIPhysioBase
from .register_images_base import RegisterImagesBase
from .register_images_greedy import RegisterImagesGreedy
from .segment_anatomy_base import SegmentAnatomyBase
from .segment_chest_total_segmentator_with_contrast import (
    SegmentChestTotalSegmentatorWithContrast,
)
from .transform_tools import TransformTools
from .usd_anatomy_tools import USDAnatomyTools


class WorkflowConvertImageToUSD(MONAIPhysioBase):
    """
    Complete workflow for converting 4D CT images to dynamic USD models.

    This class implements the full workflow from 4D CT images to painted USD files
    suitable for visualization in NVIDIA Omniverse.

    ``segmentation_method`` and ``registration_method`` accept a
    pre-configured :class:`SegmentAnatomyBase` / :class:`RegisterImagesBase`
    instance. Configure backend-specific parameters (iteration counts,
    trim_branches, mass preservation, etc.) on the instance before passing
    it in. Defaults to :class:`SegmentChestTotalSegmentatorWithContrast` /
    :class:`RegisterImagesGreedy` when omitted.
    """

    def __init__(
        self,
        time_series_images: list[itk.Image],
        reference_image: itk.Image,
        usd_project_name: str,
        output_directory: str,
        segmentation_method: Optional[SegmentAnatomyBase] = None,
        registration_method: Optional[RegisterImagesBase] = None,
        dynamic_labelmap_ids: Optional[list[int]] = None,
        mask_dilation_radius: int = 10,
        frames_per_second: float = 1.0,
        surface_reduction_rate: float = 0.0,
        log_level: int | str = logging.INFO,
        save_assets: bool = True,
    ) -> None:
        """
        Initialize the image-to-USD workflow.

        Args:
            time_series_images (list[itk.Image]): List of time-series images
            reference_image (itk.Image): Reference image
            usd_project_name (str): Project name for USD file organization
            output_directory (str): Directory path where output files will be stored
            segmentation_method (Optional[SegmentAnatomyBase]): Segmentation
                backend instance. Defaults to a new
                :class:`SegmentChestTotalSegmentatorWithContrast` when None.
            registration_method (Optional[RegisterImagesBase]): Registration
                backend instance. Defaults to a new :class:`RegisterImagesGreedy`
                when None. A caller-supplied instance is mutated (fixed
                image/mask/modality) during :meth:`process` - pass a fresh
                instance per run unless intentionally reusing state.
            dynamic_labelmap_ids (Optional[list[int]]): Labelmap ids to treat
                as dynamic anatomy (registered/contoured separately from the
                remaining static anatomy). Defaults to an empty list (no
                dynamic/static split; everything registers as "all").
            mask_dilation_radius (int): Dilation radius, in voxels, applied to
                the dynamic/static registration masks. Defaults to 10.
            frames_per_second: Frames per second for animated USD time series.
                Defaults to 24.0, matching the underlying VTK-to-USD converter.
            surface_reduction_rate: Fraction in ``[0, 1)`` of triangles to
                remove from every extracted surface.  ``0.0`` (default) skips
                decimation.
            log_level: Logging level (default: logging.INFO)
            save_assets: Write registered images, transforms, and labelmaps
                output_directory when True

        Raises:
            TypeError: If segmentation_method is neither None nor a
                SegmentAnatomyBase instance, or registration_method is
                neither None nor a RegisterImagesBase instance.
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

        self.time_series_images = time_series_images
        self.reference_image = reference_image
        self.usd_project_name = usd_project_name
        self.dynamic_labelmap_ids = dynamic_labelmap_ids if dynamic_labelmap_ids else []
        self.output_directory = output_directory
        self.frames_per_second = frames_per_second
        self.surface_reduction_rate = surface_reduction_rate
        self.save_assets = save_assets

        self.registration_results: list[
            dict[str, dict[str, Union[itk.Transform, float]]]
        ] = []
        if segmentation_method is None:
            segmentation_method = SegmentChestTotalSegmentatorWithContrast(
                log_level=log_level
            )
        elif not isinstance(segmentation_method, SegmentAnatomyBase):
            raise TypeError(
                "segmentation_method must be a SegmentAnatomyBase instance or None"
            )
        self.segmenter: SegmentAnatomyBase = segmentation_method

        if registration_method is None:
            registration_method = RegisterImagesGreedy(log_level=log_level)
        elif not isinstance(registration_method, RegisterImagesBase):
            raise TypeError(
                "registration_method must be a RegisterImagesBase instance or None"
            )
        registration_method.set_modality("ct")
        registration_method.set_mask_dilation(0)
        self.mask_dilation_radius = mask_dilation_radius
        self.registrar: RegisterImagesBase = registration_method

        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Initialize processing components
        self.contour_tools = ContourTools()
        self.image_tools = ImageTools()
        self.transform_tools = TransformTools()

        # Data storage for processing pipeline
        self._num_time_points = len(time_series_images)

        self.reference_segmentation_results: Optional[dict[str, itk.Image]] = None
        self.reference_contours: dict[str, pv.PolyData] = {}

        self.transformed_contours: dict[str, list[pv.PolyData]] = {
            "all": [],
            "dynamic": [],
            "static": [],
        }

    def process(self) -> dict[str, str]:
        """
        Execute the complete workflow from 4D CT to dynamic USD models.

        Returns:
            dict[str, str]: Dictionary of anatomy type to filename of the final painted USD file.
        """
        self.log_section("Image-to-USD Processing Pipeline")

        # Segment and register all frames
        self._segment_and_register_frames()

        # Generate reference contours
        self._generate_reference_contours()

        # Transform contours for each time point
        self._transform_all_contours()

        # Create USD files
        self._create_usd_files()

        self.log_info("Processing pipeline completed successfully")
        painted_usd_files = {}
        if len(self.dynamic_labelmap_ids) > 0:
            painted_usd_files["dynamic"] = (
                f"{self.usd_project_name}.dynamic_painted.usd"
            )
            painted_usd_files["static"] = f"{self.usd_project_name}.static_painted.usd"
        else:
            painted_usd_files["all"] = f"{self.usd_project_name}.all_painted.usd"

        return painted_usd_files

    def _register(
        self,
        fixed_image: itk.Image,
        fixed_mask: Optional[itk.Image],
        moving_image: itk.Image,
        moving_mask: Optional[itk.Image],
        filename_prefix: str = "",
    ) -> dict[str, Union[itk.Transform, float]]:
        """Register moving image with mask."""
        self.registrar.set_fixed_image(fixed_image)
        self.registrar.set_fixed_mask(fixed_mask)

        reg_results = self.registrar.register(moving_image, moving_mask)

        if self.save_assets and len(filename_prefix) > 0:
            itk.imwrite(
                self.registrar.get_registered_image(),
                os.path.join(
                    self.output_directory,
                    f"{filename_prefix}_registered.mha",
                ),
                compression=True,
            )
            itk.transformwrite(
                reg_results["inverse_transform"],
                os.path.join(self.output_directory, f"{filename_prefix}_inverse.hdf"),
                compression=True,
            )
            itk.transformwrite(
                reg_results["forward_transform"],
                os.path.join(self.output_directory, f"{filename_prefix}_forward.hdf"),
                compression=True,
            )

        return reg_results

    def _segment_and_register_frames(self) -> None:
        """Segment each frame and register to reference image."""
        self.log_info("Segmenting and registering frames...")

        # Segment reference image
        self.log_info("Segmenting reference image...")

        # Set up registrar with reference image
        self.registrar.set_fixed_image(self.reference_image)

        self.reference_segmentation_results = self.segmenter.segment(
            self.reference_image
        )
        labelmap = self.reference_segmentation_results["labelmap"]
        if self.save_assets:
            itk.imwrite(
                labelmap,
                os.path.join(self.output_directory, "reference_labelmap.mha"),
                compression=True,
            )

        if len(self.dynamic_labelmap_ids) > 0:
            labelmap_arr = itk.GetArrayFromImage(labelmap)
            is_dynamic_arr = np.isin(labelmap_arr, self.dynamic_labelmap_ids)
            dynamic_labelmap_arr = np.where(is_dynamic_arr, 1, 0)
            reference_dynamic_labelmap = itk.GetImageFromArray(dynamic_labelmap_arr)
            reference_dynamic_labelmap.CopyInformation(self.reference_image)
            reference_dynamic_mask = self.image_tools.binary_dilate_image(
                reference_dynamic_labelmap, self.mask_dilation_radius
            )
            if self.save_assets:
                itk.imwrite(
                    reference_dynamic_mask,
                    os.path.join(self.output_directory, "reference_mask.mha"),
                    compression=True,
                )

            # Static anatomy is everything labeled but not dynamic; excluding
            # background (label 0) keeps the static ROI to actual anatomy.
            static_labelmap_arr = np.where((labelmap_arr > 0) & ~is_dynamic_arr, 1, 0)
            reference_static_labelmap = itk.GetImageFromArray(static_labelmap_arr)
            reference_static_labelmap.CopyInformation(self.reference_image)
            reference_static_mask = self.image_tools.binary_dilate_image(
                reference_static_labelmap, self.mask_dilation_radius
            )

        # Process each time point
        self.registration_results = []
        for i in range(self._num_time_points):
            self.log_progress(i + 1, self._num_time_points, prefix="Processing frames")

            moving_image = self.time_series_images[i]

            if len(self.dynamic_labelmap_ids) > 0:
                self.registrar.set_fixed_mask(reference_dynamic_mask)

                moving_segmentation_results = self.segmenter.segment(moving_image)
                moving_labelmap = moving_segmentation_results["labelmap"]
                if self.save_assets:
                    itk.imwrite(
                        moving_labelmap,
                        os.path.join(
                            self.output_directory, f"slice_{i:03d}_labelmap.mha"
                        ),
                        compression=True,
                    )

                moving_labelmap_arr = itk.GetArrayFromImage(moving_labelmap)
                moving_is_dynamic_arr = np.isin(
                    moving_labelmap_arr, self.dynamic_labelmap_ids
                )
                moving_dynamic_labelmap_arr = np.where(moving_is_dynamic_arr, 1, 0)
                moving_dynamic_labelmap = itk.GetImageFromArray(
                    moving_dynamic_labelmap_arr
                )
                moving_dynamic_labelmap.CopyInformation(moving_image)
                moving_mask = self.image_tools.binary_dilate_image(
                    moving_dynamic_labelmap, self.mask_dilation_radius
                )
                if self.save_assets:
                    itk.imwrite(
                        moving_mask,
                        os.path.join(self.output_directory, f"slice_{i:03d}_mask.mha"),
                        compression=True,
                    )

                dynamic_reg_results = self._register(
                    self.reference_image,
                    reference_dynamic_mask,
                    moving_image,
                    moving_mask,
                    f"slice_{i:03d}_dynamic",
                )

                static_labelmap_arr = np.where(
                    (moving_labelmap_arr > 0) & ~moving_is_dynamic_arr, 1, 0
                )
                static_labelmap = itk.GetImageFromArray(static_labelmap_arr)
                static_labelmap.CopyInformation(moving_image)
                static_mask = self.image_tools.binary_dilate_image(
                    static_labelmap, self.mask_dilation_radius
                )

                static_reg_results = self._register(
                    self.reference_image,
                    reference_static_mask,
                    moving_image,
                    static_mask,
                    f"slice_{i:03d}_static",
                )

                self.registration_results.append(
                    {
                        "dynamic": dynamic_reg_results,
                        "static": static_reg_results,
                    }
                )
            else:
                all_reg_results = self._register(
                    self.reference_image,
                    None,
                    moving_image,
                    None,
                    f"slice_{i:03d}_all",
                )
                self.registration_results.append(
                    {
                        "all": all_reg_results,
                    }
                )

    def _generate_reference_contours(self) -> None:
        """Generate contour meshes from reference segmentation."""
        self.log_info("Generating reference contours...")

        assert self.reference_segmentation_results is not None, (
            "reference segmentation results must be set"
        )
        labelmap = self.reference_segmentation_results["labelmap"]

        # Generate all anatomy contours
        all_contours = self.contour_tools.extract_contours(
            labelmap, surface_reduction_rate=self.surface_reduction_rate
        )
        self.reference_contours = {
            "all": all_contours,
        }

        if len(self.dynamic_labelmap_ids) > 0:
            dynamic_labelmap_arr = itk.GetArrayFromImage(labelmap)
            dynamic_labelmap_arr = np.where(
                np.isin(dynamic_labelmap_arr, self.dynamic_labelmap_ids),
                dynamic_labelmap_arr,
                0,
            )
            dynamic_labelmap = itk.GetImageFromArray(dynamic_labelmap_arr)
            dynamic_labelmap.CopyInformation(labelmap)
            dynamic_contours = self.contour_tools.extract_contours(
                dynamic_labelmap, surface_reduction_rate=self.surface_reduction_rate
            )

            static_labelmap_arr = itk.GetArrayFromImage(labelmap)
            static_labelmap_arr = np.where(
                np.isin(static_labelmap_arr, self.dynamic_labelmap_ids),
                0,
                static_labelmap_arr,
            )
            static_labelmap = itk.GetImageFromArray(static_labelmap_arr)
            static_labelmap.CopyInformation(labelmap)
            static_contours = self.contour_tools.extract_contours(
                static_labelmap, surface_reduction_rate=self.surface_reduction_rate
            )

            # Store reference contours
            self.reference_contours["dynamic"] = dynamic_contours
            self.reference_contours["static"] = static_contours

    def _transform_all_contours(self) -> None:
        """Transform contours for all time points using registration transforms."""
        self.log_info("Transforming contours for all time points...")

        if len(self.dynamic_labelmap_ids) > 0:
            anatomy_types = ["dynamic", "static"]
        else:
            anatomy_types = ["all"]

        for i in range(self._num_time_points):
            self.log_progress(
                i + 1, self._num_time_points, prefix="Transforming contours"
            )

            for anatomy_type in anatomy_types:
                # Get the forward transform for this anatomy type and frame
                forward_transform = self.registration_results[i][anatomy_type][
                    "forward_transform"
                ]

                # Transform the reference contours
                transformed_anatomy_contours = self.contour_tools.transform_contours(
                    self.reference_contours[anatomy_type],
                    forward_transform,
                    with_deformation_magnitude=True,
                )

                self.transformed_contours[anatomy_type].append(
                    transformed_anatomy_contours
                )

    def _create_usd_files(self) -> None:
        """Create painted USD files for all anatomy types."""
        self.log_info("Creating USD files...")

        if len(self.dynamic_labelmap_ids) > 0:
            anatomy_types = ["dynamic", "static"]
        else:
            anatomy_types = ["all"]

        # Create USD for each anatomy type
        for anatomy_type in anatomy_types:
            self.log_info("Creating %s anatomy USD...", anatomy_type)

            # Convert VTK contours to USD. Forwarding the segmenter so labels
            # land under /World/{project}/{type}/{label_name} (and materials
            # under /World/Looks/{type}/{label_name}_material).
            converter = ConvertVTKToUSD(
                self.usd_project_name,
                self.transformed_contours[anatomy_type],
                self.segmenter.taxonomy.all_labels(),
                segmenter=self.segmenter,
                frames_per_second=self.frames_per_second,
                log_level=self.log_level,
            )
            usd_file = os.path.join(
                self.output_directory, f"{self.usd_project_name}.{anatomy_type}.usd"
            )
            stage = converter.convert(usd_file)

            # Paint the USD file
            self.log_info("Painting %s anatomy USD...", anatomy_type)
            output_filename = os.path.join(
                self.output_directory,
                f"{self.usd_project_name}.{anatomy_type}_painted.usd",
            )
            if os.path.exists(output_filename):
                os.remove(output_filename)
            painter = USDAnatomyTools(stage)
            painter.enhance_meshes(self.segmenter)
            stage.Export(output_filename)
