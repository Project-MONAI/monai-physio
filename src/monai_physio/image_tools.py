"""
Image Tools for MONAI Physio

This module provides utilities for converting between different medical image formats
and performing image processing operations.
"""

import logging
from typing import Any, Optional, Union, cast, overload

import itk
import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray

from .monai_physio_base import MONAIPhysioBase


class ImageTools(MONAIPhysioBase):
    """
    Utilities for medical image format conversions and processing.

    This class provides methods for converting between ITK (Insight Toolkit) and
    SimpleITK image formats while preserving all metadata (origin, spacing, direction,
    pixel type). Supports both scalar and vector (multi-component) images.

    Example:
        >>> tools = ImageTools()
        >>> # Convert ITK to SimpleITK
        >>> sitk_image = tools.convert_itk_image_to_sitk(itk_image)
        >>> # Convert back to ITK
        >>> itk_image_back = tools.convert_sitk_image_to_itk(sitk_image)
    """

    def __init__(self, log_level: int | str = logging.INFO) -> None:
        """Initialize ImageTools.

        Args:
            log_level: Logging level (default: logging.INFO)
        """
        super().__init__(class_name=self.__class__.__name__, log_level=log_level)

    def imreadVD3(self, filename: str) -> Any:
        """Read an ITK vector image with double precision vectors.

        ITK's imread is not wrapped for itk.Image[itk.Vector[itk.D,3],3],
        so this method reads as itk.Image[itk.Vector[itk.F,3],3] and converts
        to double precision.

        Args:
            filename (str): Path to the image file to read

        Returns:
            itk.Image[itk.Vector[itk.D,3],3]: Vector image with double precision

        Example:
            >>> displacement_field = ImageTools().imreadVD3('deformation.mha')
        """
        # Read as float precision vector image
        image = itk.imread(filename)
        if "VD" in str(type(image)):
            return image

        image_arr = itk.array_from_image(image)
        image_double = self.convert_array_to_image_of_vectors(image_arr, image, itk.D)

        return image_double

    def imwriteVD3(self, image: Any, filename: str, compression: bool = True) -> None:
        """Write an ITK vector image with double precision vectors.

        ITK's imwrite is not wrapped for itk.Image[itk.Vector[itk.D,3],3],
        so this method converts to itk.Image[itk.Vector[itk.F,3],3] and writes.

        Args:
            image (itk.Image[itk.Vector[itk.D,3],3]): Vector image to write
            filename (str): Path to the output file
            compression (bool): Whether to use compression (default: True)

        Example:
            >>> ImageTools().imwriteVD3(displacement_field, 'deformation.mha')
        """
        # Convert to float precision for writing
        if "VD" not in str(type(image)):
            raise ValueError("Image must be a vector image with double precision")

        image_arr = itk.array_from_image(image)
        image_float = self.convert_array_to_image_of_vectors(image_arr, image, itk.F)

        # Write the float image
        itk.imwrite(image_float, filename, compression=compression)

    def convert_itk_image_to_sitk(self, itk_image: itk.Image) -> sitk.Image:
        """
        Convert an ITK image to a SimpleITK image.

        This method converts an ITK (Insight Toolkit) image to SimpleITK format while
        preserving all metadata including origin, spacing, direction, and pixel type.
        Works with both scalar and vector (multi-component) images.

        Args:
            itk_image: Input ITK image (can be scalar or vector image)

        Returns:
            SimpleITK image with identical data and metadata

        Example:
            >>> tools = ImageTools()
            >>> itk_image = itk.imread('image.nii.gz')
            >>> sitk_image = tools.convert_itk_image_to_sitk(itk_image)
        """
        array = itk.array_from_image(itk_image)

        # Get image metadata
        origin = itk.origin(itk_image)
        spacing = itk.spacing(itk_image)
        direction = itk.array_from_matrix(itk_image.GetDirection())

        # Check if this is a vector image
        is_vector = False
        if hasattr(itk_image, "GetNumberOfComponentsPerPixel"):
            n_components = itk_image.GetNumberOfComponentsPerPixel()
            is_vector = n_components > 1

        if is_vector:
            sitk_image = sitk.GetImageFromArray(array, isVector=True)
        else:
            sitk_image = sitk.GetImageFromArray(array, isVector=False)

        # Set metadata
        # Convert origin and spacing to tuples (reverse order for SimpleITK: x, y, z)
        sitk_image.SetOrigin(tuple(origin))
        sitk_image.SetSpacing(tuple(spacing))

        # Direction matrix needs to be flattened
        # ITK and SimpleITK use the same direction convention, we just need to flatten it correctly
        direction_flat = direction.flatten()
        sitk_image.SetDirection(direction_flat.tolist())

        return sitk_image

    def convert_sitk_image_to_itk(self, sitk_image: sitk.Image) -> itk.Image:
        """
        Convert a SimpleITK image to an ITK image.

        This method converts a SimpleITK image to ITK (Insight Toolkit) format while
        preserving all metadata including origin, spacing, direction, and pixel type.
        Works with both scalar and vector (multi-component) images.

        Args:
            sitk_image: Input SimpleITK image (can be scalar or vector image)

        Returns:
            ITK image with identical data and metadata

        Example:
            >>> tools = ImageTools()
            >>> sitk_image = sitk.ReadImage('image.nii.gz')
            >>> itk_image = tools.convert_sitk_image_to_itk(sitk_image)
        """
        array = sitk.GetArrayFromImage(sitk_image)

        # Get image metadata
        origin = sitk_image.GetOrigin()  # Returns (x, y, z)
        spacing = sitk_image.GetSpacing()  # Returns (x, y, z)
        direction = sitk_image.GetDirection()  # Returns flattened direction matrix
        dimension = sitk_image.GetDimension()
        n_components = sitk_image.GetNumberOfComponentsPerPixel()

        # Check if this is a vector image
        is_vector = n_components > 1

        # Create ITK image from numpy array
        if is_vector:
            # Vector image
            itk_image = itk.image_from_array(array, is_vector=True)
        else:
            # Scalar image
            itk_image = itk.image_from_array(array, is_vector=False)

        # Set origin (reverse order: SimpleITK gives x,y,z, ITK expects x,y,z internally
        # but we set it using the same order)
        itk_image.SetOrigin(origin)

        # Set spacing
        itk_image.SetSpacing(spacing)

        # Set direction matrix
        # Reshape direction to matrix form
        if dimension == 2:
            direction_matrix = np.array(direction).reshape(2, 2)
        elif dimension == 3:
            direction_matrix = np.array(direction).reshape(3, 3)
        else:
            raise ValueError(f"Unsupported image dimension: {dimension}")

        # Convert numpy array to ITK matrix and set
        itk_direction = itk.matrix_from_array(direction_matrix)
        itk_image.SetDirection(itk_direction)

        return itk_image

    def convert_array_to_image_of_vectors(
        self,
        arr_data: NDArray[Any],
        reference_image: Any,
        ptype: Any = itk.D,
    ) -> Any:
        """
        Convert a numpy array to an ITK image of vector type.

        This method is needed because itk in python does not support creating
        images of vectors with itk.D precision.   Luckily array_view_from_image
        does support itk.D precision vectors.
        """
        if ptype not in [itk.F, itk.D]:
            if ptype == np.float32:
                ptype = itk.F
            elif ptype == np.float64:
                ptype = itk.D
            else:
                raise ValueError(f"Unsupported component type: {ptype}")

        itk_image = itk.Image[itk.Vector[ptype, 3], 3].New()
        itk_image.SetRegions(reference_image.GetLargestPossibleRegion())
        itk_image.SetSpacing(reference_image.GetSpacing())
        itk_image.SetOrigin(reference_image.GetOrigin())
        itk_image.SetDirection(reference_image.GetDirection())
        itk_image.Allocate()
        itk.array_view_from_image(itk_image)[:] = arr_data

        return itk_image

    def make_isotropic_image(self, image: itk.Image) -> itk.Image:
        """Resample a 3-D *image* to isotropic spacing using the finest voxel pitch.

        Args:
            image: 3-D ITK image to resample.

        Returns:
            Resampled image with uniform spacing equal to the smallest input spacing.

        Raises:
            ValueError: If *image* is not 3-D.
        """
        if image.GetImageDimension() != 3:
            raise ValueError(
                f"make_isotropic_image requires a 3-D image; "
                f"got {image.GetImageDimension()}-D"
            )
        spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
        size = np.asarray(image.GetLargestPossibleRegion().GetSize(), dtype=np.int64)

        min_spacing = float(spacing.min())
        new_spacing = [min_spacing] * 3
        # Ceiling to avoid clipping the image boundary.
        new_size = [int(np.ceil(size[i] * spacing[i] / min_spacing)) for i in range(3)]

        ImageType = type(image)
        interpolator = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()
        resampler = itk.ResampleImageFilter[ImageType, ImageType].New()
        resampler.SetInput(image)
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.Update()
        result = resampler.GetOutput()
        result.DisconnectPipeline()
        return result

    def resample_image_by_scale(
        self, image: itk.Image, scale: float, interpolate: bool = True
    ) -> itk.Image:
        """Resample a 3-D *image* to *scale* times its voxel count per axis.

        The physical extent is preserved: spacing is rescaled to compensate for
        the new voxel count, and the origin shifts by half the spacing change so
        the resampled voxel centers stay inside the original extent.

        Args:
            image: 3-D ITK image to resample.
            scale: Per-axis voxel-count multiplier.  Values below ``1.0``
                coarsen, values above ``1.0`` upsample.
            interpolate: Use linear interpolation.  ``False`` selects nearest
                neighbor, which is what labelmaps need.

        Returns:
            Resampled image, covering the same physical extent as *image*.

        Raises:
            ValueError: If *image* is not 3-D, or *scale* is not positive.
        """
        if image.GetImageDimension() != 3:
            raise ValueError(
                f"resample_image_by_scale requires a 3-D image; "
                f"got {image.GetImageDimension()}-D"
            )
        if scale <= 0.0:
            raise ValueError(f"scale must be positive; got {scale}")

        spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
        size = np.asarray(image.GetLargestPossibleRegion().GetSize(), dtype=np.int64)
        new_size = np.maximum(1, np.ceil(size * scale)).astype(np.int64)
        new_spacing = spacing * size / new_size

        ImageType = type(image)
        if interpolate:
            interpolator = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()
        else:
            interpolator = itk.NearestNeighborInterpolateImageFunction[
                ImageType, itk.D
            ].New()

        direction = itk.array_from_matrix(image.GetDirection())
        resampler = itk.ResampleImageFilter[ImageType, ImageType].New()
        resampler.SetInput(image)
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputSpacing([float(v) for v in new_spacing])
        resampler.SetSize([int(n) for n in new_size])
        resampler.SetOutputOrigin(
            np.asarray(image.GetOrigin(), dtype=np.float64)
            + direction @ ((new_spacing - spacing) / 2.0)
        )
        resampler.SetOutputDirection(image.GetDirection())
        resampler.Update()
        result = resampler.GetOutput()
        result.DisconnectPipeline()
        return result

    @staticmethod
    def _per_axis_values(
        value: Union[float, int, list, tuple, NDArray[Any]],
        dimension: int,
        name: str,
    ) -> list[float]:
        """Broadcast a scalar to every axis, or validate a per-axis sequence.

        Args:
            value: Scalar applied to every axis, or one value per axis.
            dimension: Number of image dimensions expected.
            name: Parameter name, used in error messages.

        Returns:
            One value per axis.

        Raises:
            ValueError: If a sequence has the wrong length, or any value is
                negative.
        """
        if np.isscalar(value):
            values = [float(cast(float, value))] * dimension
        else:
            values = [float(v) for v in value]  # type: ignore[union-attr]
            if len(values) != dimension:
                raise ValueError(
                    f"{name} needs a scalar or one value per image dimension "
                    f"({dimension}), got {len(values)}."
                )
        if any(v < 0.0 for v in values):
            raise ValueError(f"{name} must be >= 0, got {values}")
        return values

    def pad_image(
        self,
        image: itk.Image,
        pad_portion: Optional[
            Union[float, list[float], tuple[float, ...], NDArray[Any]]
        ] = None,
        pad_voxels: Optional[
            Union[int, list[int], tuple[int, ...], NDArray[Any]]
        ] = None,
        background_value: float = 0.0,
    ) -> itk.Image:
        """Pad *image* on every side with a constant-valued margin.

        The margin is given either as a portion of each axis' physical extent
        (*pad_portion*) or directly in voxels (*pad_voxels*); exactly one of the
        two must be supplied. Either accepts a scalar, applied to every axis, or
        one value per image dimension. Both pad the lower and the upper end of
        every axis, so ``pad_voxels=10`` grows all six faces of a 3-D image by
        ten voxels. Spacing and direction are untouched.

        The origin and size are updated together so the original voxels keep
        their physical positions: the padded image's index ``(0, 0, 0)`` sits one
        margin below the input's, and the input data occupies the interior.
        (``itk.ConstantPadImageFilter`` alone reports the margin as a negative
        start index instead, which most file formats drop on write - shifting the
        data. The region-of-interest pass here folds that index back into the
        origin.)

        Args:
            image: ITK image to pad.
            pad_portion: Portion of an axis' physical extent (``size * spacing``)
                to add at both ends, as a scalar for every axis or one value per
                dimension: ``0.1`` grows an axis spanning 200 mm by 20 mm per
                side. Rounded up to whole voxels. Mutually exclusive with
                *pad_voxels*.
            pad_voxels: Margin in voxels, as a scalar for every axis or one value
                per dimension, applied at both ends of each axis. Mutually
                exclusive with *pad_portion*.
            background_value: Pixel value written into the new margin
                (default: 0.0).

        Returns:
            Padded image with the same pixel type, spacing and direction.

        Raises:
            ValueError: If neither or both of *pad_portion* and *pad_voxels* are
                given, if either is negative, or if a sequence does not have one
                entry per image dimension.
        """
        if (pad_portion is None) == (pad_voxels is None):
            raise ValueError(
                "Specify exactly one of pad_portion or pad_voxels; got "
                f"pad_portion={pad_portion}, pad_voxels={pad_voxels}."
            )

        size = [int(s) for s in image.GetLargestPossibleRegion().GetSize()]
        if pad_portion is not None:
            portions = self._per_axis_values(pad_portion, len(size), "pad_portion")
            # extent_i = size_i * spacing_i, and pad_portion * extent_i of margin
            # is that distance divided by spacing_i, so the spacing cancels.
            margin = [int(np.ceil(p * s)) for p, s in zip(portions, size)]
            self.log_info(
                "Padding by %s voxels per side (%s of extent); size %s -> %s",
                margin,
                [f"{p * 100.0:.1f}%" for p in portions],
                size,
                [s + 2 * p for s, p in zip(size, margin)],
            )
        else:
            assert pad_voxels is not None  # guaranteed by the check above
            margin = [
                int(v)
                for v in self._per_axis_values(pad_voxels, len(size), "pad_voxels")
            ]
            self.log_info(
                "Padding by %s voxels per side; size %s -> %s",
                margin,
                size,
                [s + 2 * p for s, p in zip(size, margin)],
            )

        ImageType = type(image)
        # SetConstant is typed to the pixel type, so an integer image rejects a
        # Python float.
        pixel_type = itk.template(image)[1][0]
        constant = (
            float(background_value)
            if pixel_type in (itk.F, itk.D)
            else int(round(background_value))
        )

        pad_filter = itk.ConstantPadImageFilter[ImageType, ImageType].New()
        pad_filter.SetInput(image)
        pad_filter.SetPadLowerBound(margin)
        pad_filter.SetPadUpperBound(margin)
        pad_filter.SetConstant(constant)
        pad_filter.Update()
        padded = pad_filter.GetOutput()

        # Re-anchor the padded region at index 0, moving the margin into the
        # origin so the physical position of the original data is preserved.
        roi_filter = itk.RegionOfInterestImageFilter[ImageType, ImageType].New()
        roi_filter.SetInput(padded)
        roi_filter.SetRegionOfInterest(padded.GetLargestPossibleRegion())
        roi_filter.Update()
        result = roi_filter.GetOutput()
        result.DisconnectPipeline()
        return result

    def binary_dilate_image(
        self,
        image: itk.Image,
        radius: int,
        foreground_value: int = 1,
        background_value: int = 0,
    ) -> itk.Image:
        """Binary-dilate *image* with a ball structuring element.

        Args:
            image: Binary (or label) image to dilate.
            radius: Radius, in voxels, of the ball structuring element.
            foreground_value: Pixel value treated as foreground (default: 1).
            background_value: Pixel value written for background voxels
                (default: 0).

        Returns:
            Dilated image with the same pixel type as *image*.
        """
        ImageType = type(image)
        dimension = image.GetImageDimension()
        StructuringElementType = itk.FlatStructuringElement[dimension]
        structuring_element = StructuringElementType.Ball(int(radius))

        dilate_filter = itk.BinaryDilateImageFilter[
            ImageType, ImageType, StructuringElementType
        ].New()
        dilate_filter.SetInput(image)
        dilate_filter.SetKernel(structuring_element)
        dilate_filter.SetForegroundValue(foreground_value)
        dilate_filter.SetBackgroundValue(background_value)
        dilate_filter.Update()
        result = dilate_filter.GetOutput()
        result.DisconnectPipeline()
        return result

    def binary_erode_image(
        self,
        image: itk.Image,
        radius: int,
        foreground_value: int = 1,
        background_value: int = 0,
    ) -> itk.Image:
        """Binary-erode *image* with a ball structuring element.

        Args:
            image: Binary (or label) image to erode.
            radius: Radius, in voxels, of the ball structuring element.
            foreground_value: Pixel value treated as foreground (default: 1).
            background_value: Pixel value written for eroded-away voxels
                (default: 0).

        Returns:
            Eroded image with the same pixel type as *image*.
        """
        ImageType = type(image)
        dimension = image.GetImageDimension()
        StructuringElementType = itk.FlatStructuringElement[dimension]
        structuring_element = StructuringElementType.Ball(int(radius))

        erode_filter = itk.BinaryErodeImageFilter[
            ImageType, ImageType, StructuringElementType
        ].New()
        erode_filter.SetInput(image)
        erode_filter.SetKernel(structuring_element)
        erode_filter.SetForegroundValue(foreground_value)
        erode_filter.SetBackgroundValue(background_value)
        erode_filter.Update()
        result = erode_filter.GetOutput()
        result.DisconnectPipeline()
        return result

    def keep_largest_connected_component(
        self,
        image: itk.Image,
        foreground_value: int = 1,
        fully_connected: bool = False,
    ) -> itk.Image:
        """Keep only the largest connected component of a binary image.

        Args:
            image: Binary (non-zero = foreground) image.
            foreground_value: Value written for the retained component's
                voxels (default: 1).
            fully_connected: Whether diagonally-adjacent voxels are
                considered connected (default: False, i.e. face connectivity
                only).

        Returns:
            Binary image, same pixel type as *image*, containing only the
            largest connected component with value *foreground_value*
            (background is 0).
        """
        # SimpleITK's filters are not templated on pixel type in the Python
        # layer, so this avoids the itk.ConnectedComponentImageFilter /
        # itk.RelabelComponentImageFilter Python wrappings, which only cover
        # a limited set of input/output pixel type combinations that vary
        # across ITK Python builds.
        ImageType = type(image)
        sitk_image = self.convert_itk_image_to_sitk(image)

        cc_image = sitk.ConnectedComponent(sitk_image, fully_connected)
        relabeled = sitk.RelabelComponent(cc_image, sortByObjectSize=True)
        largest = sitk.BinaryThreshold(
            relabeled,
            lowerThreshold=1,
            upperThreshold=1,
            insideValue=foreground_value,
            outsideValue=0,
        )

        result = self.convert_sitk_image_to_itk(largest)
        if type(result) is ImageType:
            return result
        return itk.cast_image_filter(result, ttype=(type(result), ImageType))

    @overload
    def flip_image(
        self,
        in_image: itk.Image,
        in_mask: None = None,
        flip_x: bool = False,
        flip_y: bool = False,
        flip_z: bool = False,
        flip_and_make_identity: bool = False,
    ) -> itk.Image: ...

    @overload
    def flip_image(
        self,
        in_image: itk.Image,
        in_mask: itk.Image,
        flip_x: bool = False,
        flip_y: bool = False,
        flip_z: bool = False,
        flip_and_make_identity: bool = False,
    ) -> tuple[itk.Image, itk.Image]: ...

    def flip_image(
        self,
        in_image: itk.Image,
        in_mask: Optional[itk.Image] = None,
        flip_x: bool = False,
        flip_y: bool = False,
        flip_z: bool = False,
        flip_and_make_identity: bool = False,
    ) -> Union[itk.Image, tuple[itk.Image, itk.Image]]:
        """
        Flip the image and mask.

        Only axis-aligned flips are supported. If ``flip_and_make_identity`` is
        True, the image and mask are first flipped along any axes whose
        corresponding diagonal entries in the direction matrix are negative
        (assuming the direction matrix encodes only axis-aligned flips), then
        any additional requested flips are performed, and finally the direction
        matrix is set to the identity matrix. This is useful when combining ITK
        images with VTK objects (that often do not support a direction matrix).

        Args:
            in_image: The input image to flip
            in_mask: The input mask to flip
            flip_x: Flip the image and mask along the x-axis
            flip_y: Flip the image and mask along the y-axis
            flip_z: Flip the image and mask along the z-axis
            flip_and_make_identity: Flip the image and mask and make the direction
                matrix identity.
        """
        flip0 = False
        flip1 = False
        flip2 = False
        if flip_and_make_identity:
            # itk.array_from_matrix avoids itk.Matrix.__array__, whose missing
            # copy keyword triggers a numpy>=2.0 DeprecationWarning.
            direction = itk.array_from_matrix(in_image.GetDirection())
            flip0 = direction[0, 0] < 0
            flip1 = direction[1, 1] < 0
            flip2 = direction[2, 2] < 0
        if flip_x:
            flip0 = True
        if flip_y:
            flip1 = True
        if flip_z:
            flip2 = True
        if flip0 or flip1 or flip2:
            self.log_info(f"Flipping image: {flip0}, {flip1}, {flip2}")
            flip_filter = itk.FlipImageFilter.New(Input=in_image)
            flip_filter.SetFlipAxes([int(flip0), int(flip1), int(flip2)])
            flip_filter.SetFlipAboutOrigin(True)
            flip_filter.Update()
            out_image = flip_filter.GetOutput()
            if flip_and_make_identity:
                id_mat = itk.Matrix[itk.D, 3, 3]()
                id_mat.SetIdentity()
                out_image.SetDirection(id_mat)
            if in_mask is not None:
                flip_filter = itk.FlipImageFilter.New(Input=in_mask)
                flip_filter.SetFlipAxes([int(flip0), int(flip1), int(flip2)])
                flip_filter.SetFlipAboutOrigin(True)
                flip_filter.Update()
                out_mask = flip_filter.GetOutput()
                if flip_and_make_identity:
                    id_mat = itk.Matrix[itk.D, 3, 3]()
                    id_mat.SetIdentity()
                    out_mask.SetDirection(id_mat)
                return out_image, out_mask
            else:
                return out_image
        else:
            if in_mask is not None:
                return in_image, in_mask
            else:
                return in_image
