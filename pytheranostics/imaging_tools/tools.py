"""Tools for medical image manipulation and processing."""

from __future__ import annotations

import glob
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy
import pydicom
import SimpleITK
from rt_utils import RTStructBuilder
from SimpleITK import Image

from pytheranostics.dicomtools.dicomtools import sitk_load_dcm_series

if TYPE_CHECKING:
    # Imported only for type checking to avoid circular imports at runtime
    from pytheranostics.imaging_ds.metadata import ImagingMetadata

from pytheranostics.registration.ct_to_spect import (
    register_ct_to_spect,
    transform_ct_mask_to_spect,
)

logger = logging.getLogger(__name__)

# TODO: Move under dicomtools, and have two sets: one generic (the current dicomtools.py) and on specific for pyTheranostic functions (containing
# the code below)


def load_metadata(dir: str, modality: str) -> ImagingMetadata:
    """Load relevant meta-data from a dicom dataset.

    Args:
        dir (str): Directory path containing DICOM files.
        modality (str): The imaging modality.

    Raises
    ------
    AssertionError
        If no DICOM data found in directory.
    ValueError
        If modality mismatch detected.

    Returns
    -------
    ImagingMetadata
        Metadata object with imaging information.
    """
    # Convert Path to string if needed
    dir_str = str(dir)

    dicom_slices = [
        pydicom.dcmread(fname)
        for fname in glob.glob(dir_str + "/*.dcm", recursive=False)
    ]

    if len(dicom_slices) == 0:
        raise AssertionError(f"No Dicom data was found under {dir}")

    radionuclide = None
    injected_activity = None

    if modality == "CT":
        dicom_slices = [f for f in dicom_slices if hasattr(f, "SliceLocation")]
        dicom_slices = sorted(dicom_slices, key=lambda s: s.SliceLocation)

        if dicom_slices[0].Modality != "CT":
            raise ValueError(
                f"Wrong modality. User specified CT, howere dicom indicates {dicom_slices[0].Modality}."
            )

    else:
        if dicom_slices[0].Modality not in ["NM", "PT"]:
            raise ValueError(
                f"Wrong modality. User specified NM/PT, however dicom indicates {dicom_slices[0].Modality}."
            )

        radionuclide = modality.split("_")[0]

        # This only applies to Q-SPECT TODO: replace for something more generic.
        injected_activity = None

        if hasattr(dicom_slices[0], "RadiopharmaceuticalInformationSequence"):
            rp_seq = dicom_slices[0].RadiopharmaceuticalInformationSequence
            if len(rp_seq) > 0:
                try:
                    injected_activity = rp_seq[0].RadionuclideTotalDose

                    # Currently we don't have a way to know the units ... so we use common sense.
                    if (
                        injected_activity > 20000
                    ):  # Activity likely in Bq instead of MBq
                        injected_activity /= 1e6
                    logger.info(
                        f"Injected activity found in DICOM Header: {injected_activity:2.1f} MBq. Please verify."
                    )
                except AttributeError:
                    # Sequence exists but RadionuclideTotalDose attribute is missing
                    logger.warning(
                        "RadiopharmaceuticalInformationSequence found but RadionuclideTotalDose is missing."
                    )
            else:
                # Sequence exists but is empty - this may indicate a data quality issue
                logger.warning(
                    "RadiopharmaceuticalInformationSequence is empty. This may indicate a data quality issue."
                )

        if injected_activity is None:
            logger.info("Using default injected activity: 7400 MBq")
            injected_activity = 7400.0

    # Global attributes. Should be the same in all slices!
    slice_ = dicom_slices[0]

    # Local import from shared types to avoid circular dependencies
    from pytheranostics.shared.types import ImagingMetadata

    meta = ImagingMetadata(
        PatientID=slice_.PatientID,
        AcquisitionDate=slice_.AcquisitionDate,
        AcquisitionTime=slice_.AcquisitionTime,
        HoursAfterInjection=None,
        Radionuclide=radionuclide,
        Injected_Activity_MBq=injected_activity,
    )

    return meta


def itk_image_from_array(
    array: numpy.ndarray, ref_image: Image, is_mask: bool = False
) -> Image:
    """Create an ITK Image object with a new array and existing meta-data.

    Uses meta-data from another reference ITK image.

    Args:
        array (numpy.ndarray): Array data for the new image.
        ref_image (Image): Reference ITK image for metadata.
        is_mask (bool): Whether the array represents a mask.

    Returns
    -------
    Image
        New ITK Image object with array data and reference metadata.
    """
    # Cast if masks:
    if is_mask:
        array = array.astype(numpy.uint8)

    image = SimpleITK.GetImageFromArray(array)

    if is_mask:
        image = SimpleITK.Cast(image, SimpleITK.sitkUInt8)

    # Set Manually basic meta:
    tmp_spacing = list(ref_image.GetSpacing())
    tmp_origin = list(ref_image.GetOrigin())

    if (
        len(tmp_spacing) - len(array.shape) == 1
    ):  # Sometime we get NM data with 4 dimensions, the last one being just a dummy.
        tmp_spacing = tmp_spacing[:-1]
        tmp_origin = tmp_origin[:-1]

    image.SetSpacing([tmp_spacing[0], tmp_spacing[1], tmp_spacing[2]])
    image.SetOrigin(tmp_origin)
    tmp_direction = list(ref_image.GetDirection())

    if len(tmp_direction) > 9:
        image.SetDirection(
            tmp_direction[0:3] + tmp_direction[4:7] + tmp_direction[8:11]
        )
    else:
        image.SetDirection(tmp_direction)

    # Here we set the additional meta-data.
    for key in ref_image.GetMetaDataKeys():
        image.SetMetaData(key, ref_image.GetMetaData(key))

    return image


def _parse_dicom_date_time(
    date_val: Optional[str], time_val: Optional[str]
) -> Optional[datetime]:
    """Parse DICOM DA/TM values into a Python datetime."""
    if not date_val:
        return None

    date_str = str(date_val)
    time_str = str(time_val or "000000").split(".")[0]
    if len(time_str) < 6:
        time_str = time_str.ljust(6, "0")

    try:
        return datetime.strptime(f"{date_str}{time_str[:6]}", "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _parse_dicom_datetime(dt_val: Optional[str]) -> Optional[datetime]:
    """Parse a DICOM DT value into a Python datetime."""
    if not dt_val:
        return None

    value = str(dt_val).split(".")[0]
    if len(value) < 8:
        return None

    return _parse_dicom_date_time(value[:8], value[8:14] if len(value) > 8 else None)


def _get_acquisition_start_datetime(ds: pydicom.Dataset) -> Optional[datetime]:
    """Return the best available acquisition-start datetime from a DICOM dataset."""
    for date_key, time_key in [
        ("AcquisitionDate", "AcquisitionTime"),
        ("SeriesDate", "SeriesTime"),
        ("ContentDate", "ContentTime"),
        ("StudyDate", "StudyTime"),
    ]:
        dt = _parse_dicom_date_time(
            getattr(ds, date_key, None), getattr(ds, time_key, None)
        )
        if dt is not None:
            return dt
    return None


def _get_radiopharm_admin_datetime_and_half_life(
    ds: pydicom.Dataset,
) -> Tuple[Optional[datetime], Optional[float]]:
    """Return radiopharmaceutical administration datetime and half-life in seconds."""
    if not hasattr(ds, "RadiopharmaceuticalInformationSequence"):
        return None, None

    rp_seq = ds.RadiopharmaceuticalInformationSequence
    if len(rp_seq) == 0:
        return None, None

    rp_info = rp_seq[0]
    admin_dt = _parse_dicom_datetime(
        getattr(rp_info, "RadiopharmaceuticalStartDateTime", None)
    )
    if admin_dt is None:
        admin_date = getattr(rp_info, "RadiopharmaceuticalStartDate", None)
        if admin_date is None:
            admin_date = getattr(ds, "StudyDate", None)
        admin_dt = _parse_dicom_date_time(
            admin_date,
            getattr(rp_info, "RadiopharmaceuticalStartTime", None),
        )

    half_life_seconds = None
    half_life_val = getattr(rp_info, "RadionuclideHalfLife", None)
    if half_life_val is not None:
        try:
            half_life_seconds = float(half_life_val)
        except (TypeError, ValueError):
            half_life_seconds = None

    return admin_dt, half_life_seconds


def apply_qspect_dcm_scaling(
    image: Image, dir: str, scale_factor: Optional[Tuple[float, float]] = None
) -> Image:
    """Read DICOM metadata to scale quantitative NM/PT images and harmonize decay reference.

    NM images are converted using the Real World Value Mapping when available. NM and PT
    images are then optionally converted from ADMIN-referenced decay correction to START.
    """
    path_dir = Path(dir)
    dicom_files = sorted(path_dir.glob("*.dcm"))
    if len(dicom_files) == 0:
        raise AssertionError(f"No Dicom data was found under {dir}")

    dcm_data = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True, force=True)
    modality = getattr(dcm_data, "Modality", None)

    if modality not in ["NM", "PT"]:
        raise AssertionError(
            f"Wrong Modality, expecting NM/PT quantitative data, but got {modality}"
        )

    if modality == "NM" and scale_factor is None and len(dicom_files) != 1:
        raise AssertionError(
            f"Found more than 1 .dcm file inside {path_dir.name}, not sure which one is the right SPECT."
        )

    image_array = numpy.squeeze(SimpleITK.GetArrayFromImage(image)).astype(
        numpy.float32
    )

    if scale_factor is not None:
        slope = float(scale_factor[0])
        intercept = float(scale_factor[1])
        logger.info(
            "Applying user-provided quantitative scaling for %s: slope=%s intercept=%s.",
            modality,
            slope,
            intercept,
        )
        image_array = slope * image_array + intercept
    elif modality == "NM":
        slope = float(dcm_data.RealWorldValueMappingSequence[0].RealWorldValueSlope)
        intercept = float(
            dcm_data.RealWorldValueMappingSequence[0].RealWorldValueIntercept
        )
        logger.info(
            "Applying NM quantitative scaling obtained from DICOM "
            "RealWorldValueMappingSequence: slope=%s intercept=%s.",
            slope,
            intercept,
        )
        image_array = slope * image_array + intercept
    else:
        logger.info(
            "PT quantitative values are assumed to be scaled correctly by SimpleITK; no slope/intercept scaling applied."
        )

    decay_correction = str(getattr(dcm_data, "DecayCorrection", "") or "").upper()
    logger.info(
        "DICOM DecayCorrection for %s image is '%s'.",
        modality,
        decay_correction or "MISSING",
    )

    if decay_correction == "START":
        logger.info(
            "No additional decay normalization is needed because the image is already referenced to acquisition start."
        )
    elif decay_correction == "ADMIN":
        acquisition_dt = _get_acquisition_start_datetime(dcm_data)
        admin_dt, half_life_seconds = _get_radiopharm_admin_datetime_and_half_life(
            dcm_data
        )

        if acquisition_dt is None:
            logger.warning(
                "DecayCorrection is ADMIN but acquisition start time could not be resolved. Skipping decay normalization to START."
            )
        elif admin_dt is None:
            logger.warning(
                "DecayCorrection is ADMIN but radiopharmaceutical administration time could not be resolved. Skipping decay normalization to START."
            )
        elif half_life_seconds is None or half_life_seconds <= 0:
            logger.warning(
                "DecayCorrection is ADMIN but RadionuclideHalfLife is missing or invalid (%s). Skipping decay normalization to START.",
                half_life_seconds,
            )
        else:
            delta_seconds = (acquisition_dt - admin_dt).total_seconds()
            if delta_seconds < 0:
                logger.warning(
                    "DecayCorrection is ADMIN but acquisition start (%s) is earlier than administration time (%s). Skipping decay normalization to START.",
                    acquisition_dt.isoformat(),
                    admin_dt.isoformat(),
                )
            else:
                decay_factor = math.exp(
                    -math.log(2.0) * delta_seconds / half_life_seconds
                )
                logger.info(
                    "Converting %s image from ADMIN to START decay reference using administration time %s, acquisition time %s, half-life %.6g s, elapsed time %.3f s, factor %.9g.",
                    modality,
                    admin_dt.isoformat(),
                    acquisition_dt.isoformat(),
                    half_life_seconds,
                    delta_seconds,
                    decay_factor,
                )
                image_array *= decay_factor
    elif decay_correction:
        logger.info(
            "DecayCorrection '%s' is not explicitly handled. Image values are left unchanged.",
            decay_correction,
        )
    else:
        logger.info("DecayCorrection tag is missing. Image values are left unchanged.")

    return itk_image_from_array(array=image_array, ref_image=image)


def apply_qspect_dcm_origin(image: Image, dir: str) -> Image:
    """Apply Origin and Direction from dicom header if needed.

    This could happen when SPECT data is stored as a single .dcm file (i.e., stored as "NM" modality),
    ITKSnap sometimes fails to read the Position and Direction correctly, so we pull it from pydicom.

    Parameters
    ----------
    image : SimpleITK.Image
        SimpleITK image object
    dir : str
        Path to dcm file containing SPECT reconstruction

    Returns
    -------
    Image
        Image with correct Origin and Direction
    """
    # We use pydicom to access the appropriate tag:
    # First, find the SPECT dicom file:
    path_dir = Path(dir)
    nm_files = [files for files in path_dir.glob("*.dcm")]

    dcm_data = pydicom.dcmread(str(nm_files[0]))
    modality = dcm_data.Modality

    if (
        modality == "PT"
    ):  # PET modality stores individual dicom files for each slice, as CT.
        # Therefore nothing to do here.
        return image
    elif modality != "NM":
        raise AssertionError(f"Data is not SPECT. Modality found: {modality}")

    if len(nm_files) > 1:
        raise AssertionError(
            "Found more than 1 dicom file inside the folder but loaded sample is stored as NM."
            " There should only be a single dicome file."
        )

    if getattr(dcm_data, "ImagePositionPatient", None) is None:

        # Verify SimpleITK got the right origin and direction
        dcm_origin = dcm_data.DetectorInformationSequence[0].ImagePositionPatient
        itk_origin = image.GetOrigin()

        for idx in range(len(dcm_origin)):
            if abs(dcm_origin[idx] - itk_origin[idx]) > 0.1:
                raise AssertionError(
                    f"Missmatch between DCM origin {dcm_origin} and ITK origin {itk_origin}"
                )

        dcm_direction = dcm_data.DetectorInformationSequence[0].ImageOrientationPatient
        itk_direction = image.GetDirection()

        for idx in range(len(dcm_direction)):
            if abs(dcm_direction[idx] - itk_direction[idx]) > 0.1:
                raise AssertionError(
                    f"Missmatch between DCM direction {dcm_direction} and ITK direction {itk_direction}"
                )

    else:
        image.SetOrigin(dcm_data.ImagePositionPatient)
        image.SetDirection(list(dcm_data.ImageOrientationPatient) + [0, 0, 1])

    return image


def squeeze_sitk_image_dimension(
    img: SimpleITK.Image, dim: int = 3, slice_index: int = 0
) -> SimpleITK.Image:
    """
    Remove a singleton dimension from a SimpleITK image, like numpy.squeeze.

    Parameters
    ----------
    img : SimpleITK.Image
        Your input image (e.g. a 4D volume with size (Nx, Ny, Nz, 1)).
    dim : int
        The zero-based dimension to remove (for (Nx,Ny,Nz,1), dim=3).
    slice_index : int
        Which slice along that dimension to keep (must be < img.GetSize()[dim]).

    Returns
    -------
    squeezed : SimpleITK.Image
        A new image with one fewer dimension (e.g. (Nx,Ny,Nz)).
    """
    # 0) If image is 2-D, error; if image is 3-D, nothing to do.
    if img.GetDimension() < 3:
        raise AssertionError(
            f"Image Dimensions are not valid:  dim={img.GetDimension()}, size={img.GetSize()}"
        )

    if img.GetDimension() == 3:
        return img

    # 1) build size vector, set the target dim to 0 => collapse it
    size = list(img.GetSize())
    size[dim] = 0

    # 2) build index vector, pick which slice of the dropped dim you want
    index = [0] * img.GetDimension()
    index[dim] = slice_index

    # 3) run the extractor
    extractor = SimpleITK.ExtractImageFilter()
    extractor.SetSize(size)
    extractor.SetIndex(index)

    return extractor.Execute(img)


def load_from_dicom_dir(
    dir: str, modality: str, calibration_factor: Optional[float] = None
) -> Tuple[Image, ImagingMetadata]:
    """Load CT or SPECT data from DICOM files in the specified folder.

    Returns the Image object and some relevant metadata.

    Args:
        dir (str): Directory path containing DICOM files.
        modality (str): The imaging modality.
        calibration_factor (str, optional): Factor to scale SPECT voxel values (e.g., could be SPECT calibration Factor in BQ/CPS or dimensionless factor)

    Returns
    -------
    Tuple[Image, ImagingMetadata]
        Tuple containing the Image object and metadata.
    """
    # Read image content and spatial information using SimpleITK
    image = sitk_load_dcm_series(dcm_dir=Path(dir))

    # If Q-SPECT, need to re-scale Data and possibly add Origin/Direction:
    if modality != "CT":

        # Remove redundant dimension
        image = squeeze_sitk_image_dimension(img=image)
        image = apply_qspect_dcm_origin(image=image, dir=dir)

        # QSPECT - Uses scale_factor provided by user, or attempts to get it from DICOM (if QSPECT)
        scale_factor = None

        if calibration_factor is not None:
            scale_factor = (calibration_factor, 0)

        try:
            image = apply_qspect_dcm_scaling(
                image=image, dir=dir, scale_factor=scale_factor
            )

        except AttributeError:
            logger.warning(
                "No calibration factor provided, Data might not be in BQ/ML ..."
            )

    # Load Meta Data using pydicom.
    meta = load_metadata(dir=dir, modality=modality)

    # Force Orthogonality of Patient Orientation
    image = force_orthogonality(image=image)

    # Display Origin and Orientation.
    logger.debug(
        f"Modality: {modality} -> Origin: {image.GetOrigin()}; Direction: {image.GetDirection()}"
    )

    return image, meta


def are_vectors_orthogonal(origin: List[float], tol: float = 1e-24):
    """Check if the patient orientation is given by orthogonal vectors.

    Returns True if a·b, a·c, and b·c are all within ±tol; otherwise False.

    Parameters
    ----------
    origin : List[float]
        Coordinates of Patient Orientation
    tol : float, optional
        Tolerance, by default 1e-8

    Returns
    -------
    _type_
        _description_
    """
    # split into three 3D vectors
    a = origin[0:3]
    b = origin[3:6]
    c = origin[6:9]

    def dot(u, v):
        return sum(ui * vi for ui, vi in zip(u, v))

    return abs(dot(a, b)) < tol and abs(dot(a, c)) < tol and abs(dot(b, c)) < tol


def force_orthogonality(image: SimpleITK.Image) -> SimpleITK.Image:
    """Force orthogonality of patient orientation vectors.

    Parameters
    ----------
    image : SimpleITK.Image
        Input image.

    Returns
    -------
    SimpleITK.Image
        Image with orthogonal orientation vectors.
    """
    if not are_vectors_orthogonal(image.GetDirection()):
        logger.warning("Patient Orientation Vectors are NOT orthogonal. Forcing...")
        prev_origin = image.GetDirection()
        new_origin = [round(vec_element) for vec_element in prev_origin]
        logger.debug(
            f">> Original Orientation: {prev_origin}, New Orientation: {new_origin} "
        )
        image.SetDirection(new_origin)
    else:
        prev_origin = image.GetDirection()
        new_origin = [round(vec_element) for vec_element in prev_origin]

    return image


def load_RTStruct(
    ref_dicom_ct_dir: str, rt_struct_file: str
) -> Dict[str, SimpleITK.Image]:
    """Load RTStruct Contours and Generate Masks.

    Parameters
    ----------
    ref_dicom_ct_dir : str
        Path to reference Dicom dir of CT slices associated with RTStruct
    rt_struct_file : str
        Path to RTStruct file.

    Returns
    -------
    Dict[str, SimpleITK.Image]
        A Dictionary containing each mask present in the RTStruct file.
    """

    def clean_roi_name(roi_name: str) -> str:
        cleaned_roi_name = (
            roi_name.replace(" ", "")
            .replace("-", "_")
            .replace("(", "_")
            .replace(")", "")
        )
        return cleaned_roi_name

    CT_folder = Path(ref_dicom_ct_dir)

    if not CT_folder.exists():
        raise FileNotFoundError(f"Folder {CT_folder.name} does not exists.")

    CT_sitk = force_orthogonality(image=sitk_load_dcm_series(dcm_dir=CT_folder))

    RT = RTStructBuilder.create_from(
        dicom_series_path=ref_dicom_ct_dir, rt_struct_path=rt_struct_file
    )

    roi_masks: Dict[str, SimpleITK.Image] = {}
    roi_names = RT.get_roi_names()

    # Clean names, as they might come with unsupported characters from third party software.
    for roi_name in roi_names:
        cleaned_roi_name = clean_roi_name(roi_name)
        mask = RT.get_roi_mask_by_name(roi_name)
        roi_masks[cleaned_roi_name] = itk_image_from_array(
            array=numpy.transpose(mask, axes=(2, 0, 1)), ref_image=CT_sitk, is_mask=True
        )

    return roi_masks


def resample_mask_to_target(
    mask_img: SimpleITK.Image, target_img: SimpleITK.Image
) -> SimpleITK.Image:
    """
    Resample a binary mask (originally from CT) to match a target ITK image in physical space (location/voxel spacing).

    Parameters
    ----------
    mask_img : SimpleITK.Image
      Binary CT mask (e.g. sitkUInt8 or sitkUInt16).
    target_img : SimpleITK.Image
        The reference image (SPECT) whose spacing, origin,
        direction, and size you want to match.

    Returns
    -------
    resampled_mask_img : SimpleITK.Image
        The mask resampled into the target image's space.
    resampled_mask_array : np.ndarray
        A NumPy array of shape (z, y, x) aligned exactly with
        the SimpleITK target image.
    """
    # Quick geometry check: if mask and target already match, skip resampling
    if (
        mask_img.GetSize() == target_img.GetSize()
        and mask_img.GetSpacing() == target_img.GetSpacing()
        and mask_img.GetOrigin() == target_img.GetOrigin()
        and mask_img.GetDirection() == target_img.GetDirection()
    ):
        return mask_img

    # ensure mask is of an integer type suitable for NN interpolation
    mask_cast = SimpleITK.Cast(mask_img, SimpleITK.sitkUInt8)

    resampler = SimpleITK.ResampleImageFilter()
    # copy geometry from target
    resampler.SetReferenceImage(target_img)
    # nearest‐neighbor to keep it binary
    resampler.SetInterpolator(SimpleITK.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(mask_cast)


def load_and_resample_RT_to_target(
    ref_dicom_ct_dir: str, rt_struct_file: str, target_img: SimpleITK.Image
) -> Tuple[Dict[str, SimpleITK.Image], Dict[str, SimpleITK.Image]]:
    """Load and resample RT structure to target image.

    Parameters
    ----------
    ref_dicom_ct_dir : str
        Directory containing reference DICOM CT files.
    rt_struct_file : str
        Path to RT structure file.
    target_img : SimpleITK.Image
        Target image for resampling.

    Returns
    -------
    Tuple[Dict[str, SimpleITK.Image], Dict[str, SimpleITK.Image]]
        Reference (CT) and Resampleda (SPECT) Masks from RTStruct.
    """
    ref_masks = load_RTStruct(
        ref_dicom_ct_dir=ref_dicom_ct_dir, rt_struct_file=rt_struct_file
    )

    resampled_masks: Dict[str, SimpleITK.Image] = {}

    for mask_name, mask_image in ref_masks.items():
        logger.debug(f"Resampling Masks: {mask_name} ...")
        resampled_masks[mask_name] = resample_mask_to_target(
            mask_img=mask_image, target_img=target_img
        )

    return ref_masks, resampled_masks


def load_and_register_RT_to_target(
    ref_dicom_ct_dir: str, rt_struct_file: str, target_img: SimpleITK.Image
) -> Tuple[Dict[str, SimpleITK.Image], Dict[str, SimpleITK.Image]]:
    """Load and register RT structure to target image.

    Parameters
    ----------
    ref_dicom_ct_dir : str
        Directory containing reference DICOM CT files.
    rt_struct_file : str
        Path to RT structure file.
    target_img : SimpleITK.Image
        _description_

    Returns
    -------
    Tuple[Dict[str, SimpleITK.Image], Dict[str, SimpleITK.Image]]
        Reference (CT) and Resampleda (SPECT) Masks from RTStruct.
    """
    ref_masks = load_RTStruct(
        ref_dicom_ct_dir=ref_dicom_ct_dir, rt_struct_file=rt_struct_file
    )
    ref_ct, _ = load_from_dicom_dir(dir=ref_dicom_ct_dir, modality="CT")

    ref_ct = SimpleITK.Cast(ref_ct, SimpleITK.sitkFloat32)
    target_img = SimpleITK.Cast(target_img, SimpleITK.sitkFloat32)

    # Register:
    _, transform = register_ct_to_spect(ct_image=ref_ct, spect_image=target_img)

    resampled_masks: Dict[str, SimpleITK.Image] = {}

    for mask_name, mask_image in ref_masks.items():
        logger.debug(f"Registering Masks: {mask_name} ...")
        resampled_masks[mask_name] = transform_ct_mask_to_spect(
            mask=mask_image, spect=target_img, transform=transform
        )

    return ref_masks, resampled_masks


def resample_to_target(
    source_img: SimpleITK.Image,
    target_img: SimpleITK.Image,
    default_value: float = -1000.0,
) -> SimpleITK.Image:
    """Resample source_img to match the grid of target_image.

    Matches origin, spacing, direction, and size of target_image using the
    SimpleITK Linear interpolator.

    Parameters
    ----------
    source_img : sitk.Image
        The image to be resampled.
    target_img : sitk.Image
        The reference image defining the desired grid.
    default_value : float
        Pixel value for voxels outside source_img domain. Defaults to CT air values.

    Returns
    -------
    sitk.Image
        The resampled image.
    """
    # Set up the resampler
    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetReferenceImage(target_img)
    resampler.SetInterpolator(SimpleITK.sitkLinear)
    resampler.SetDefaultPixelValue(default_value)

    # Use an identity transform to align in physical space
    identity = SimpleITK.Transform(source_img.GetDimension(), SimpleITK.sitkIdentity)
    resampler.SetTransform(identity)

    # Perform resampling
    resampled_img = resampler.Execute(source_img)
    return resampled_img


def ensure_masks_disconnect(
    original_masks: Dict[str, numpy.ndarray],
) -> Dict[str, numpy.ndarray]:
    """Ensure masks are disconnected by resolving overlaps.

    Args:
        original_masks (Dict[str, numpy.ndarray]): Dictionary of mask arrays.

    Returns
    -------
    Dict[str, numpy.ndarray]
        Dictionary of disconnected masks.
    """
    if len(original_masks) == 0:
        return original_masks

    # Create multi-label array from all masks. Each mask is a different ID, overwriting the previous one if there are overlaps.
    original_masks_names = [region for region in original_masks.keys()]
    all_original_mask = numpy.zeros(
        original_masks[original_masks_names[0]].shape, dtype=numpy.int16
    )

    id = 1
    final_regions: List[str] = []
    for region, mask in original_masks.items():
        all_original_mask[numpy.where(mask)] = id
        final_regions.append(region)
        id += 1

    # Split array into individual masks arrays.
    final_masks: Dict[str, numpy.ndarray] = {}
    for id_final in range(1, id):
        final_masks[final_regions[id_final - 1]] = numpy.where(
            all_original_mask == id_final, True, False
        )

    return final_masks


def extract_masks(
    time_id: int,
    mask_dataset: Dict[int, Dict[str, numpy.ndarray]],
    requested_rois: List[str],
) -> Dict[str, numpy.ndarray]:
    """Extract masks from NM dataset, according to user-defined list. Enforce that masks are disconnected.

    Constrains:
    - Tumors are always going to be removed from organs.
    - For non-tumor regions with overlapping voxels, the newly added region will prevail.

    Returns
    -------
    Dict[str, numpy.ndarray]
        Dictionary of compliant masks.
    """
    # Available Mask Names:
    exclude = ["WholeBody", "RemainderOfBody"]
    if "BoneMarrow" not in mask_dataset[0]:
        exclude.append("BoneMarrow")

    mask_names = [name for name in requested_rois if name not in exclude]

    # Disconnect tumor masks (if there is any overlap among them)
    tumor_labels = [region for region in requested_rois if "Lesion" in region]
    tumors_masks = ensure_masks_disconnect(
        original_masks={
            tumor_label: mask_dataset[time_id][tumor_label]
            for tumor_label in tumor_labels
        }
    )

    # Get mask of total tumor burden
    tumor_burden_mask = numpy.zeros_like(mask_dataset[time_id][requested_rois[0]])

    for _, tumor_mask in tumors_masks.items():
        tumor_burden_mask[numpy.where(tumor_mask)] = True

    # Remove tumor from normal tissue regions.
    non_tumor_masks_aggregate: Dict[str, numpy.ndarray] = {
        region: (
            numpy.clip(
                (mask_dataset[time_id][region]).astype(numpy.int8)
                - tumor_burden_mask.astype(numpy.int8),
                0,
                1,
            )
        ).astype(bool)
        for region in mask_names
        if region not in tumor_labels
    }

    corrected_masks = ensure_masks_disconnect(original_masks=non_tumor_masks_aggregate)
    corrected_masks.update(tumors_masks)

    # Generate Remainder of Body Mask:
    remainder = (
        numpy.ones(tumor_burden_mask.shape, dtype=numpy.int8)
        if "WholeBody" not in mask_dataset[time_id].keys()
        else mask_dataset[time_id]["WholeBody"].astype(numpy.int8)
    )

    for _, mask in corrected_masks.items():
        remainder -= mask

    corrected_masks["RemainderOfBody"] = (
        numpy.clip(remainder, 0, 1) != 0
    )  # Cast to boolean.

    # Generate Whole Body Mask:
    whole_body = numpy.zeros_like(remainder)
    for _, mask in corrected_masks.items():
        whole_body += mask

    corrected_masks["WholeBody"] = numpy.clip(whole_body, 0, 1) != 0  # Cast to boolean.

    return corrected_masks


def jaccard_index(mask_1: numpy.ndarray, mask_2: numpy.ndarray) -> float:
    """Compute the Jaccard index between two binary masks.

    Args:
        mask_1 (numpy.ndarray): First binary mask as a numpy array where 1s represent the mask and 0s represent the background.
        mask_2 (numpy.ndarray): Second binary mask as a numpy array similar to mask_1.

    Returns
    -------
    float
        Jaccard index value.
    """
    intersection = numpy.logical_and(mask_1, mask_2)
    union = numpy.logical_or(mask_1, mask_2)
    jaccard = numpy.sum(intersection) / numpy.sum(union)

    return jaccard
