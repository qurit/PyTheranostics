"""Utility functions for reading and modifying nuclear medicine DICOM files."""

import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import SimpleITK
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid

from pytheranostics.shared.radioactive_decay import get_activity_at_injection

_MANUFACTURER_ALIASES = {
    "siemens": "siemens",
    "siemens healthineers": "siemens",
    "siemens medical solutions": "siemens",
    "siemens medical systems": "siemens",
    "siemens nm": "siemens",
    "ge": "ge",
    "ge healthcare": "ge",
    "ge medical systems": "ge",
}


class DicomModify:
    """Edit DICOM headers and pixel data for quantitative SPECT studies.

    Parameters
    ----------
    fname : str
        Path to the source DICOM file.
    CF : float
        Camera calibration factor used to convert count-rate data to activity
        concentration.
    """

    def __init__(self, fname: str, CF: float) -> None:
        """Load a DICOM file and store calibration information.

        Parameters
        ----------
        fname : str
            Path to the source DICOM file.
        CF : float
            Camera calibration factor used to convert count-rate data to
            activity concentration, in units of MBq/(counts/s).
        """
        self.ds: FileDataset = pydicom.dcmread(fname)
        self.CF: float = CF
        self.fname: str = fname

    def make_bqml_suv(
        self,
        weight: float,
        height: float,
        injection_date: str,
        pre_inj_activity: float,
        pre_inj_time: str,
        post_inj_activity: float,
        post_inj_time: str,
        injection_time: str,
        activity_meter_scale_factor: float = 1.0,
        half_life: float = 574300,
        radiopharmaceutical: str = "Lutetium-PSMA-617",
        n_detectors: int = 2,
        siemens_frame_duration_fallback: Optional[float] = None,
    ) -> pd.DataFrame:
        """Convert raw counts to BQML/SUV units and update DICOM metadata.

        Parameters
        ----------
        weight : float
            Patient weight in kilograms.
        height : float
            Patient height in centimeters.
        injection_date : str
            Injection date formatted as ``YYYYMMDD``.
        pre_inj_activity : float
            Pre-injection syringe activity in MBq.
        pre_inj_time : str
            Time of the pre-injection syringe measurement formatted as
            ``HHMM``.
        post_inj_activity : float
            Post-injection syringe activity in MBq.
        post_inj_time : str
            Time of the post-injection syringe measurement formatted as
            ``HHMM``.
        injection_time : str
            Injection time formatted as ``HHMM``.
        activity_meter_scale_factor : float
            Scale factor applied to the calculated injected activity. Default is 1.0.
        half_life : float, optional
            Radionuclide half-life in seconds. The default is 574300 for Lu-177.
        radiopharmaceutical : str, optional
            Radiopharmaceutical name written to the DICOM metadata. The default
            is ``"Lutetium-PSMA-617"``.
        n_detectors : int, optional
            Number of detectors used to correct Siemens projection counts. The
            default is 2.
        siemens_frame_duration_fallback : float, optional
            Frame duration in seconds to use for Siemens data when
            ``ActualFrameDuration`` is missing or invalid. By default no
            fallback is used and invalid DICOM metadata raises ``ValueError``.

        Returns
        -------
        pandas.DataFrame
            One-row summary of patient, injection, and scan timing metadata.

        Notes
        -----
        This method mutates ``self.ds`` in place. Call :meth:`save` to write the
        modified dataset to disk.
        """
        # Half-life is in seconds

        if n_detectors <= 0:
            raise ValueError(f"n_detectors must be positive; got {n_detectors}.")

        manufacturer = _normalize_dicom_manufacturer(self.ds)

        # Siemens has an issue setting up the times. We are using the Acquisition time which is the time of the start of the last bed to harmonize.
        if manufacturer == "siemens":
            acquisition_time = _require_dicom_text(self.ds, "AcquisitionTime")
            self.ds.SeriesTime = acquisition_time
            self.ds.ContentTime = acquisition_time
        elif manufacturer == "ge":  # i think it applies to ge as well
            acquisition_time = _require_dicom_text(self.ds, "AcquisitionTime")
            self.ds.SeriesTime = acquisition_time
            self.ds.ContentTime = acquisition_time

        # Get the frame duration in seconds
        rotation_info = _require_dicom_sequence_item(
            self.ds, "RotationInformationSequence"
        )
        frame_duration = _get_frame_duration_seconds(
            rotation_info,
            manufacturer,
            siemens_frame_duration_fallback,
        )

        # get number of projections because manufacturers scale by this in the dicomfile
        frames_in_rotation = _require_positive_dicom_int(
            rotation_info,
            "NumberOfFramesInRotation",
            context="RotationInformationSequence[0]",
        )
        if manufacturer == "siemens":
            n_proj = frames_in_rotation * n_detectors
        elif manufacturer == "ge":
            n_proj = frames_in_rotation
        else:
            raise ValueError(
                "Unsupported DICOM manufacturer for projection scaling: "
                f"{_require_dicom_text(self.ds, 'Manufacturer')}"
            )

        # get voxel volume in ml
        vox_vol = _get_voxel_volume_ml(self.ds)

        # Get image in Bq/ml
        A = self.ds.pixel_array.astype(np.float64)
        A = A / (frame_duration * n_proj) * self.CF * 1e6 / vox_vol

        slope, intercept = dicom_slope_intercept(A)

        # update the PixelData
        # GE DICOM is signed, so use int16.
        A = ((A - intercept) / slope).astype(np.int16)

        # bring the new image to the pixel bytes
        self.ds.PixelData = A.tobytes()
        _update_int16_pixel_metadata(self.ds, A, slope, intercept)

        # update DICOM tags
        # self.ds.Units = 'BQML'
        series_description = _require_dicom_text(self.ds, "SeriesDescription")
        self.ds.SeriesDescription = "QSPECT_" + series_description

        # add the RealWorldValueMappingSequence tag [0040,9096]
        self.ds.add_new((0x0040, 0x9096), "SQ", [])
        self.ds.RealWorldValueMappingSequence += [Dataset(), Dataset()]

        for i in range(2):
            self.ds.RealWorldValueMappingSequence[i].RealWorldValueIntercept = intercept
            self.ds.RealWorldValueMappingSequence[i].RealWorldValueSlope = slope
            self.ds.RealWorldValueMappingSequence[i].RealWorldValueLastValueMapped = (
                int(A.max())
            )
            self.ds.RealWorldValueMappingSequence[i].RealWorldValueFirstValueMapped = (
                int(A.min())
            )

            self.ds.RealWorldValueMappingSequence[i].LUTLabel = "BQML"
            self.ds.RealWorldValueMappingSequence[i].add_new((0x0040, 0x08EA), "SQ", [])
            self.ds.RealWorldValueMappingSequence[i].MeasurementUnitsCodeSequence += [
                Dataset()
            ]
            self.ds.RealWorldValueMappingSequence[i].MeasurementUnitsCodeSequence[
                0
            ].CodeValue = "Bq/ml"

        # add info for SUV
        self.ds.PatientWeight = str(weight)  # in kg
        self.ds.PatientSize = str(height / 100)  # in m

        self.ds.DecayCorrection = "START"
        _prepend_corrected_image_value(self.ds, "DECY")

        self.ds.add_new((0x0054, 0x0016), "SQ", [])
        self.ds.RadiopharmaceuticalInformationSequence += [Dataset()]

        # values for net injected activity and injection date and time
        start_datetime, total_injected_activity = get_activity_at_injection(
            injection_date,
            pre_inj_activity,
            pre_inj_time,
            post_inj_activity,
            post_inj_time,
            injection_time,
            half_life=half_life,
        )
        total_injected_activity = total_injected_activity * activity_meter_scale_factor

        series_date = _require_dicom_text(self.ds, "SeriesDate")
        series_time = _require_dicom_text(self.ds, "SeriesTime")
        scan_datetime = _parse_dicom_date_time(
            series_date,
            series_time,
            context="SeriesDate/SeriesTime",
        )
        delta_scan_inj = (scan_datetime - start_datetime).total_seconds() / (
            60 * 60 * 24
        )

        pre_inj_datetime = datetime.strptime(
            injection_date + pre_inj_time, "%Y%m%d%H%M"
        )
        post_inj_datetime = datetime.strptime(
            injection_date + post_inj_time, "%Y%m%d%H%M"
        )

        inj_dic = {
            "patient_id": [_require_dicom_text(self.ds, "PatientID")],
            "weight_kg": [weight],
            "height_cm": [height],
            "pre_inj_activity_MBq": [pre_inj_activity],
            "pre_inj_datetime": [pre_inj_datetime],
            "post_inj_activity_MBq": [post_inj_activity],
            "post_inj_datetime": [post_inj_datetime],
            "injected_activity_MBq": [total_injected_activity],
            "injection_datetime": [start_datetime],
            "scan_datetime": [scan_datetime],
            "delta_t_days": [delta_scan_inj],
        }
        inj_df = pd.DataFrame(data=inj_dic)

        self.ds.RadiopharmaceuticalInformationSequence[0].Radiopharmaceutical = (
            radiopharmaceutical
        )
        self.ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalVolume = ""
        self.ds.RadiopharmaceuticalInformationSequence[
            0
        ].RadiopharmaceuticalStartTime = start_datetime.strftime("%H%M%S.%f")
        self.ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose = str(
            round(total_injected_activity, 4)
        )
        self.ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife = str(
            half_life
        )
        self.ds.RadiopharmaceuticalInformationSequence[
            0
        ].RadionuclidePositronFraction = ""
        self.ds.RadiopharmaceuticalInformationSequence[
            0
        ].RadiopharmaceuticalStartDateTime = start_datetime.strftime("%Y%m%d%H%M%S.%f")

        # for storing as new series data
        sop_ins_uid = _require_dicom_text(self.ds, "SOPInstanceUID")
        self.ds.SOPInstanceUID = _increment_uid_suffix(sop_ins_uid, "SOPInstanceUID")

        ser_ins_uid = _require_dicom_text(self.ds, "SeriesInstanceUID")
        prefix = _uid_prefix_for_generated_uid(ser_ins_uid, "SeriesInstanceUID")
        self.ds.SeriesInstanceUID = generate_uid(prefix=prefix)

        # self.ds.MediaStorageSOPInstaceUID
        return inj_df

    def save(self) -> None:
        """Persist the modified dataset alongside the original file.

        Notes
        -----
        The output path is generated by appending ``"_out"`` to the input file
        stem and preserving the original suffix.
        """
        path = Path(self.fname)
        output_path = path.with_name(f"{path.stem}_out{path.suffix}")
        self.ds.save_as(output_path)


def dicom_slope_intercept(img: np.ndarray) -> Tuple[float, float]:
    """Calculate GE-style slope and intercept values.

    Parameters
    ----------
    img : numpy.ndarray
        Floating-point image array to be quantized into signed 16-bit DICOM
        pixel data.

    Returns
    -------
    tuple of float
        Slope and intercept used to recover real-world voxel values from the
        stored integer pixel data.

    Notes
    -----
    GE PET images are stored as signed ``int16`` values with magnitude limited
    to 32767. The computed slope maps the largest absolute voxel value to this
    range once quantized, while the intercept remains zero.
    """
    max_val = np.max(img)
    min_val = np.min(img)

    slope = np.float32(max(max_val, -min_val) / 32767)
    intercept = 0  # GE has assigned it to zero

    return float(slope), float(intercept)


def _update_int16_pixel_metadata(
    ds: FileDataset,
    pixel_array: np.ndarray,
    slope: float,
    intercept: float,
) -> None:
    """Update DICOM pixel metadata for signed 16-bit stored pixel data.

    Parameters
    ----------
    ds : pydicom.dataset.FileDataset
        Dataset whose ``PixelData`` has been rewritten.
    pixel_array : numpy.ndarray
        Signed 16-bit stored-value pixel array written to ``PixelData``.
    slope : float
        Rescale slope that maps stored pixel values back to real-world values.
    intercept : float
        Rescale intercept that maps stored pixel values back to real-world
        values.
    """
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.SmallestImagePixelValue = int(pixel_array.min())
    ds.LargestImagePixelValue = int(pixel_array.max())
    ds.RescaleSlope = str(slope)
    ds.RescaleIntercept = str(intercept)


def _require_dicom_value(
    ds: Dataset,
    keyword: str,
    context: str = "DICOM dataset",
) -> Any:
    """Return a required DICOM value or raise a clear validation error."""
    if not hasattr(ds, keyword):
        raise ValueError(f"Required DICOM tag '{keyword}' is missing from {context}.")

    value = getattr(ds, keyword)
    if value is None:
        raise ValueError(f"Required DICOM tag '{keyword}' is empty in {context}.")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"Required DICOM tag '{keyword}' is empty in {context}.")

    return value


def _get_frame_duration_seconds(
    rotation_info: Dataset,
    manufacturer: str,
    siemens_fallback: Optional[float] = None,
) -> float:
    """Return the DICOM frame duration, with an opt-in Siemens fallback.

    ``ActualFrameDuration`` is stored in milliseconds, while the fallback is
    supplied in seconds to match the value used by the conversion calculation.
    """
    if siemens_fallback is not None:
        try:
            siemens_fallback = float(siemens_fallback)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "siemens_frame_duration_fallback must be a positive number "
                f"of seconds; got {siemens_fallback!r}."
            ) from exc
        if not np.isfinite(siemens_fallback) or siemens_fallback <= 0:
            raise ValueError(
                "siemens_frame_duration_fallback must be a positive number "
                f"of seconds; got {siemens_fallback!r}."
            )

    try:
        duration_ms = _require_positive_dicom_float(
            rotation_info,
            "ActualFrameDuration",
            context="RotationInformationSequence[0]",
        )
    except ValueError:
        if manufacturer != "siemens" or siemens_fallback is None:
            raise
        warnings.warn(
            "Siemens ActualFrameDuration is missing or invalid; using "
            f"siemens_frame_duration_fallback={siemens_fallback} seconds.",
            RuntimeWarning,
            stacklevel=2,
        )
        return siemens_fallback

    return duration_ms / 1000


def _require_dicom_text(
    ds: Dataset,
    keyword: str,
    context: str = "DICOM dataset",
) -> str:
    """Return a required DICOM value as stripped text."""
    value = _require_dicom_value(ds, keyword, context)
    text = str(value).strip()
    if not text:
        raise ValueError(f"Required DICOM tag '{keyword}' is empty in {context}.")

    return text


def _normalize_dicom_manufacturer(ds: Dataset) -> str:
    """Return the supported canonical manufacturer name for a DICOM dataset."""
    manufacturer = _require_dicom_text(ds, "Manufacturer")
    normalized = " ".join(manufacturer.casefold().replace("_", " ").split())

    if normalized in _MANUFACTURER_ALIASES:
        return _MANUFACTURER_ALIASES[normalized]

    # Manufacturer values commonly include a modality, division, or model name.
    if "siemens" in normalized:
        return "siemens"
    if re.search(r"(?:^|[^a-z0-9])ge(?:$|[^a-z0-9])", normalized):
        return "ge"

    supported = ", ".join(sorted(set(_MANUFACTURER_ALIASES.values())))
    raise ValueError(
        f"Unsupported DICOM manufacturer for QSPECT conversion: "
        f"{manufacturer!r}. Supported manufacturers: {supported}."
    )


def _require_dicom_float(
    ds: Dataset,
    keyword: str,
    context: str = "DICOM dataset",
) -> float:
    """Return a required DICOM value as a float."""
    value = _require_dicom_value(ds, keyword, context)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Required DICOM tag '{keyword}' in {context} must be numeric; "
            f"got {value!r}."
        ) from exc


def _require_dicom_int(
    ds: Dataset,
    keyword: str,
    context: str = "DICOM dataset",
) -> int:
    """Return a required DICOM value as an integer."""
    value = _require_dicom_value(ds, keyword, context)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Required DICOM tag '{keyword}' in {context} must be an integer; "
            f"got {value!r}."
        ) from exc


def _require_positive_dicom_float(
    ds: Dataset,
    keyword: str,
    context: str = "DICOM dataset",
) -> float:
    """Return a required DICOM value as a positive float."""
    value = _require_dicom_float(ds, keyword, context)
    if value <= 0:
        raise ValueError(
            f"Required DICOM tag '{keyword}' in {context} must be positive; "
            f"got {value}."
        )

    return value


def _require_positive_dicom_int(
    ds: Dataset,
    keyword: str,
    context: str = "DICOM dataset",
) -> int:
    """Return a required DICOM value as a positive integer."""
    value = _require_dicom_int(ds, keyword, context)
    if value <= 0:
        raise ValueError(
            f"Required DICOM tag '{keyword}' in {context} must be positive; "
            f"got {value}."
        )

    return value


def _require_dicom_sequence_item(
    ds: Dataset,
    keyword: str,
    context: str = "DICOM dataset",
    index: int = 0,
) -> Dataset:
    """Return a required item from a DICOM sequence."""
    sequence = _require_dicom_value(ds, keyword, context)
    try:
        item = sequence[index]
    except (IndexError, TypeError) as exc:
        raise ValueError(
            f"Required DICOM sequence '{keyword}' in {context} must contain "
            f"item {index}."
        ) from exc

    if not isinstance(item, Dataset):
        raise ValueError(
            f"Item {index} of DICOM sequence '{keyword}' in {context} must be "
            f"a Dataset; got {type(item).__name__}."
        )

    return item


def _get_voxel_volume_ml(ds: Dataset) -> float:
    """Return voxel volume in milliliters from required spacing tags."""
    pixel_spacing = _require_dicom_value(ds, "PixelSpacing")
    try:
        spacing_mm = np.asarray(pixel_spacing, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Required DICOM tag 'PixelSpacing' must contain numeric row and "
            f"column spacing values; got {pixel_spacing!r}."
        ) from exc

    if spacing_mm.size < 2:
        raise ValueError(
            "Required DICOM tag 'PixelSpacing' must contain row and column "
            f"spacing values; got {pixel_spacing!r}."
        )

    if np.any(spacing_mm[:2] <= 0):
        raise ValueError(
            "Required DICOM tag 'PixelSpacing' must contain positive row and "
            f"column spacing values; got {pixel_spacing!r}."
        )

    slice_thickness_mm = _require_positive_dicom_float(ds, "SliceThickness")
    voxel_dimensions_mm = np.append(spacing_mm[:2], slice_thickness_mm)
    voxel_volume_ml = float(np.prod(voxel_dimensions_mm / 10))
    if voxel_volume_ml <= 0:
        raise ValueError(
            "Voxel volume calculated from PixelSpacing and SliceThickness must "
            f"be positive; got {voxel_volume_ml} ml."
        )

    return voxel_volume_ml


def _parse_dicom_date_time(date: str, time_value: str, context: str) -> datetime:
    """Parse DICOM DA/TM text into a datetime."""
    for time_format in ("%H%M%S.%f", "%H%M%S", "%H%M"):
        try:
            return datetime.strptime(date + time_value, "%Y%m%d" + time_format)
        except ValueError:
            continue

    raise ValueError(
        f"DICOM {context} must be formatted as YYYYMMDD and HHMM[SS[.ffffff]]; "
        f"got {date!r} and {time_value!r}."
    )


def _increment_uid_suffix(uid: str, keyword: str) -> str:
    """Increment the final numeric component of a DICOM UID."""
    uid_parts = uid.split(".")
    if (
        not uid_parts
        or "" in uid_parts
        or not all(part.isdigit() for part in uid_parts)
    ):
        raise ValueError(
            f"Required DICOM tag '{keyword}' must be a dot-separated numeric "
            f"UID ending in a numeric component; got {uid!r}."
        )

    uid_parts[-1] = str(int(uid_parts[-1]) + 1)
    return ".".join(uid_parts)


def _uid_prefix_for_generated_uid(uid: str, keyword: str) -> str:
    """Return a UID prefix suitable for pydicom.uid.generate_uid."""
    uid_parts = uid.split(".")
    if len(uid_parts) < 2 or "" in uid_parts:
        raise ValueError(
            f"Required DICOM tag '{keyword}' must be a dot-separated UID with "
            f"at least two components; got {uid!r}."
        )
    if not all(part.isdigit() for part in uid_parts):
        raise ValueError(
            f"Required DICOM tag '{keyword}' must contain only numeric UID "
            f"components; got {uid!r}."
        )

    prefix = ".".join(uid_parts[:-1]) + "."
    if len(prefix) > 54:
        raise ValueError(
            f"UID prefix derived from DICOM tag '{keyword}' is too long for "
            f"generate_uid; got {len(prefix)} characters."
        )

    return prefix


def _prepend_corrected_image_value(ds: Dataset, value: str) -> None:
    """Prepend a CorrectedImage value, creating the tag if needed."""
    corrected_image = getattr(ds, "CorrectedImage", [])
    if corrected_image is None:
        values = []
    elif isinstance(corrected_image, str):
        values = [corrected_image]
    else:
        try:
            values = list(corrected_image)
        except TypeError as exc:
            raise ValueError(
                "DICOM tag 'CorrectedImage' must be text or a multi-value "
                f"sequence; got {corrected_image!r}."
            ) from exc

    if value not in values:
        values.insert(0, value)

    ds.CorrectedImage = values


def generate_basic_dcm_tags(
    img_size: Tuple[int, int, int],
    slice_thickness: float,
    name: str,
    description: str,
    direction: Tuple[float, ...],
    date: str,
    time: str,
) -> List[Any]:
    """Generate the minimal tag set needed for a synthetic DICOM series."""
    series_tag_values = [
        ("0008|0031", time),  # Series Time
        ("0008|0021", date),  # Series Date
        ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
        (
            "0020|000e",
            "1.2.826.0.1.3680043.2.1125." + date + ".1" + time,
        ),  # Series Instance UID
        (
            "0020|0037",
            "\\".join(
                map(
                    str,
                    (
                        direction[0],
                        direction[3],
                        direction[6],  # Image Orientation (Patient)
                        direction[1],
                        direction[4],
                        direction[7],
                    ),
                )
            ),
        ),
        ("0008|103e", description),  # Series Description
        # patient information
        ("0010|0010", name),  # Patient name
        ("0010|0020", name),  # Patient ID
        # image space information
        ("0028|0010", str(img_size[1])),  # rows
        ("0028|0011", str(img_size[2])),  # columns
        ("0018|0050", f"{slice_thickness: 1.3f}"),  # slice thickness
        ("0054|0081", str(img_size[0])),  # number of slices
    ]

    return series_tag_values


def numpy_to_dcm_basic(
    array: np.ndarray,
    voxel_spacing: Tuple[float, float, float],
    output_dir: Path,
    patien_name: str = "Patient",
    scale: int = 1,
) -> None:
    """Write a NumPy array as a basic DICOM series for visualization/testing.

    Notes
    -----
    Adapted from: R. Fedrigo et al., "Development of the Lymphatic System in the
    4D XCAT Phantom for Improved Multimodality Imaging Research," J. Nucl. Med.,
    62, 113 (2021).
    """
    # Create SimpleITK image from array
    array = array * scale
    sitk_image = SimpleITK.GetImageFromArray(array.astype(np.int16))
    sitk_image.SetSpacing(voxel_spacing)

    # Create output Folder
    output_dir.mkdir(exist_ok=True, parents=True)

    # Write the 3D image as a DCM Series
    writer = SimpleITK.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    direction = sitk_image.GetDirection()

    tag_values = generate_basic_dcm_tags(
        img_size=array.shape,
        slice_thickness=voxel_spacing[0],
        name=patien_name,
        description=patien_name,
        direction=direction,
        date=modification_date,
        time=modification_time,
    )
    # Loop through slices
    for i in range(sitk_image.GetDepth()):
        image_slice = sitk_image[:, :, i]

        # Tags shared by the series.
        for tag, value in tag_values:
            image_slice.SetMetaData(tag, value)

        # Slice specific tags.
        image_slice.SetMetaData(
            "0008|0012", time.strftime("%Y%m%d")
        )  # Instance Creation Date
        image_slice.SetMetaData(
            "0008|0013", time.strftime("%H%M%S")
        )  # Instance Creation Time

        # Setting the type to CT preserves the slice location.
        image_slice.SetMetaData("0008|0060", "CT")  # set the type as a PET image

        # (0020, 0032) image position patient determines the 3D spacing between slices.
        image_slice.SetMetaData(
            "0020|0032",
            "\\".join(map(str, sitk_image.TransformIndexToPhysicalPoint((0, 0, i)))),
        )  # Image Position (Patient)
        image_slice.SetMetaData("0020,0013", str(i))  # Instance Number

        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName(str(output_dir / f"{i}.dcm"))
        writer.Execute(image_slice)

    return None


def sitk_load_dcm_series(dcm_dir: Path) -> SimpleITK.Image:
    """Load a DICOM series using SimpleITK and return it as an image volume."""
    reader = SimpleITK.ImageSeriesReader()
    dcm_file_names = reader.GetGDCMSeriesFileNames(str(dcm_dir))
    reader.SetFileNames(dcm_file_names)

    return reader.Execute()
