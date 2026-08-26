import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian

from pytheranostics.dicomtools.dicomtools import (
    DicomModify,
    _get_frame_duration_seconds,
    _get_ge_pixel_scale,
    _normalize_dicom_manufacturer,
    _update_int16_pixel_metadata,
)


def _write_minimal_ge_spect(path, pixel_scale=None):
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.Manufacturer = "GE MEDICAL SYSTEMS"
    dataset.Modality = "NM"
    dataset.PatientID = "TEST001"
    dataset.SeriesDate = "20220101"
    dataset.SeriesTime = "090000"
    dataset.AcquisitionTime = "090000"
    dataset.SeriesDescription = "RECON_COUNTS"
    dataset.SOPInstanceUID = "1.2.826.0.1.3680043.8.498.1"
    dataset.SeriesInstanceUID = "1.2.826.0.1.3680043.8.498.2"
    dataset.Rows = 1
    dataset.Columns = 1
    dataset.NumberOfFrames = 1
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.PixelSpacing = [10, 10]
    dataset.SliceThickness = 10
    dataset.CorrectedImage = ["ATTN"]
    dataset.PixelData = np.array([100], dtype=np.int16).tobytes()

    rotation = Dataset()
    rotation.ActualFrameDuration = 1000
    rotation.NumberOfFramesInRotation = 1
    dataset.RotationInformationSequence = Sequence([rotation])
    if pixel_scale is not None:
        dataset.add_new((0x0011, 0x103B), "DS", str(pixel_scale))

    dataset.save_as(path)


@pytest.mark.parametrize(
    ("manufacturer", "expected"),
    [
        ("SIEMENS", "siemens"),
        ("SIEMENS NM", "siemens"),
        ("Siemens Healthineers", "siemens"),
        ("ACME / SIEMENS Symbia", "siemens"),
        ("GE MEDICAL SYSTEMS", "ge"),
        ("GE Healthcare NM 870", "ge"),
    ],
)
def test_normalize_dicom_manufacturer_aliases(manufacturer, expected):
    dataset = Dataset()
    dataset.Manufacturer = manufacturer

    assert _normalize_dicom_manufacturer(dataset) == expected


def test_normalize_dicom_manufacturer_rejects_unknown_value():
    dataset = Dataset()
    dataset.Manufacturer = "Unknown Scanner Company"

    with pytest.raises(ValueError, match="Unsupported DICOM manufacturer"):
        _normalize_dicom_manufacturer(dataset)


def test_normalize_dicom_manufacturer_does_not_match_ge_inside_a_word():
    dataset = Dataset()
    dataset.Manufacturer = "Edge Imaging"

    with pytest.raises(ValueError, match="Unsupported DICOM manufacturer"):
        _normalize_dicom_manufacturer(dataset)


def test_get_frame_duration_seconds_prefers_valid_dicom_value():
    rotation_info = Dataset()
    rotation_info.ActualFrameDuration = 12000

    assert _get_frame_duration_seconds(rotation_info, "siemens", 15) == 12


def test_get_frame_duration_seconds_uses_opt_in_siemens_fallback():
    rotation_info = Dataset()

    with pytest.warns(RuntimeWarning, match="using siemens_frame_duration_fallback"):
        duration = _get_frame_duration_seconds(rotation_info, "siemens", 15)

    assert duration == 15


def test_get_frame_duration_seconds_remains_strict_without_fallback():
    with pytest.raises(ValueError, match="ActualFrameDuration"):
        _get_frame_duration_seconds(Dataset(), "siemens")


def test_get_frame_duration_seconds_does_not_apply_siemens_fallback_to_ge():
    with pytest.raises(ValueError, match="ActualFrameDuration"):
        _get_frame_duration_seconds(Dataset(), "ge", 15)


@pytest.mark.parametrize("fallback", [0, -1, float("nan"), "invalid"])
def test_get_frame_duration_seconds_rejects_invalid_fallback(fallback):
    rotation_info = Dataset()
    rotation_info.ActualFrameDuration = 12000

    with pytest.raises(ValueError, match="must be a positive number"):
        _get_frame_duration_seconds(rotation_info, "siemens", fallback)


def test_update_int16_pixel_metadata_removes_conventional_rescale_mapping():
    dataset = Dataset()
    dataset.RescaleSlope = "42.0"
    dataset.RescaleIntercept = "7.0"
    dataset.RescaleType = "US"
    pixels = np.array([[-2, 0, 3]], dtype=np.int16)

    _update_int16_pixel_metadata(dataset, pixels)

    assert "RescaleSlope" not in dataset
    assert "RescaleIntercept" not in dataset
    assert "RescaleType" not in dataset
    assert dataset.SmallestImagePixelValue == -2
    assert dataset.LargestImagePixelValue == 3


def test_get_ge_pixel_scale_defaults_to_one_when_tag_is_absent():
    assert _get_ge_pixel_scale(Dataset()) == 1.0


@pytest.mark.parametrize(("vr", "value"), [("DS", "2.5"), ("UN", b"4.0\x00")])
def test_get_ge_pixel_scale_reads_numeric_private_tag(vr, value):
    dataset = Dataset()
    dataset.add_new((0x0011, 0x103B), vr, value)

    with pytest.warns(RuntimeWarning, match="Applying GE Pixel Scale"):
        pixel_scale = _get_ge_pixel_scale(dataset)

    assert pixel_scale == pytest.approx(
        float(value.rstrip(b"\x00")) if isinstance(value, bytes) else float(value)
    )


@pytest.mark.parametrize("value", ["invalid", "0", "-2", "nan", b"\xff"])
def test_get_ge_pixel_scale_rejects_invalid_values(value):
    dataset = Dataset()
    vr = "UN" if isinstance(value, bytes) else "LO"
    dataset.add_new((0x0011, 0x103B), vr, value)

    with pytest.raises(ValueError, match="GE Pixel Scale DICOM tag"):
        _get_ge_pixel_scale(dataset)


def test_make_bqml_suv_applies_ge_pixel_scale(tmp_path):
    unscaled_path = tmp_path / "unscaled.dcm"
    scaled_path = tmp_path / "scaled.dcm"
    _write_minimal_ge_spect(unscaled_path)
    _write_minimal_ge_spect(scaled_path, pixel_scale=4)

    conversion_kwargs = {
        "weight": 70,
        "height": 170,
        "injection_date": "20220101",
        "pre_inj_activity": 1000,
        "pre_inj_time": "0800",
        "post_inj_activity": 10,
        "post_inj_time": "0820",
        "injection_time": "0810",
    }
    unscaled = DicomModify(str(unscaled_path), CF=1.0)
    unscaled_summary = unscaled.make_bqml_suv(**conversion_kwargs)
    scaled = DicomModify(str(scaled_path), CF=1.0)
    with pytest.warns(RuntimeWarning, match="Applying GE Pixel Scale"):
        scaled_summary = scaled.make_bqml_suv(**conversion_kwargs)

    unscaled_slope = float(
        unscaled.ds.RealWorldValueMappingSequence[0].RealWorldValueSlope
    )
    scaled_slope = float(scaled.ds.RealWorldValueMappingSequence[0].RealWorldValueSlope)
    assert scaled_slope == pytest.approx(unscaled_slope * 4)
    assert unscaled_summary.loc[0, "ge_pixel_scale"] == 1.0
    assert scaled_summary.loc[0, "ge_pixel_scale"] == 4.0
