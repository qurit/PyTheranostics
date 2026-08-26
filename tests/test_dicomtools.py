import numpy as np
import pytest
from pydicom.dataset import Dataset

from pytheranostics.dicomtools.dicomtools import (
    _get_frame_duration_seconds,
    _normalize_dicom_manufacturer,
    _update_int16_pixel_metadata,
)


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
