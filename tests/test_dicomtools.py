import pytest
from pydicom.dataset import Dataset

from pytheranostics.dicomtools.dicomtools import _normalize_dicom_manufacturer


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
