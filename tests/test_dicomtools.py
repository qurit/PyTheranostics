import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import UID, ExplicitVRLittleEndian

from pytheranostics.dicomtools.dicomtools import (
    DicomModify,
    _apply_input_pixel_mapping,
    _get_frame_duration_seconds,
    _get_ge_pixel_scale,
    _normalize_dicom_manufacturer,
    _update_int16_pixel_metadata,
)


def _write_minimal_ge_spect(
    path,
    pixel_scale=None,
    sop_instance_uid="1.2.826.0.1.3680043.8.498.1",
    series_instance_uid="1.2.826.0.1.3680043.8.498.2",
):
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    dataset = FileDataset(path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.Manufacturer = "GE MEDICAL SYSTEMS"
    dataset.Modality = "NM"
    dataset.PatientID = "TEST001"
    dataset.SeriesDate = "20220101"
    dataset.SeriesTime = "090000"
    dataset.AcquisitionDate = "20220101"
    dataset.AcquisitionTime = "090000"
    dataset.SeriesDescription = "RECON_COUNTS"
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.SeriesInstanceUID = series_instance_uid
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
    assert dataset[0x00280106].VR == "SS"
    assert dataset[0x00280107].VR == "SS"


def test_update_int16_pixel_metadata_resolves_existing_ambiguous_pixel_value_vr():
    dataset = Dataset()
    dataset.add_new((0x0028, 0x0106), "US or SS", 0)
    dataset.add_new((0x0028, 0x0107), "US or SS", 1)
    pixels = np.array([[-2, 0, 3]], dtype=np.int16)

    _update_int16_pixel_metadata(dataset, pixels)

    assert dataset[0x00280106].VR == "SS"
    assert dataset[0x00280107].VR == "SS"


def test_apply_input_pixel_mapping_defaults_to_identity():
    pixels = np.array([1, 2], dtype=np.int16)

    mapped, source, slope, intercept = _apply_input_pixel_mapping(Dataset(), pixels)

    np.testing.assert_array_equal(mapped, [1.0, 2.0])
    assert source == "Identity"
    assert slope == 1.0
    assert intercept == 0.0


def test_apply_input_pixel_mapping_uses_linear_rwvm():
    dataset = Dataset()
    mapping = Dataset()
    mapping.RealWorldValueSlope = 0.25
    mapping.RealWorldValueIntercept = 2.0
    dataset.RealWorldValueMappingSequence = Sequence([mapping])

    mapped, source, slope, intercept = _apply_input_pixel_mapping(
        dataset, np.array([4, 8], dtype=np.int16)
    )

    np.testing.assert_array_equal(mapped, [3.0, 4.0])
    assert source == "RealWorldValueMappingSequence"
    assert slope == 0.25
    assert intercept == 2.0


def test_apply_input_pixel_mapping_rwvm_takes_precedence_over_rescale():
    dataset = Dataset()
    mapping = Dataset()
    mapping.RealWorldValueSlope = 2.0
    mapping.RealWorldValueIntercept = 1.0
    dataset.RealWorldValueMappingSequence = Sequence([mapping])
    dataset.RescaleSlope = "100"
    dataset.RescaleIntercept = "1000"

    mapped, source, _, _ = _apply_input_pixel_mapping(dataset, np.array([3]))

    np.testing.assert_array_equal(mapped, [7.0])
    assert source == "RealWorldValueMappingSequence"


@pytest.mark.parametrize(
    ("slope", "intercept", "expected"),
    [
        ("2", "3", [5.0, 7.0]),
        ("2", None, [2.0, 4.0]),
        (None, "3", [4.0, 5.0]),
    ],
)
def test_apply_input_pixel_mapping_uses_rescale_defaults(slope, intercept, expected):
    dataset = Dataset()
    if slope is not None:
        dataset.RescaleSlope = slope
    if intercept is not None:
        dataset.RescaleIntercept = intercept

    mapped, source, _, _ = _apply_input_pixel_mapping(dataset, np.array([1, 2]))

    np.testing.assert_array_equal(mapped, expected)
    assert source == "Rescale"


def test_apply_input_pixel_mapping_accepts_identical_rwvm_items():
    dataset = Dataset()
    mappings = []
    for _ in range(2):
        mapping = Dataset()
        mapping.RealWorldValueSlope = 0.5
        mapping.RealWorldValueIntercept = 0.0
        mappings.append(mapping)
    dataset.RealWorldValueMappingSequence = Sequence(mappings)

    mapped, _, _, _ = _apply_input_pixel_mapping(dataset, np.array([2, 4]))

    np.testing.assert_array_equal(mapped, [1.0, 2.0])


def test_apply_input_pixel_mapping_rejects_conflicting_rwvm_items():
    dataset = Dataset()
    first = Dataset()
    first.RealWorldValueSlope = 1.0
    first.RealWorldValueIntercept = 0.0
    second = Dataset()
    second.RealWorldValueSlope = 2.0
    second.RealWorldValueIntercept = 0.0
    dataset.RealWorldValueMappingSequence = Sequence([first, second])

    with pytest.raises(ValueError, match="conflicting linear mappings"):
        _apply_input_pixel_mapping(dataset, np.array([1]))


def test_apply_input_pixel_mapping_rejects_incomplete_rwvm():
    dataset = Dataset()
    mapping = Dataset()
    mapping.RealWorldValueSlope = 1.0
    dataset.RealWorldValueMappingSequence = Sequence([mapping])

    with pytest.raises(ValueError, match="RealWorldValueIntercept"):
        _apply_input_pixel_mapping(dataset, np.array([1]))


def test_apply_input_pixel_mapping_rejects_non_linear_rwvm():
    dataset = Dataset()
    mapping = Dataset()
    mapping.RealWorldValueLUTData = [1.0, 2.0]
    dataset.RealWorldValueMappingSequence = Sequence([mapping])

    with pytest.raises(ValueError, match="unsupported non-linear"):
        _apply_input_pixel_mapping(dataset, np.array([1]))


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
    assert unscaled_summary.loc[0, "input_mapping_source"] == "Identity"
    assert unscaled_summary.loc[0, "input_mapping_slope"] == 1.0
    assert unscaled_summary.loc[0, "input_mapping_intercept"] == 0.0
    assert unscaled_summary.loc[0, "ge_pixel_scale"] == 1.0
    assert scaled_summary.loc[0, "ge_pixel_scale"] == 4.0


def test_make_bqml_suv_applies_input_rwvm_once_and_reports_it(tmp_path):
    identity_path = tmp_path / "identity.dcm"
    mapped_path = tmp_path / "mapped.dcm"
    _write_minimal_ge_spect(identity_path)
    _write_minimal_ge_spect(mapped_path)

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
    identity = DicomModify(str(identity_path), CF=1.0)
    identity.make_bqml_suv(**conversion_kwargs)

    mapped = DicomModify(str(mapped_path), CF=1.0)
    input_mapping = Dataset()
    input_mapping.RealWorldValueSlope = 0.25
    input_mapping.RealWorldValueIntercept = 0.0
    mapped.ds.RealWorldValueMappingSequence = Sequence([input_mapping])
    summary = mapped.make_bqml_suv(**conversion_kwargs)

    identity_output_slope = float(
        identity.ds.RealWorldValueMappingSequence[0].RealWorldValueSlope
    )
    mapped_output_slope = float(
        mapped.ds.RealWorldValueMappingSequence[0].RealWorldValueSlope
    )
    assert mapped_output_slope == pytest.approx(identity_output_slope * 0.25)
    assert summary.loc[0, "input_mapping_source"] == "RealWorldValueMappingSequence"
    assert summary.loc[0, "input_mapping_slope"] == 0.25
    assert summary.loc[0, "input_mapping_intercept"] == 0.0


def test_make_bqml_suv_uses_acquisition_date_and_time_for_scan_datetime(tmp_path):
    source_path = tmp_path / "acquisition_datetime.dcm"
    _write_minimal_ge_spect(source_path)

    image = DicomModify(str(source_path), CF=1.0)
    image.ds.SeriesDate = "20220103"
    image.ds.SeriesTime = "120000"
    image.ds.AcquisitionDate = "20220102"
    image.ds.AcquisitionTime = "101112.25"

    summary = image.make_bqml_suv(
        weight=70,
        height=170,
        injection_date="20220101",
        pre_inj_activity=1000,
        pre_inj_time="0800",
        post_inj_activity=10,
        post_inj_time="0820",
        injection_time="0810",
    )

    assert summary.loc[0, "scan_datetime"].isoformat() == "2022-01-02T10:11:12.250000"


@pytest.mark.parametrize("missing_keyword", ["AcquisitionDate", "AcquisitionTime"])
def test_make_bqml_suv_requires_acquisition_date_and_time(tmp_path, missing_keyword):
    source_path = tmp_path / f"missing_{missing_keyword}.dcm"
    _write_minimal_ge_spect(source_path)

    image = DicomModify(str(source_path), CF=1.0)
    del image.ds[missing_keyword]

    with pytest.raises(ValueError, match=missing_keyword):
        image.make_bqml_suv(
            weight=70,
            height=170,
            injection_date="20220101",
            pre_inj_activity=1000,
            pre_inj_time="0800",
            post_inj_activity=10,
            post_inj_time="0820",
            injection_time="0810",
        )


def test_make_bqml_suv_falls_back_for_long_uid_prefix(tmp_path):
    source_path = tmp_path / "long_uid.dcm"
    source_sop_uid = "2.16.840.1.114362.1.12164994.27025353676.690008536.501.7081"
    source_series_uid = "2.16.840.1.114362.1.12164994.27025353676.690008536.501.7082"
    _write_minimal_ge_spect(
        source_path,
        sop_instance_uid=source_sop_uid,
        series_instance_uid=source_series_uid,
    )

    image = DicomModify(str(source_path), CF=1.0)
    with pytest.warns(RuntimeWarning, match="too long for generate_uid"):
        image.make_bqml_suv(
            weight=70,
            height=170,
            injection_date="20220101",
            pre_inj_activity=1000,
            pre_inj_time="0800",
            post_inj_activity=10,
            post_inj_time="0820",
            injection_time="0810",
        )

    new_sop_uid = str(image.ds.SOPInstanceUID)
    new_series_uid = str(image.ds.SeriesInstanceUID)
    assert UID(new_sop_uid).is_valid
    assert UID(new_series_uid).is_valid
    assert new_sop_uid != source_sop_uid
    assert new_series_uid != source_series_uid
    assert new_sop_uid != new_series_uid
    assert str(image.ds.file_meta.MediaStorageSOPInstanceUID) == new_sop_uid
