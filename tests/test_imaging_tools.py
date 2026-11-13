"""Tests for imaging tools utilities."""

import logging

import numpy as np
import pytest
import SimpleITK
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, NuclearMedicineImageStorage

from pytheranostics.imaging_tools import tools


def test_apply_qspect_scaling_logs_real_world_value_mapping_source(
    tmp_path, monkeypatch, caplog
):
    dicom_path = tmp_path / "qspect.dcm"
    dicom_path.touch()

    mapping = Dataset()
    mapping.RealWorldValueSlope = 2.0
    mapping.RealWorldValueIntercept = 3.0
    dataset = Dataset()
    dataset.Modality = "NM"
    dataset.DecayCorrection = "START"
    dataset.RealWorldValueMappingSequence = Sequence([mapping])
    monkeypatch.setattr(tools.pydicom, "dcmread", lambda *args, **kwargs: dataset)

    input_array = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    image = SimpleITK.GetImageFromArray(input_array)
    with caplog.at_level(logging.INFO, logger=tools.__name__):
        scaled = tools.apply_qspect_dcm_scaling(image, str(tmp_path))

    np.testing.assert_allclose(SimpleITK.GetArrayFromImage(scaled), input_array * 2 + 3)
    assert "obtained from DICOM RealWorldValueMappingSequence" in caplog.text


def _write_minimal_nm_spect(path, **overrides):
    """Write a synthetic single-frame NM DICOM with no patient data in it."""
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = NuclearMedicineImageStorage
    file_meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.8.498.11"

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = NuclearMedicineImageStorage
    dataset.SOPInstanceUID = "1.2.826.0.1.3680043.8.498.11"
    dataset.Modality = "NM"
    dataset.Manufacturer = "SIEMENS NM"
    dataset.PatientID = "TEST001"
    dataset.PatientName = "TEST^PATIENT"
    dataset.StudyDate = "20240115"
    dataset.AcquisitionDate = "20240115"
    dataset.AcquisitionTime = "113704.000000"
    dataset.Rows = 1
    dataset.Columns = 1
    dataset.NumberOfFrames = 1
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelSpacing = [4.8, 4.8]
    dataset.SliceThickness = 4.8
    dataset.PixelData = np.array([100], dtype=np.uint16).tobytes()

    for keyword, value in overrides.items():
        setattr(dataset, keyword, value)

    dataset.save_as(str(path))
    return dataset


def test_load_metadata_from_spect_folder(tmp_path):
    """Metadata extraction should read the identifying tags off an NM series."""
    single_case_dir = tmp_path / "spect_case"
    single_case_dir.mkdir()
    _write_minimal_nm_spect(single_case_dir / "case.dcm")

    meta = tools.load_metadata(str(single_case_dir), modality="Lu177_NM")
    assert meta.PatientID == "TEST001"
    assert meta.AcquisitionDate == "20240115"
    # DICOM lacks injected activity tag -> default should apply
    assert meta.Injected_Activity_MBq == 7400.0
    assert meta.Radionuclide == "Lu177"


def test_load_metadata_reads_injected_activity_from_dicom(tmp_path):
    """A RadionuclideTotalDose in the header should win over the default."""
    single_case_dir = tmp_path / "spect_case"
    single_case_dir.mkdir()

    radiopharm = Dataset()
    radiopharm.RadionuclideTotalDose = 7100.0
    _write_minimal_nm_spect(
        single_case_dir / "case.dcm",
        RadiopharmaceuticalInformationSequence=Sequence([radiopharm]),
    )

    meta = tools.load_metadata(str(single_case_dir), modality="Lu177_NM")
    assert meta.Injected_Activity_MBq == 7100.0


@pytest.mark.parametrize("is_mask", [True, False])
def test_itk_image_from_array_preserves_metadata(is_mask):
    """Array conversion should preserve spacing/origin/direction."""
    ref = SimpleITK.Image(2, 2, 2, SimpleITK.sitkFloat32)
    ref.SetSpacing((2.0, 2.0, 5.0))
    ref.SetOrigin((1.0, 1.0, -3.0))
    ref.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    ref.SetMetaData("0010|0010", "Test Patient")

    array = np.ones((2, 2, 2), dtype=np.uint8 if is_mask else np.float32)
    image = tools.itk_image_from_array(array, ref_image=ref, is_mask=is_mask)

    assert tuple(image.GetSpacing()) == pytest.approx(ref.GetSpacing())
    assert tuple(image.GetOrigin()) == pytest.approx(ref.GetOrigin())
    assert tuple(image.GetDirection()) == tuple(ref.GetDirection())
    assert image.GetMetaData("0010|0010") == "Test Patient"
    if is_mask:
        assert image.GetPixelID() == SimpleITK.sitkUInt8
    else:
        assert image.GetPixelID() == ref.GetPixelID()
