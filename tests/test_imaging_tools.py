"""Tests for imaging tools utilities."""

import shutil

import numpy as np
import pytest
import SimpleITK

from pytheranostics.imaging_tools import tools


def test_load_metadata_from_sample_spect_folder(spect_example_dir, tmp_path):
    """Ensure metadata extraction works on bundled DICOM samples."""
    single_case_dir = tmp_path / "spect_case"
    single_case_dir.mkdir()
    shutil.copy(spect_example_dir / "016.dcm", single_case_dir / "case.dcm")

    meta = tools.load_metadata(str(single_case_dir), modality="Lu177_NM")
    assert meta.PatientID == "PR21-CAVA-0016"
    assert meta.AcquisitionDate == "20220617"
    # DICOM lacks injected activity tag -> default should apply
    assert meta.Injected_Activity_MBq == 7400.0
    assert meta.Radionuclide == "Lu177"


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
