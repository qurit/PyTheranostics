"""Tests for data access via importlib.resources-based helpers."""

import numpy as np

from pytheranostics.dosimetry.dvk import DoseVoxelKernel
from pytheranostics.dosimetry.olinda import load_phantom_mass


def test_load_phantom_mass_returns_expected_value():
    """The packaged phantom mass table should include standard organs."""
    liver_mass = load_phantom_mass(gender="Male", organ="Liver")
    assert liver_mass == 1800  # matches bundled phantom data


def test_dose_voxel_kernel_falls_back_to_packaged_kernel():
    """DoseVoxelKernel should load the packaged default kernel if the exact voxel size is missing."""
    kernel = DoseVoxelKernel(isotope="Lu177", voxel_size_mm=5.5)
    assert kernel.kernel.shape == (51, 51, 51)
    assert kernel.kernel.dtype == np.float64
