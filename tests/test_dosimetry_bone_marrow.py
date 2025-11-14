"""Unit tests for bone marrow dosimetry helpers."""

import math

import pytest

from pytheranostics.dosimetry.bone_marrow import bm_scaling_factor


def test_bm_scaling_factor_uses_phantom_mass_by_default():
    """If no patient mass is provided, phantom data should be used."""
    result = bm_scaling_factor(gender="Female", mass_bm=None, hematocrit=None)
    assert math.isclose(result, 900.0, rel_tol=1e-6)


@pytest.mark.parametrize(
    ("gender", "mass_bm", "hematocrit", "expected"),
    [
        ("Male", 1000.0, None, 1000.0),
        ("Female", 900.0, 0.45, 0.19 / (1 - 0.45) * 900.0),
    ],
)
def test_bm_scaling_factor_handles_custom_values(gender, mass_bm, hematocrit, expected):
    """The scaling factor should respect custom masses and hematocrit."""
    result = bm_scaling_factor(gender=gender, mass_bm=mass_bm, hematocrit=hematocrit)
    assert math.isclose(result, expected, rel_tol=1e-6)
