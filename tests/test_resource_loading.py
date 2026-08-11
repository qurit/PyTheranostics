"""Tests for data access via importlib.resources-based helpers."""

from pytheranostics.dosimetry.olinda import load_phantom_mass


def test_load_phantom_mass_returns_expected_value():
    """The packaged phantom mass table should include standard organs."""
    liver_mass = load_phantom_mass(gender="Male", organ="Liver")
    assert liver_mass == 1800  # matches bundled phantom data
