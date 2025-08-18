"""Smoke Tests for basic setup"""

import pytest


@pytest.mark.smoke
def test_basic_import() -> None:
    import pytheranostics

    assert pytheranostics is not None  # Use the import
