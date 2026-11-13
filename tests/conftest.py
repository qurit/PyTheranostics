"""Test configuration and fixtures for PyTheranostics."""

import numpy as np
import pytest


def pytest_collection_modifyitems(config, items):
    """Modify test collection to run smoke tests first."""
    # Separate smoke tests from other tests
    smoke_tests = []
    other_tests = []

    for item in items:
        if "smoke" in item.keywords:
            smoke_tests.append(item)
        else:
            other_tests.append(item)

    # Reorder: smoke tests first, then others
    items[:] = smoke_tests + other_tests


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.rand(100, 100)


@pytest.fixture
def sample_activity():
    """Create sample activity data."""
    return np.array([1000, 800, 600, 400, 200])


@pytest.fixture
def sample_time_points():
    """Create sample time points."""
    return np.array([0, 1, 2, 3, 4])
