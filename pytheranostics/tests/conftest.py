"""Test configuration and fixtures for PyTheranostics."""

import pytest
import numpy as np
import os

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.rand(100, 100)

@pytest.fixture
def sample_dicom_path():
    """Return path to sample DICOM file."""
    return os.path.join(os.path.dirname(__file__), 'data', 'sample.dcm')

@pytest.fixture
def sample_activity():
    """Create sample activity data."""
    return np.array([1000, 800, 600, 400, 200])

@pytest.fixture
def sample_time_points():
    """Create sample time points."""
    return np.array([0, 1, 2, 3, 4]) 