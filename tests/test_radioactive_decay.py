"""Tests for the radioactive decay module."""

import numpy as np
import pytest

from pytheranostics.shared.radioactive_decay import decay_act


def test_decay_act():
    """Test the decay_act function."""
    # Test with known values
    initial_activity = 1000  # MBq
    half_life = 6.0  # hours
    time = 12.0  # hours

    expected = initial_activity * np.exp(-np.log(2) * time / half_life)
    result = decay_act(initial_activity, time, half_life)

    assert np.isclose(result, expected, rtol=1e-10)


def test_decay_act_array():
    """Test decay_act with array inputs."""
    initial_activity = np.array([1000, 2000])
    half_life = 6.0
    time = np.array([6.0, 12.0])

    expected = initial_activity * np.exp(-np.log(2) * time / half_life)
    result = decay_act(initial_activity, time, half_life)

    assert np.allclose(result, expected, rtol=1e-10)


def test_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    with pytest.raises(ValueError):
        decay_act(-1000, 6.0, 12.0)  # Negative activity

    with pytest.raises(ValueError):
        decay_act(1000, -6.0, 12.0)  # Negative half-life

    with pytest.raises(ValueError):
        decay_act(1000, 6.0, -12.0)  # Negative time


# def test_get_activity_at_injection():
#     """Test the get_activity_at_injection function."""
#     current_activity = 500  # MBq
#     half_life = 6.0  # hours
#     time_since_injection = 12.0  # hours

#     expected = current_activity * np.exp(np.log(2) * time_since_injection / half_life)
#     result = decay_act(current_activity, time_since_injection, half_life)
#     result = get_activity_at_injection(
#         current_activity, half_life, time_since_injection
#     )

#     assert np.isclose(result, expected, rtol=1e-10)
