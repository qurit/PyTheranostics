"""Tests for the radioactive decay module."""

import numpy as np
import pytest

from pytheranostics.shared.radioactive_decay import decay_act, get_activity_at_injection


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


def test_decay_act_negative_time_extrapolates_backward():
    """A negative interval back-corrects a later activity measurement."""
    measured_activity = 500.0
    half_life = 6.0
    delta_t = -2.0

    expected = measured_activity * np.exp(-np.log(2) * delta_t / half_life)

    assert np.isclose(
        decay_act(measured_activity, delta_t, half_life), expected, rtol=1e-10
    )


def test_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    with pytest.raises(ValueError):
        decay_act(-1000, 6.0, 12.0)  # Negative activity

    with pytest.raises(ValueError):
        decay_act(1000, 6.0, -12.0)  # Negative half-life


def test_get_activity_at_injection_with_post_measurement_after_injection():
    """Correct pre/post syringe readings to the common injection time."""
    half_life = 574300.0

    injection_datetime, injected_activity = get_activity_at_injection(
        injection_date="20220616",
        pre_inj_activity=7450.0,
        pre_inj_time="0804",
        post_inj_activity=14.4,
        post_inj_time="0955",
        injection_time="0918",
        half_life=half_life,
    )

    expected_pre = 7450.0 * np.exp(-np.log(2) * (74 * 60) / half_life)
    expected_post = 14.4 * np.exp(np.log(2) * (37 * 60) / half_life)

    assert injection_datetime.isoformat() == "2022-06-16T09:18:00"
    assert np.isclose(injected_activity, expected_pre - expected_post, rtol=1e-10)


@pytest.mark.parametrize(
    ("pre_inj_time", "post_inj_time", "error_message"),
    [
        ("0930", "0955", "pre_inj_time must be at or before injection_time"),
        ("0804", "0900", "post_inj_time must be at or after injection_time"),
    ],
)
def test_get_activity_at_injection_rejects_invalid_chronology(
    pre_inj_time, post_inj_time, error_message
):
    with pytest.raises(ValueError, match=error_message):
        get_activity_at_injection(
            injection_date="20220616",
            pre_inj_activity=7450.0,
            pre_inj_time=pre_inj_time,
            post_inj_activity=14.4,
            post_inj_time=post_inj_time,
            injection_time="0918",
            half_life=574300.0,
        )
