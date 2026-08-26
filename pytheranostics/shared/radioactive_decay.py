"""Radioactive decay helpers shared across modules."""

from datetime import datetime

import numpy as np


def decay_act(a_initial, delta_t, half_life):
    """Return activity corrected by ``delta_t`` given the half-life.

    Positive time intervals decay activity forward in time. Negative time
    intervals extrapolate a later measurement backward to an earlier time.
    """
    if np.any(np.asarray(a_initial) < 0):
        raise ValueError("a_initial must be positive")
    if np.any(np.asarray(half_life) < 0):
        raise ValueError("half_life must be positive")

    return a_initial * np.exp(-np.log(2) / half_life * delta_t)


def get_activity_at_injection(
    injection_date,
    pre_inj_activity,
    pre_inj_time,
    post_inj_activity,
    post_inj_time,
    injection_time,
    half_life,
):
    """Compute injection datetime and activity from pre/post syringe readings."""
    # Pass half-life in seconds

    # Set the times and the time deltas to injection time
    pre_datetime = datetime.strptime(
        injection_date + pre_inj_time + "00.00", "%Y%m%d%H%M%S.%f"
    )
    post_datetime = datetime.strptime(
        injection_date + post_inj_time + "00.00", "%Y%m%d%H%M%S.%f"
    )
    inj_datetime = datetime.strptime(
        injection_date + injection_time + "00.00", "%Y%m%d%H%M%S.%f"
    )

    if pre_datetime > inj_datetime:
        raise ValueError(
            "pre_inj_time must be at or before injection_time; "
            f"got pre_inj_time={pre_inj_time!r} and "
            f"injection_time={injection_time!r}."
        )
    if post_datetime < inj_datetime:
        raise ValueError(
            "post_inj_time must be at or after injection_time; "
            f"got post_inj_time={post_inj_time!r} and "
            f"injection_time={injection_time!r}."
        )

    delta_inj_pre = (inj_datetime - pre_datetime).total_seconds()
    delta_post_inj = (inj_datetime - post_datetime).total_seconds()

    pre_activity = decay_act(pre_inj_activity, delta_inj_pre, half_life)
    post_activity = decay_act(post_inj_activity, delta_post_inj, half_life)

    injected_activity = pre_activity - post_activity

    return inj_datetime, injected_activity
