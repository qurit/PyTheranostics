"""Tests for the fitting helper functions."""

import numpy as np
import pytest

from pytheranostics.fits.fits import (
    calculate_r_squared,
    exponential_fit_lmfit,
    get_exponential,
)
from pytheranostics.fits.functions import monoexp_fun


def test_get_exponential_defaults():
    """Default configuration for mono-exponential fits should be stable."""
    func, params, bounds = get_exponential(order=1, param_init=None, decayconst=0.1)
    assert func is monoexp_fun
    assert params == (1, 1)
    assert bounds[0][0] == 0
    assert pytest.approx(bounds[0][1]) == 0.1
    assert np.isinf(bounds[1])


def test_calculate_r_squared_perfect_fit():
    """A perfect mono-exponential fit should have r^2 == 1."""
    x = np.linspace(0, 4, 5)
    y = monoexp_fun(x, 2.0, 0.5)
    r2, residuals = calculate_r_squared(x, y, (2.0, 0.5), monoexp_fun)
    assert pytest.approx(r2, rel=1e-9) == 1.0
    assert np.allclose(residuals, 0.0)


def test_exponential_fit_lmfit_mono_handles_noise():
    """Mono-exponential fit should recover parameters from noisy data."""
    rng = np.random.default_rng(42)
    x = np.linspace(0, 6, 20)
    y_true = monoexp_fun(x, 5.0, 0.4)
    y_noisy = y_true + rng.normal(scale=0.05, size=x.shape)

    result, fitted_model = exponential_fit_lmfit(
        x_data=x, y_data=y_noisy, num_exponentials=1
    )

    assert pytest.approx(result.params["A1"].value, rel=0.05) == 5.0
    assert pytest.approx(result.params["A2"].value, rel=0.1) == 0.4
    assert np.allclose(
        fitted_model(x[:3]),
        monoexp_fun(x[:3], result.params["A1"].value, result.params["A2"].value),
    )


def test_exponential_fit_lmfit_applies_uptake_constraint():
    """Bi-exponential fits with uptake should constrain the amplitudes."""
    x = np.linspace(0, 4, 15)
    y = monoexp_fun(x, 2.0, 0.5) + monoexp_fun(x, -2.0, 1.5)

    result, _ = exponential_fit_lmfit(x, y, num_exponentials=2, with_uptake=True)

    assert result.params["B1"].expr == "-A1"
    assert pytest.approx(result.params["A1"].value, rel=0.1) == 2.0
