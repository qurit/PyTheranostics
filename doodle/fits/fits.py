from typing import Any, Callable, Optional, Tuple

import numpy
from doodle.fits.functions import biexp_fun, monoexp_fun, triexp_fun, biexp_fun_uptake
from scipy.optimize import curve_fit


def calculate_r_squared(
        time: numpy.ndarray,
        activity: numpy.ndarray,
        popt: numpy.ndarray,
        func: Callable
    ) -> Tuple[float, numpy.ndarray]:
    """ Calculate r_squared and residuals between fit and data-points.
    """

    residuals = activity - func(time, *popt)

    ss_res = numpy.sum(residuals**2)
    ss_tot = numpy.sum((activity - numpy.mean(activity))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared, residuals

def get_exponential(
        order: int, 
        param_init: Optional[Tuple[float, ...]], 
        decayconst: float) -> Tuple[Callable, Tuple[float, ...], Optional[Tuple[Any, ...]]]:
    """Retrieve an exponential function given an input order 'order', initial parameters and a decay-constant
    value (for defatult constrains)"""

    # Default initial parameters:
    default_initial = {1: (1, 1),
                       2: (1, 1, 1, 0.1),
                       -2: (1, 1, 1),  
                       3: (1, 1, 1, 1, 1, 1)
                       }
    
    # Bounds: It can't decay slower than physical decay!
    bounds = {1: ([0, decayconst], numpy.inf), 
              2: ([0, decayconst, 0, decayconst], numpy.inf),
              -2: ([0, decayconst, decayconst], [numpy.inf, numpy.inf, numpy.inf]),
              3: (-numpy.inf, numpy.inf)}

    if order == 1:
        func = monoexp_fun
    elif order == 2:
        func = biexp_fun
    elif order == -2:
        func = biexp_fun_uptake
    elif order == 3:
        func = triexp_fun
    else:
        NotImplementedError("Function not implemented.")

    return func, default_initial[order] if param_init is None else param_init, bounds[order]

def fit_tac(
        time: numpy.ndarray, 
        activity: numpy.ndarray, 
        decayconst: float = 1000000,
        exp_order: int = 1,
        weights: Optional[numpy.ndarray] = None,
        param_init: Optional[Tuple[float, ...]] = None,
        through_origin: bool = False,
        maxev: int = 100000
                ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Generic Time Activity Curve fitting function. Supports:
        - exp_order = 1 -> Mono-exponential
        - exp_order = 2 -> Bi-exponential
        - exp_order = -2 -> Bi-exponential with uptake constrain.
        - exp_order = 3 -> Tri-exponential"""

    if abs(exp_order) < 1 or abs(exp_order) > 3:
        raise ValueError("Not supported. Please use order 1, 2 or 3.")
    
    # Get function
    exp_function, initial_params, bounds = get_exponential(order=exp_order, 
                                                           param_init=param_init, 
                                                           decayconst=decayconst)

    # Checks
    if time.shape != activity.shape:
        raise AssertionError("Time and Activity arrays have different shapes.")

    if weights is not None and weights and weights.shape != time.shape:
        raise AssertionError("Time and Weights arrays have different shapes.")

    if time.shape[0] < 2 * abs(exp_order):
        raise AssertionError(f"Only {time.shape[0]} data points available. Not enough points to perform fit.")

        
    popt, _ = curve_fit(exp_function, time, activity, sigma=weights,
                        p0=initial_params,
                        bounds=bounds, maxfev=maxev)

    r_squared, residuals = calculate_r_squared(time=time, activity=activity, popt=popt, func=exp_function)

    popt = numpy.append(popt, r_squared)

    return popt, residuals


def fit_tac_with_fixed_biokinetics(
        time: numpy.ndarray, 
        activity: numpy.ndarray, 
        decayconst: float = 1000000,
        exp_order: int = 1,
        weights: Optional[numpy.ndarray] = None,
        param_init: Optional[Tuple[float, ...]] = None,
        through_origin: bool = False,
        fixed_biokinetics: Optional[numpy.ndarray] = None,
        maxev: int = 100000
                ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Generic Time Activity Curve fitting function. Supports:
        - exp_order = 1 -> Mono-exponential
        - exp_order = 2 -> Bi-exponential
        - exp_order = -2 -> Bi-exponential with uptake constrain.
        - exp_order = 3 -> Tri-exponential"""

    if abs(exp_order) < 1 or abs(exp_order) > 3:
        raise ValueError("Not supported. Please use order 1, 2 or 3.")
    
    # Get function
    exp_function, initial_params, bounds = get_exponential(order=exp_order, 
                                                           param_init=param_init, 
                                                           decayconst=decayconst)

    # Checks
    if time.shape != activity.shape:
        raise AssertionError("Time and Activity arrays have different shapes.")

    if weights is not None and weights and weights.shape != time.shape:
        raise AssertionError("Time and Weights arrays have different shapes.")


    if exp_function == monoexp_fun:
        popt, _ = curve_fit(lambda x, a: exp_function(x, a, fixed_biokinetics[0]), 
                            time, activity, sigma=weights,
                            p0=param_init,
                            bounds=(0, numpy.inf), maxfev=maxev)
        r_squared, residuals = calculate_r_squared(time=time, activity=activity, popt=popt, func=(lambda x, a: exp_function(x, a, fixed_biokinetics[0])))
    elif exp_function == biexp_fun:
        popt, _ = curve_fit(lambda x, a, c: exp_function(x, a, fixed_biokinetics[0], c, fixed_biokinetics[1]), 
                            time, activity, sigma=weights,
                            p0=param_init,
                            bounds=(0, numpy.inf), maxfev=maxev)
        r_squared, residuals = calculate_r_squared(time=time, activity=activity, popt=popt, func=(lambda x, a, c: exp_function(x, a, fixed_biokinetics[0], c, fixed_biokinetics[1])))
    elif exp_function == biexp_fun_uptake:
        popt, _ = curve_fit(lambda x, a: exp_function(x, a, fixed_biokinetics[0], fixed_biokinetics[1]), 
                            time, activity, sigma=weights,
                            p0=param_init,
                            bounds=(0, numpy.inf), maxfev=maxev)
        r_squared, residuals = calculate_r_squared(time=time, activity=activity, popt=popt, func=(lambda x, a: exp_function(x, a, fixed_biokinetics[0], fixed_biokinetics[1])))
    elif exp_function == triexp_fun:
        popt, _ = curve_fit(lambda x, a, c: exp_function(x, a, fixed_biokinetics[0], c, fixed_biokinetics[1], fixed_biokinetics[2]), 
                            time, activity, sigma=weights,
                            p0=param_init, maxfev=maxev)
        r_squared, residuals = calculate_r_squared(time=time, activity=activity, popt=popt, func=(lambda x, a, c: exp_function(x, a, fixed_biokinetics[0], c, fixed_biokinetics[1], fixed_biokinetics[2])))

    
    popt = numpy.append(popt, r_squared)

    return popt, residuals