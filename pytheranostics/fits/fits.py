from typing import Any, Callable, Optional, Tuple, Dict

import numpy
from pytheranostics.fits.functions import biexp_fun, monoexp_fun, triexp_fun, biexp_fun_uptake
from scipy.optimize import curve_fit
import lmfit
from lmfit import Model


def exponential_fit_lmfit(x_data: numpy.ndarray, y_data: numpy.ndarray, 
                          num_exponentials: int = 1, 
                          fixed_params: Optional[Dict[str, float]] = None,
                          bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                          params_init: Optional[Dict[str, float]] = None,
                          with_uptake: bool = False
                          ) -> Tuple[lmfit.model.ModelResult, Callable]:
    """
    Fit data to a sum of exponentials with flexible parameter fixing using lmfit.

    Parameters
    ----------
    x_data : array-like
        Independent variable data points.
    y_data : array-like
        Dependent variable data points.
    num_exponentials : int
        Number of exponential terms (1, 2, or 3).
    fixed_params : dict, optional
        Parameters to fix with their values.
        Keys are parameter names ('A1', 'A2', 'B1', etc.), and values are the fixed values.
    bounds : dict, optional
        Boundaries of parameter estimates.
        Keys are parameter names ('A1', 'A2', 'B1', etc.), and values are tuples representing the (min, max) values.
    params_init : dict, optional
        Initial values for parameter estimates.
        Keys are parameter names ('A1', 'A2', 'B1', etc.), and values are the initial parameter estimates.
    with_uptake : bool
        Apply constraints for an uptake phase.
        
    Returns
    -------
    result : lmfit.model.ModelResult
        Object containing the fit results.
    fitted_model : callable
        The fitted model function that can be used for predictions.
    """

    if num_exponentials not in [1, 2, 3]:
        raise ValueError("num_exponentials must be 1, 2, or 3.")

    if fixed_params is None:
        fixed_params = {}
        
    # Define the model function
    def model_func(x, **params):
        y = numpy.zeros_like(x)
        terms = ['A', 'B', 'C'][:num_exponentials]
        for term in terms:
            A1 = params.get(f'{term}1', 0)
            A2 = params.get(f'{term}2', 0)
            y += A1 * numpy.exp(-A2 * x)
        return y

    # Create a Model using lmfit
    model = Model(model_func, independent_vars=['x'])

    # Create parameters with initial guesses
    params = lmfit.Parameters()
    terms = ['A', 'B', 'C'][:num_exponentials]

    for term in terms:
        # Amplitude parameter
        amp_name = f'{term}1'
        # Check if parameter will have an expression
        will_have_expr = False
        if num_exponentials == 2 and with_uptake and amp_name == 'B1':
            will_have_expr = True
        if num_exponentials == 3 and with_uptake and amp_name == 'C1':
            will_have_expr = True

        if amp_name in fixed_params:
            params.add(amp_name, value=fixed_params[amp_name], vary=False)
        elif will_have_expr:
            # Add parameter that will have an expression
            params.add(amp_name, value=0, vary=False)
        else:
            min_b = -numpy.inf  # Allow negative amplitudes
            max_b = numpy.inf
            init_param = (max(y_data) - min(y_data)) / num_exponentials

            if bounds is not None and amp_name in bounds:
                min_b, max_b = bounds[amp_name]
            if params_init is not None and amp_name in params_init:
                init_param = params_init[amp_name]

            params.add(amp_name, value=init_param, min=min_b, max=max_b)

        # Exponent parameter
        exp_name = f'{term}2'
        if exp_name in fixed_params:
            params.add(exp_name, value=fixed_params[exp_name], vary=False)
        else:
            min_b = 0  # Exponents should be positive for decay
            max_b = numpy.inf
            init_param = 1.0

            if bounds is not None and exp_name in bounds:
                min_b, max_b = bounds[exp_name]
            if params_init is not None and exp_name in params_init:
                init_param = params_init[exp_name]

            params.add(exp_name, value=init_param, min=min_b, max=max_b)

    # Apply constraints if specified
    if num_exponentials == 2 and with_uptake:
        if 'B1' in params and 'A1' in params:
            print("Adding constraint for uptake: B1 = -A1")
            params['B1'].set(expr='-A1')
        else:
            raise ValueError("Parameters 'A1' and 'B1' must be present to apply the constraint 'B1 = -A1'.")

    if num_exponentials == 3 and with_uptake:
        if 'C1' in params and 'A1' in params and 'B1' in params:
            print("Adding constraint for uptake: C1 = -(A1 + B1)")
            params['C1'].set(expr='-(A1 + B1)')
        else:
            raise ValueError("Parameters 'A1', 'B1', and 'C1' must be present to apply the constraint 'C1 = -(A1 + B1)'.")

    # Perform the fit
    result = model.fit(y_data, params, x=x_data)

    # Define the fitted model function
    def fitted_model(x):
        return model_func(x, **result.params.valuesdict())

    return result, fitted_model

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
