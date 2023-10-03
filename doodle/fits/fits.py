from typing import Optional, Tuple
import numpy as np
from numpy import exp
from scipy.optimize import curve_fit


def monoexp_fun(t,a,b):
    return a*exp(-b*t)

def biexp_fun(t, a, b,c,d):
    return a*exp(-b*t) + c*exp(-d*t)

def find_a_initial(f, b, t):
    return f * exp(b * t)

def get_residuals(
    time: np.ndarray,
    activity: np.ndarray,
    popt: np.ndarray,
    eq: str = 'monoexp'
    ) -> Tuple[float, np.ndarray]:

    if eq == 'monoexp':
        residuals = activity - monoexp_fun(time, popt[0], popt[1])
    elif eq == 'biexp':
        residuals = activity - biexp_fun(time, popt[0], popt[1], popt[2], popt[3])

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((activity - np.mean(activity))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared, residuals

def fit_monoexp(time: np.ndarray, 
                activity: np.ndarray, 
                decayconst: float = 1000000,
                skip_points: int = 0,
                weights: Optional[np.ndarray] = None,
                monoguess: Tuple[int, int] = (1,1),
                maxev: int = 100000
                ) -> Tuple[np.ndarray, np.ndarray]:

    if time.shape != activity.shape:
        raise AssertionError("Time and Activity arrays have different shapes.")

    # The minimum decay is with physical decay, can't decay slower. (decayconst to inf)
    if skip_points > 0:
        time = time[skip_points:]
        activity = activity[skip_points:]
        weights = weights[skip_points:] if weights is not None else weights

        if weights and weights.shape != time.shape:
            raise AssertionError("Time and Weights arrays have different shapes.")

    if time.shape[0] < 2:
        raise AssertionError(f"Only {time.shape[0]} data points available. Not enough points to perform fit.")

    popt, _ = curve_fit(monoexp_fun, time, activity, sigma=weights,
                        p0=monoguess,
                        bounds=([0,decayconst], np.inf), maxfev=maxev)

    r_squared, residuals = get_residuals(time=time, activity=activity, popt=popt, eq='monoexp')

    popt = np.append(popt, r_squared)

    return popt, residuals


def fit_biexp(t,a,decayconst,skip_points=0,ignore_weights=True,append_zero=True,biguess=(1,1,1,0.1),maxev=100000):

  
    if ignore_weights:
        weights = None
    else:
        weights = sigmas
        weights = np.append(1,weights)

      
    if weights: 
        popt, pconv = curve_fit(biexp_fun,t[skip_points:],a[skip_points:],sigma=weights[skip_points:],
                                    p0=biguess,bounds=([0,decayconst,0,decayconst],np.inf),maxfev=maxev)
    else:
        popt, pconv = curve_fit(biexp_fun,t[skip_points:],a[skip_points:],
                                    p0=biguess,bounds=([0,decayconst,0,decayconst],np.inf),maxfev=maxev)
    

    [r_squared, residuals] = get_residuals(t,a,skip_points,popt,eq='biexp')

    popt = np.append(popt,r_squared)

    # create times for displaying the function
    tt = np.linspace(0,t.max()*2.5,1000)
    yy = biexp_fun(tt,*popt[:4])

    return popt,tt,yy,residuals



def fit_biexp_uptake(t,a,decayconst,skip_points=0,ignore_weights=True,append_zero=True,uptakeguess=(1,1,-1,1),maxev=100000):

     # Append point (0,0) if flag set to true.
    if append_zero:
        t = np.append(0,t)
        a = np.append(0,a)
        if ignore_weights:
            weights = None
        else:
            weights = sigmas
            weights = np.append(1,weights)

    #monoexponential fit
    # bounds show that the minimum decay is with physical decay, can't decay slower. (decayconst to inf)
     
 
    popt, pconv = curve_fit(biexp_fun,t,a,sigma=weights,
                            p0=uptakeguess,bounds=([0,decayconst,-np.inf,decayconst],[np.inf,np.inf,0,np.inf]),maxfev=maxev)
    

    [r_squared, residuals] = get_residuals(t,a,skip_points,popt,eq='biexp')

    popt = np.append(popt,r_squared)

    # create times for displaying the function
    tt = np.linspace(0,t.max()*2.5,1000)
    yy = biexp_fun(tt,*popt[:4])

    return popt,tt,yy,residuals

