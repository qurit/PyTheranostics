import numpy as np
from numpy import exp,log
import pandas as pd

import matplotlib.pylab as plt

from scipy import integrate
from scipy.optimize import curve_fit


def monoexp_fun(x, a, b):
    return a * exp(-b * x)

def biexp_fun(x, a, b, c, d):
    return a * exp(-b * x) + c * exp(-d * x)

def triexp_fun(x, a, b, c, d, e, f):
    return a * exp(-b * x) + c * exp(-d * x) + e * exp(-f * x)

def find_a_initial(f, b, t):
    return f * exp(b * t)

def get_residuals(t,a,skip_points,popt,eq='monoexp'):

    if eq == 'monoexp':
        residuals = a[skip_points:] - monoexp_fun(t[skip_points:],popt[0],popt[1])
    elif eq == 'biexp':
        residuals = a[skip_points:] - biexp_fun(t[skip_points:],popt[0],popt[1],popt[2],popt[3])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((a[skip_points:]-np.mean(a[skip_points:]))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared, residuals

def Hanscheid(a, t):
    return a * ((2 * t) / (log(2)))



def fit_monoexp(t,a,decayconst,skip_points=0,ignore_weights=True,monoguess=(1,1),maxev=100000,limit_bounds=False, sigmas = None):

    if ignore_weights:
        weights = None
    else:
        weights = sigmas

    #monoexponential fit
    # bounds show that the minimum decay is with physical decay, can't decay slower. (decayconst to inf)
    
    if limit_bounds:
        upper_bound = decayconst
    else:
        upper_bound = decayconst #* 1e6 ## to check! it works with preclinical dosimetry; is it the same in humn cases?
    
     
    if weights:    
        popt, pconv = curve_fit(monoexp_fun,t[skip_points:],a[skip_points:],sigma=weights[skip_points:],
                                p0=monoguess,bounds=([0,upper_bound],np.inf),maxfev=maxev)
    else:
        popt, pconv = curve_fit(monoexp_fun,t[skip_points:],a[skip_points:],sigma=weights,
                                p0=monoguess,bounds=([0,upper_bound],np.inf), maxfev=maxev)


    [r_squared, residuals] = get_residuals(t,a,skip_points,popt,eq='monoexp')

    popt = np.append(popt,r_squared)

    # create times for displaying the function
    tt = np.linspace(0,t.max()*2.5,1000)
    yy = monoexp_fun(tt,*popt[:2])
    # yy = monoexp_fun(tt[tt>=t[skip_points]],*popt[:2])

    # popt_df = pd.DataFrame(popt).T
    # popt_df.columns = ['A0','lambda_eff','R2']

    return popt,tt,yy,residuals


def fit_biexp(t,a,decayconst,skip_points=0,ignore_weights=True,append_zero=True,biguess=(1,1,1,0.1),maxev=100000, sigmas = None):

  
    if ignore_weights:
        weights = None
    else:
        weights = sigmas
        #weights = np.append(1,weights)

    if any(weight is not None for weight in weights): 
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
    # yy = monoexp_fun(tt[tt>=t[skip_points]],*popt[:2])

    # popt_df = pd.DataFrame(popt).T
    # popt_df.columns = ['A0','lambda_eff','R2']

    return popt,tt,yy,residuals



def fit_biexp_uptake(t,a,decayconst,skip_points=0,ignore_weights=True,append_zero=True,uptakeguess=(1,1,-1,1),maxev=100000, sigmas = None):

     # Append point (0,0) if flag set to true.
    if append_zero:
        t = np.append(0,t)
        a = np.append(0,a)
        if ignore_weights:
            weights = None
        else:
            weights = sigmas
            weights = np.append(1,weights)


    if any(weight is not None for weight in weights): 
        popt, pconv = curve_fit(biexp_fun,t[skip_points:],a[skip_points:],sigma=weights[skip_points:],
                                    p0=uptakeguess,bounds=([0,decayconst,-np.inf,decayconst],[np.inf,np.inf,0,np.inf]),maxfev=maxev)
    else:
        popt, pconv = curve_fit(biexp_fun,t[skip_points:],a[skip_points:],
                                    p0=biguess,bounds=([0,decayconst,0,decayconst],np.inf),maxfev=maxev)
        


    [r_squared, residuals] = get_residuals(t,a,skip_points,popt,eq='biexp')

    popt = np.append(popt,r_squared)

    # create times for displaying the function
    tt = np.linspace(0,t.max()*2.5,1000)
    yy = biexp_fun(tt,*popt[:4])
    # yy = monoexp_fun(tt[tt>=t[skip_points]],*popt[:2])

    # popt_df = pd.DataFrame(popt).T
    # popt_df.columns = ['A0','lambda_eff','R2']

    return popt,tt,yy,residuals

def fit_triexp(t,a,decayconst,skip_points=0,ignore_weights=True,append_zero=True,uptakeguess=(1,1,1,1,1,1),maxev=100000):

     # Append point (0,0) if flag set to true.
    if append_zero:
        t = np.append(0,t)
        a = np.append(0,a)
        if ignore_weights:
            weights = None
        else:
            weights = sigmas
            weights = np.append(1,weights)


    popt, pconv = curve_fit(triexp_fun,t,a,sigma=weights,
                            p0=uptakeguess,bounds=None,maxfev=maxev)
    

    [r_squared, residuals] = get_residuals(t,a,skip_points,popt,eq='triexp')

    popt = np.append(popt,r_squared)

    # create times for displaying the function
    tt = np.linspace(0,t.max()*2.5,1000)
    yy = biexp_fun(tt,*popt[:6])
    # yy = monoexp_fun(tt[tt>=t[skip_points]],*popt[:2])

    # popt_df = pd.DataFrame(popt).T
    # popt_df.columns = ['A0','lambda_eff','R2']

    return popt,tt,yy,residuals
