import matplotlib.pyplot as plt
import numpy as np
from doodle.fits.functions import monoexp_fun, biexp_fun, triexp_fun, biexp_fun_uptake
from typing import Tuple

def ewin_montage(img,ewin):
    '''
    img: image data

    ewin: energy window dictionary

    '''

    plt.figure(figsize=(22,6))
    for ind,i in enumerate(range(0,int(img.shape[0]),2)):
        keys = list(ewin.keys())

        # Top row Detector 1
        plt.subplot(2,6,ind+1)
        plt.imshow(img[i,:,:])
        plt.title(f'Detector1 {ewin[keys[ind]]["center"]} keV')
        plt.colorbar()
        
        
    #     # Bottom row Detector 2
        plt.subplot(2,6,ind+7)
        plt.imshow(img[i+1,:,:])
        plt.title(f'Detector2 {ewin[keys[ind]]["center"]} keV')
        plt.colorbar()


    plt.tight_layout()



def plot_tac(
        time: np.ndarray, 
        activity: np.ndarray, 
        exp_order: int,
        parameters: np.ndarray, 
        residuals: np.ndarray, 
        organ: str, 
        xlabel: str = 't (hours)', 
        ylabel:str = 'A (MBq)',
        skip_points=0,
        sigmas=None,
        **kwargs) -> None:
    
    """Generic Time Activity Curve plotting function. Supports:
        - exp_order = 1 -> Mono-exponential
        - exp_order = 2 -> Bi-exponential
        - exp_order = -2 -> Bi-exponential with uptake constrain.
        - exp_order = 3 -> Tri-exponential"""
        
    if abs(exp_order) < 1 or abs(exp_order) > 3:
        raise ValueError("Not supported. Please use order 1, 2 or 3.")
      
    if exp_order == 1:
        fit_function = monoexp_fun
        label=f'$A(t) = {round(parameters[0], 1)} e^{{-{round(parameters[1], 2)} t}}$'
    elif exp_order == 2:
        fit_function = biexp_fun
        label=f'$A(t) = {round(parameters[0], 1)} e^{{-{round(parameters[1], 2)} t}} + {round(parameters[2], 1)} e^{{-{round(parameters[3], 2)} t}}$'
    elif exp_order == -2:
        fit_function = biexp_fun_uptake
        label=f'$A(t) = {round(parameters[0], 1)} e^{{-{round(parameters[1], 2)} t}} - {round(parameters[0], 1)} e^{{-{round(parameters[2], 2)} t}}$'
    elif exp_order == 3:
        fit_function = triexp_fun
        label=f"$A(t) = {round(parameters[0], 1)} e^{{-{round(parameters[1], 2)} t}} + {round(parameters[2], 2)} e^{{-{round(parameters[3], 2)} t}} - {round(parameters[0] + parameters[2], 1)} e^{{-{round(parameters[4], 2)} t}}$"

    tt = np.linspace(0, time.max() * 2, 1000)
    yy = fit_function(tt, *parameters[:-1])
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    if sigmas is not None and any(sigma is not None for sigma in sigmas):
        axes[0].errorbar(time, activity, yerr=sigmas, fmt='o', color='#1f77b4', markeredgecolor='black')
    else:
        axes[0].plot(time, activity, 'o', color='#1f77b4', markeredgecolor='black')

    if skip_points:
        axes[0].plot(tt[tt >= time[skip_points]], yy[tt >= time[skip_points]])
        axes[1].semilogy(tt[tt >= time[skip_points]], yy[tt >= time[skip_points]])
    else:
        axes[0].plot(tt, yy)
        axes[1].semilogy(tt, yy)

    axes[0].set_title(f'{organ}')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].text(0.6, 0.85, f'$R^2={parameters[-1]:.4f}$', transform=axes[0].transAxes)
    axes[0].set_xlim(-10, axes[0].get_xlim()[1])  
    axes[0].set_ylim(0.00001, axes[0].get_ylim()[1]) 
    
    axes[1].semilogy(time[skip_points:], activity[skip_points:], 'o', color='#1f77b4', markeredgecolor='black')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title(f'{label}')

    axes[2].plot(time[skip_points:], residuals, 'o', color='#1f77b4', markeredgecolor='black')
    axes[2].set_title('Residuals')
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    

    plt.tight_layout()

    if 'save_path' in kwargs:
        plt.savefig(f"{kwargs['save_path']}/{organ}_{fit_function.__name__}_fit.png", dpi=300)

    return None



def plot_tac_fixed_biokinetics(
        time: np.ndarray, 
        activity: np.ndarray, 
        exp_order: int,
        parameters: np.ndarray,
        fixed_biokinetics: np.ndarray, 
        residuals: np.ndarray, 
        organ: str, 
        xlabel: str = 't (hours)', 
        ylabel:str = 'A (MBq)',
        skip_points=0,
        sigmas=None,
        **kwargs) -> None:
    
    """Generic Time Activity Curve plotting function. Supports:
        - exp_order = 1 -> Mono-exponential
        - exp_order = 2 -> Bi-exponential
        - exp_order = -2 -> Bi-exponential with uptake constrain.
        - exp_order = 3 -> Tri-exponential"""
        
    if abs(exp_order) < 1 or abs(exp_order) > 3:
        raise ValueError("Not supported. Please use order 1, 2 or 3.")
      
    if exp_order == 1:
        fit_function = (lambda x, a: monoexp_fun(x, a, fixed_biokinetics[0]))
        label=f'$A(t) = {round(parameters[0], 1)} e^{{-{round(fixed_biokinetics[0], 2)} t}}$'
    elif exp_order == 2:
        fit_function = (lambda x, a, c: biexp_fun(x, a, fixed_biokinetics[0], c, fixed_biokinetics[1]))
        label=f'$A(t) = {round(parameters[0], 1)} e^{{-{round(fixed_biokinetics[0], 2)} t}} + {round(parameters[1], 1)} e^{{-{round(fixed_biokinetics[1], 2)} t}}$'
    elif exp_order == -2:
        fit_function = (lambda x, a: biexp_fun_uptake(x, a, fixed_biokinetics[0], fixed_biokinetics[1]))
        label=f'$A(t) = {round(parameters[0], 1)} e^{{-{round(fixed_biokinetics[0], 2)} t}} - {round(parameters[0], 1)} e^{{-{round(fixed_biokinetics[1], 2)} t}}$'
    elif exp_order == 3:
        fit_function = (lambda x, a, c: triexp_fun(x, a, fixed_biokinetics[0], c, fixed_biokinetics[1], fixed_biokinetics[2]))
        label=f"$A(t) = {round(parameters[0], 1)} e^{{-{round(fixed_biokinetics[0], 2)} t}} + {round(parameters[1], 2)} e^{{-{round(fixed_biokinetics[1], 2)} t}} - {round(parameters[0] + parameters[1], 1)} e^{{-{round(fixed_biokinetics[2], 2)} t}}$"

    tt = np.linspace(0, time.max() * 2, 1000)
    yy = fit_function(tt, *parameters[:-1])
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    if sigmas is not None and any(sigma is not None for sigma in sigmas):
        axes[0].errorbar(time, activity, yerr=sigmas, fmt='o', color='#1f77b4', markeredgecolor='black')
    else:
        axes[0].plot(time, activity, 'o', color='#1f77b4', markeredgecolor='black')

    if skip_points:
        axes[0].plot(tt[tt >= time[skip_points]], yy[tt >= time[skip_points]])
        axes[1].semilogy(tt[tt >= time[skip_points]], yy[tt >= time[skip_points]])
    else:
        axes[0].plot(tt, yy)
        axes[1].semilogy(tt, yy)

    axes[0].set_title(f'{organ}')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].text(0.6, 0.85, f'$R^2={parameters[-1]:.4f}$', transform=axes[0].transAxes)
    axes[0].set_xlim(-10, axes[0].get_xlim()[1])  
    axes[0].set_ylim(0.00001, axes[0].get_ylim()[1]) 
    
    axes[1].semilogy(time[skip_points:], activity[skip_points:], 'o', color='#1f77b4', markeredgecolor='black')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title(f'{label}')

    axes[2].plot(time[skip_points:], residuals, 'o', color='#1f77b4', markeredgecolor='black')
    axes[2].set_title('Residuals')
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)
    

    plt.tight_layout()

    if 'save_path' in kwargs:
        plt.savefig(f"{kwargs['save_path']}/{organ}_{fit_function.__name__}_fit.png", dpi=300)

    return None
