import matplotlib.pyplot as plt
import numpy as np

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





def monoexp_fit_plots(t,a,tt,yy,organ, r_squared, residuals, xlabel = 't (days)', ylabel = 'A (MBq)',skip_points=0,sigmas=None,**kwargs):
    fig, axes = plt.subplots(1, 3, figsize=(10,4))
    if sigmas is not None and any(sigma is not None for sigma in sigmas):
        axes[0].errorbar(t,a,yerr=sigmas,fmt='o', color='#1f77b4',markeredgecolor='black')
        # axes[1,0].errorbar(t,a,yerr=sigmas,fmt='o')
    else:
        axes[0].plot(t,a,'o',color='#1f77b4',markeredgecolor='black')

    if skip_points:
        axes[0].plot(tt[tt>=t[skip_points]],yy)
        axes[1].semilogy(tt[tt>=t[skip_points]],yy)
    else:
        axes[0].plot(tt,yy)
        axes[1].semilogy(tt,yy)

    axes[0].set_title('{} Mono-Exponential'.format(organ))
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].text(0.6,0.85,'$R^2={:.4f}$'.format(r_squared),transform=axes[0].transAxes)

    axes[1].semilogy(t[skip_points:],a[skip_points:],'o',color='#1f77b4',markeredgecolor='black')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title('{} Mono-Exponential'.format(organ))

    axes[2].plot(t[skip_points:],residuals,'o', color='#1f77b4',markeredgecolor='black')
    axes[2].set_title('Residuals')
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)

    plt.tight_layout()

    if 'save_path' in kwargs:
        plt.savefig(f"{kwargs['save_path']}/{organ}_monoexp_fit.png",dpi=300)



def biexp_fit_plots(t,a,tt,yy,organ, r_squared,residuals, xlabel = 't (days)', ylabel = 'A (MBq)', skip_points=0,sigmas=None,**kwargs):
    fig, axes = plt.subplots(1, 3, figsize=(10,4))

    if sigmas is not None and any(sigma is not None for sigma in sigmas):
        axes[0].errorbar(t,a,yerr=sigmas,fmt='o',color='#1f77b4',markeredgecolor='black')
        # axes[1,0].errorbar(t,a,yerr=sigmas,fmt='o')
    else:
        axes[0].plot(t,a,'o',color='#1f77b4',markeredgecolor='black')

    if skip_points:
        axes[0].plot(tt[tt>=t[skip_points]],yy)
        axes[1].semilogy(tt[tt>=t[skip_points]],yy)
    else:
        axes[0].plot(tt,yy)
        axes[1].semilogy(tt,yy)

    axes[0].set_title('{} Bi-Exponential'.format(organ))
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].text(0.6,0.85,'$R^2={:.4f}$'.format(r_squared),transform=axes[0].transAxes)
    


    axes[1].semilogy(t[skip_points:],a[skip_points:],'o',color='#1f77b4',markeredgecolor='black')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title('{} Bi-Exponential'.format(organ))

    try:
        axes[2].plot(t[skip_points:],residuals,'o', color='#1f77b4',markeredgecolor='black')
    except:
        t = np.append(0,t)
        axes[2].plot(t[skip_points:],residuals,'o', color='#1f77b4',markeredgecolor='black')
    axes[2].set_title('Residuals')
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(ylabel)


    axes[0].set_xlim(left=0)
    axes[0].set_ylim(bottom=0)

    axes[1].set_xlim(left=0)



    plt.tight_layout()

    if 'save_path' in kwargs:
        plt.savefig(f"{kwargs['save_path']}/{organ}_biexpuptake_fit.png",dpi=300)
