import matplotlib.pyplot as plt


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