import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import scipy
import skimage.io as io
from scipy import signal


class DoseVoxelKernel:
    def __init__(self, TIAMBqs):
        self.TIAMBqs = np.asarray(TIAMBqs, dtype=np.float64) 

    def kernel(self):
        vsv = np.fromfile('/mnt/c/Users/skurkowska/Desktop/PR21_dosimetry/convolution/DK_softtissueICRP_mGyperMBqs.img', dtype='float32')
        vsv = vsv.reshape((51,51,51))
        self.vsv = np.asarray(vsv, dtype=np.float64)
            
    def convolution(self):
        conv_img = signal.fftconvolve(self.TIAMBqs, self.vsv, mode='same', axes=None)
        np.save('convolved_vol.npy', conv_img)
        plt.imshow(conv_img[60,:,:])
        print(type(conv_img))
        conv_img= np.transpose(conv_img, (1, 2, 0))
        return conv_img
        