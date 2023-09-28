import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import scipy
import skimage.io as io
from scipy import signal


class DoseVoxelKernel:
    def __init__(self, TIAMBqs, CT):
        self.TIAMBqs = np.asarray(TIAMBqs, dtype=np.float64)
        self.CT = CT 

    def kernel(self):
        vsv = np.fromfile('/mnt/c/Users/skurkowska/Desktop/PR21_dosimetry/convolution/DK_softtissueICRP_mGyperMBqs.img', dtype='float32')
        vsv = vsv.reshape((51,51,51))
        self.vsv = np.asarray(vsv, dtype=np.float64)

            
    def convolution(self):
        conv_img = signal.fftconvolve(self.TIAMBqs, self.vsv, mode='same', axes=None)
        np.save('convolved_vol.npy', conv_img)
        #plt.imshow(conv_img[60,:,:])
        #print(type(conv_img))
        self.conv_img= np.transpose(conv_img, (1, 2, 0))
        return self.conv_img
        
    def weighting(self, image):
        xdim = self.CT.shape[0]
        ydim = self.CT.shape[1]
        zdim = self.CT.shape[2]

        # Assumed density for simulated voxel s values [g/ccm] --> voxel mass [g]
        rho_ICRPsoft = 1.00

        # Start HU to density conversion based on CT
        # Taken from patient-HUmaterials.db
        # Conversion according to Schneider et al. 2000
        # RHOMAT=zeros(xdim,ydim,zdim);
        


        # Initialize the RHOMAT array with the same shape as CT
        RHOMAT = np.zeros_like(self.CT)
        for i in range(xdim):
            for j in range(ydim):
                for k in range(zdim):
                    if self.CT[i, j, k] <= -1050:
                        self.CT[i, j, k] = -1050
                    # taken from patient-HUmaterials.db from GATE, rho in g/ccm
                    if self.CT[i, j, k] >= -1050:
                        if self.CT[i, j, k] <= -1050:
                            RHOMAT[i, j, k] = 0.00121
                        elif self.CT[i, j, k] <= -950:
                            RHOMAT[i, j, k] = 0.102695
                        elif self.CT[i, j, k] <= -852.884:
                            RHOMAT[i, j, k] = 0.202695
                        elif self.CT[i, j, k] <= -755.769:
                            RHOMAT[i, j, k] = 0.302695
                        elif self.CT[i, j, k] <= -658.653:
                            RHOMAT[i, j, k] = 0.402695                     
                        elif self.CT[i, j, k] <= -561.538:
                            RHOMAT[i, j, k]=0.502695 
                        elif self.CT[i, j, k] <= -464.422:
                            RHOMAT[i, j, k]=0.602695 
                        elif self.CT[i, j, k] <= -367.306:
                            RHOMAT[i, j, k]=0.702695 
                        elif self.CT[i, j, k] <= -270.191:
                            RHOMAT[i, j, k]=0.802695 
                        elif self.CT[i, j, k] <= -173.075:
                            RHOMAT[i, j, k]=0.880021                     
                        elif self.CT[i, j, k] <= -120:
                            RHOMAT[i, j, k]=0.926911 
                        elif self.CT[i, j, k] <= -82:
                            RHOMAT[i, j, k]=0.957382 
                        elif self.CT[i, j, k] <= -52:
                            RHOMAT[i, j, k]=0.984277 
                        elif self.CT[i, j, k] <= -22:
                            RHOMAT[i, j, k]=1.01117 
                        elif self.CT[i, j, k] <= 8:
                            RHOMAT[i, j, k]=1.02955 
                        elif self.CT[i, j, k] <= 19:
                            RHOMAT[i, j, k]=1.0616 
                        elif self.CT[i, j, k] <= 80:
                            RHOMAT[i, j, k]=1.1199 
                        elif self.CT[i, j, k] <= 120:
                            RHOMAT[i, j, k]=1.11115 
                        elif self.CT[i, j, k] <= 200:
                            RHOMAT[i, j, k]=1.16447 
                        elif self.CT[i, j, k] <= 300:
                            RHOMAT[i, j, k]=1.22371         
                        elif self.CT[i, j, k] <= 400:
                            RHOMAT[i, j, k]=1.28295         
                        elif self.CT[i, j, k] <= 500:
                            RHOMAT[i, j, k]=1.34219         
                        elif self.CT[i, j, k] <= 600:
                            RHOMAT[i, j, k]=1.40142                             
                        elif self.CT[i, j, k] <= 700:
                            RHOMAT[i, j, k]=1.46066         
                        elif self.CT[i, j, k] <= 800:
                            RHOMAT[i, j, k]=1.5199         
                        elif self.CT[i, j, k] <= 900:
                            RHOMAT[i, j, k]=1.57914         
                        elif self.CT[i, j, k] <= 1000:
                            RHOMAT[i, j, k]=1.63838         
                        elif self.CT[i, j, k] <= 1100:
                            RHOMAT[i, j, k]=1.69762         
                        elif self.CT[i, j, k] <= 1200:
                            RHOMAT[i, j, k]=1.75686         
                        elif self.CT[i, j, k] <= 1300:
                            RHOMAT[i, j, k]=1.8161         
                        elif self.CT[i, j, k] <= 1400:
                            RHOMAT[i, j, k]=1.87534         
                        elif self.CT[i, j, k] <= 1500:
                            RHOMAT[i, j, k]=1.94643         
                        elif self.CT[i, j, k] <= 1640:
                            RHOMAT[i, j, k]=2.03808    
                        elif self.CT[i, j, k] <= 1807.5:
                            RHOMAT[i, j, k]=2.13808    
                        elif self.CT[i, j, k] <= 1975.01:
                            RHOMAT[i, j, k]=2.23808    
                        elif self.CT[i, j, k] <= 2142.51:
                            RHOMAT[i, j, k]=2.33509    
                        elif self.CT[i, j, k] <= 2300:
                            RHOMAT[i, j, k]=2.4321    
                        elif self.CT[i, j, k] <= 2467.5:
                            RHOMAT[i, j, k]=2.5321    
                        elif self.CT[i, j, k] <= 2635.01:
                            RHOMAT[i, j, k]=2.6321    
                        elif self.CT[i, j, k] <= 2802.51:
                            RHOMAT[i, j, k]=2.7321    
                        elif self.CT[i, j, k] <= 2970.02:
                            RHOMAT[i, j, k]=2.79105    
                        elif self.CT[i, j, k] <= 3000:
                            RHOMAT[i, j, k]=2.9   
                        elif self.CT[i, j, k] <= 4000:
                            RHOMAT[i, j, k] = 2.9
                        elif self.CT[i, j, k] <= 4001:
                            RHOMAT[i, j, k] = 2.9
        # RHOMAT now contains the converted values

        weighting = np.zeros_like(RHOMAT)
        for i in range(xdim):
            for j in range(ydim):
                for k in range(zdim):
                    weighting[i, j, k] = rho_ICRPsoft / RHOMAT[i, j, k]
                    

        # Calculate weighted image
        image_weighted = np.zeros_like(image)
        for i in range(xdim):
            for j in range(ydim):
                for k in range(zdim):
                    image_weighted[i, j, k] = image[i, j, k] * weighting[i, j, k]
                    
        return RHOMAT, image_weighted

