# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
from datetime import datetime
import re
import glob
import pydicom 
from rt_utils import RTStructBuilder
import gatetools as gt
import scipy
import SimpleITK as sitk
from skimage import io
from skimage.transform import resize
from skimage import img_as_bool
from doodle.fits.fits import monoexp_fun, fit_monoexp
from doodle.plots.plots import monoexp_fit_plots


# %%
class PatientDosimetry:
    def __init__(self, df, patient_id, cycle, isotope, activity_tp1_df, activity_tp2_df, inj_timepoint1, inj_timepoint2, CT, SPECTMBq, roi_masks_resampled):
        self.df = df
        self.patient_id = patient_id
        self.cycle = cycle
        self.isotope = isotope
        self.activity_tp1_df = activity_tp1_df  
        self.activity_tp2_df = activity_tp2_df 
        self.inj_timepoint1 = inj_timepoint1
        self.inj_timepoint2 = inj_timepoint2 
        self.CT = CT 
        self.SPECTMBq = SPECTMBq 
        self.roi_masks_resampled = roi_masks_resampled

    def dataframe(self):
        #################################### Output dataframe #########################################
        if self.patient_id not in self.df['patient_id'].values:
            z = len(self.activity_tp1_df)
            self.df = pd.concat([self.df, pd.DataFrame({'patient_id': [self.patient_id] * z, 'cycle': [self.cycle] * z})],
                                ignore_index=True)  # add patient variables in columns: patient_id, cycle
            self.df['organ'] = self.activity_tp1_df['Contour'].tolist() + [None] * (len(self.df) - len(self.activity_tp1_df))  # copy observations from df `tp1`
            self.df['volume_ml'] = self.activity_tp1_df['Volume (ml)'].tolist() + [None] * (len(self.df) - len(self.activity_tp1_df))  # copy observations from df `tp1`
            print(f"Patient {self.patient_id} cycle0{self.cycle} loaded to the final dataframe.") 
        else:
            print(f"Patient {self.patient_id} cycle0{self.cycle} is already on the list!")
            
        column = pd.Series(index=self.df.index, name='lamda_eff_1/s')
        self.df = pd.concat((self.df, column), axis=1)
        column = pd.Series(index=self.df.index, name='a0_Bq')
        self.df = pd.concat((self.df, column), axis=1)
        column = pd.Series(index=self.df.index, name='tia_bqs')
        self.df = pd.concat((self.df, column), axis=1)
        column = pd.Series(index=self.df.index, name='tiac_h')
        self.df = pd.concat((self.df, column), axis=1)
        
        self.organslist = self.df['organ'].unique()        
            
    def fitting(self):
        ################################# Isotope information #########################################
        if self.isotope == '177Lu':
            half_life = 159.528 * 3600 #s
            decayconst = np.log(2)/half_life
        else:
            print('Isotope currently not implemented in the code. Add it in the class code.')        

        ###################################### Variables ##############################################          
        t = [self.inj_timepoint1.loc[0, 'delta_t_days'], self.inj_timepoint2.loc[0, 'delta_t_days']]
        ts = np.array([i * 24 * 3600 for i in t]) # seconds
        inj_activity = self.inj_timepoint1.loc[0, 'injected_activity_MBq'] # MBq
        inj_activity = inj_activity * 10**6 # Bq
        
        ###################################### Fitting ##############################################
        for index,row in self.df.iterrows():
            a_tp1 = self.activity_tp1_df.iloc[index][['Integral Total (BQML*ml)']].values
            a_tp2 = self.activity_tp2_df.iloc[index][['Integral Total (BQML*ml)']].values
            activity = [a_tp1, a_tp2]
            activity = np.array([float(arr[0]) for arr in activity]) # reduce dimensons and extract values
            popt,tt,yy,residuals = fit_monoexp(ts,activity,decayconst,monoguess=(1e6,1e-6))
            monoexp_fit_plots(ts,activity,tt,yy,row['organ'],popt[2],residuals)
            plt.show()
            self.df.at[index, 'lamda_eff_1/s'] = popt[1]
            self.df.at[index, 'a0_Bq'] = popt[0]
            self.df.at[index, 'tia_bqs'] = popt[0]/popt[1]
            self.df.at[index, 'tiac_h'] = (popt[0]/popt[1])/(inj_activity * 3600)

        ########################### Dictionary with lamda eff of each VOIs ############################
        self.lamda_eff_dict = {}
        for organ in self.organslist:
            organ_data = self.df[self.df['organ'] == organ]
            lamda_eff = organ_data['lamda_eff_1/s'].iloc[0]
            self.lamda_eff_dict[organ] = lamda_eff
        ## to do - save df frame
 
   
    def image_visualisation(self, image):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))    
        axs[0].imshow(image[:, :, 50])
        axs[0].set_title('Slice at index 50')
        axs[1].imshow(image[100, :, :].T)
        axs[1].set_title('Slice at index 100')
        axs[2].imshow(image[:,47, :])
        axs[2].set_title('Slice at index 47')

        plt.tight_layout()
        plt.show()
        
    def create_TIA(self):
        self.TIA = np.zeros((self.SPECTMBq.shape[0], self.SPECTMBq.shape[1], self.SPECTMBq.shape[2]))
        t2 = self.inj_timepoint2.loc[0, 'delta_t_days'] * 24 * 3600 # s
        
        for organ in self.organslist:
            self.TIA[self.roi_masks_resampled[organ]] = self.SPECTMBq[self.roi_masks_resampled[organ]] * np.exp(self.lamda_eff_dict[organ] * t2) / self.lamda_eff_dict[organ]

    def flip_images(self):
        self.CTp = np.transpose(self.CT, (2, 0, 1))
        print(self.CTp.shape)
        self.TIAp = np.transpose(self.TIA, (2, 0, 1))
        print('CT image:')
        self.image_visualisation(self.CTp)
        print('SPECT image:')
        self.image_visualisation(self.TIAp)
        
    def normalise_TIA(self):
        self.total_acc_A = np.sum(np.sum(np.sum(self.TIAp)))
        self.source_normalized = self.TIAp / self.total_acc_A

    def save_images_and_df(self):
        self.df.to_csv("/mnt/y/Sara/PR21_dosimetry/df.csv")
        
        self.CTp = np.array(self.CTp, dtype=np.float32)
        image = sitk.GetImageFromArray(self.CTp)
        image.SetSpacing([4.7952, 4.7952, 4.7952]) 
        sitk.WriteImage(image, f'{self.output_path}/MC/data/CT.mhd')
        
        self.source_normalized = np.array(self.source_normalized, dtype=np.float32)
        image2 = sitk.GetImageFromArray(self.source_normalized)
        image2.SetSpacing([4.7952, 4.7952, 4.7952]) 
        sitk.WriteImage(image2, f'{self.output_path}/MC/data/Source_normalized.mhd')

        folder = f'{self.output_path}/MC/output'
        image_path = os.path.join(folder, 'TotalAccA.txt')
        with open(image_path, 'w') as fileID:
            fileID.write('%.2f' % self.total_acc_A)

# %%
