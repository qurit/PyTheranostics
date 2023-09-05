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
from doodle.fits.fits import monoexp_fun, fit_monoexp, find_a_initial
from doodle.plots.plots import monoexp_fit_plots


# %%
class PatientDosimetry:
    def __init__(self, df, patient_id, cycle, isotope, CT, SPECTMBq, roi_masks_resampled, activity_tp1_df, inj_timepoint1, activity_tp2_df = None, inj_timepoint2 = None):
        self.df = df
        self.patient_id = patient_id
        self.cycle = int(cycle)
        self.isotope = isotope
        self.activity_tp1_df = activity_tp1_df  
        self.activity_tp2_df = activity_tp2_df 
        self.inj_timepoint1 = inj_timepoint1
        self.inj_timepoint2 = inj_timepoint2  
        self.CT = CT 
        self.SPECTMBq = SPECTMBq 
        self.roi_masks_resampled = roi_masks_resampled

    def dataframe(self):
        self.new_data = pd.DataFrame()
        #################################### Output dataframe #########################################            
        if self.patient_id in self.df['patient_id'].values:
            if self.cycle in self.df['cycle'].values:
                print(f"Patient {self.patient_id} cycle0{self.cycle} is already on the list!")
            elif self.cycle not in self.df['cycle'].values:
                print("im here")
                self.new_data = pd.DataFrame({
                    'patient_id': [self.patient_id] * len(self.activity_tp1_df),
                    'cycle': [self.cycle] * len(self.activity_tp1_df),
                    'organ': self.activity_tp1_df['Contour'].tolist(),
                    'volume_ml': self.activity_tp1_df['Volume (ml)'].tolist()
                    })
                #self.df = pd.concat([self.df, new_data], ignore_index=True)
                #print(f"Patient {self.patient_id} cycle0{self.cycle} loaded to the final dataframe.")
        else:
            print("3")
            self.new_data = pd.DataFrame({
                'patient_id': [self.patient_id] * len(self.activity_tp1_df),
                'cycle': [self.cycle] * len(self.activity_tp1_df),
                'organ': self.activity_tp1_df['Contour'].tolist(),
                'volume_ml': self.activity_tp1_df['Volume (ml)'].tolist()
                })
        print(self.new_data)
        self.organslist = self.activity_tp1_df['Contour'].unique()        
        
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
        for index,row in self.new_data.iterrows():
            a_tp1 = self.activity_tp1_df.iloc[index][['Integral Total (BQML*ml)']].values
            a_tp2 = self.activity_tp2_df.iloc[index][['Integral Total (BQML*ml)']].values
            activity = [a_tp1, a_tp2]
            activity = np.array([float(arr[0]) for arr in activity]) # reduce dimensons and extract values
            popt,tt,yy,residuals = fit_monoexp(ts,activity,decayconst,monoguess=(1e6,1e-6))
            monoexp_fit_plots(ts,activity,tt,yy,row['organ'],popt[2],residuals)
            plt.show()
            self.new_data.at[index, 'lamda_eff_1/s'] = popt[1]
            self.new_data.at[index, 'a0_Bq'] = popt[0]
            self.new_data.at[index, 'tia_bqs'] = popt[0]/popt[1]
            self.new_data.at[index, 'tiac_h'] = (popt[0]/popt[1])/(inj_activity * 3600)

        ########################### Dictionary with lamda eff of each VOIs ############################
        self.lamda_eff_dict = {}
        for organ in self.organslist:
            organ_data = self.new_data[self.new_data['organ'] == organ]
            lamda_eff = organ_data['lamda_eff_1/s'].iloc[0]
            self.lamda_eff_dict[organ] = lamda_eff

    def takefitfromcycle1_andintegrate(self):
        cycle1_df = self.df[(self.df['patient_id'] == self.patient_id) & (self.df['cycle'] == 1)]
        t = [self.inj_timepoint1.loc[0, 'delta_t_days']]
        ts = np.array([i * 24 * 3600 for i in t]) # seconds
        inj_activity = self.inj_timepoint1.loc[0, 'injected_activity_MBq'] # MBq
        inj_activity = inj_activity * 10**6 # Bq
        self.lamda_eff_dict = {}
        a0_Bq_dict = {}
        for organ in self.organslist:
            if organ in cycle1_df['organ'].values:
                # Get the lambda_eff value for the organ from cycle 1
                lamda_eff = cycle1_df.loc[cycle1_df['organ'] == organ, 'lamda_eff_1/s'].iloc[0]
                self.lamda_eff_dict[organ] = np.float128(lamda_eff)
                a_tp1 = self.activity_tp1_df.loc[self.activity_tp1_df['Contour'] == organ, 'Integral Total (BQML*ml)'].iloc[0] 
                aO_Bq = find_a_initial(a_tp1 , lamda_eff, ts)
                a0_Bq_dict[organ] = aO_Bq[0]
            else:
                self.lamda_eff_dict[organ] = None  
                print(f"The {organ} is not found in the cycle 1.")

        self.new_data['lamda_eff_1/s'] = self.new_data['organ'].map(self.lamda_eff_dict)
        self.new_data['a0_Bq'] = self.new_data['organ'].map(a0_Bq_dict)
        self.new_data['tia_bqs'] = self.new_data['a0_Bq'] / self.new_data['lamda_eff_1/s']
        self.new_data['tiac_h'] = (self.new_data['a0_Bq'] / self.new_data['lamda_eff_1/s'])/(inj_activity * 3600)
   
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
        if self.cycle == 1:
            time = self.inj_timepoint2.loc[0, 'delta_t_days'] * 24 * 3600 # s
        else:
            time = self.inj_timepoint1.loc[0, 'delta_t_days'] * 24 * 3600 # s
        
        for organ in self.organslist:
            self.TIA[self.roi_masks_resampled[organ]] = self.SPECTMBq[self.roi_masks_resampled[organ]] * np.exp(self.lamda_eff_dict[organ] * time) / self.lamda_eff_dict[organ]

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
        self.df = pd.concat([self.df, self.new_data], ignore_index=True)
        self.df = self.df.dropna(subset=['patient_id'])
        print(self.df)
        self.df.to_csv("/mnt/y/Sara/PR21_dosimetry/df.csv")
        
#        self.CTp = np.array(self.CTp, dtype=np.float32)
#        image = sitk.GetImageFromArray(self.CTp)
#        image.SetSpacing([4.7952, 4.7952, 4.7952]) 
#        sitk.WriteImage(image, f'{self.output_path}/MC/data/CT.mhd')
#        
#        self.source_normalized = np.array(self.source_normalized, dtype=np.float32)
#        image2 = sitk.GetImageFromArray(self.source_normalized)
#        image2.SetSpacing([4.7952, 4.7952, 4.7952]) 
#        sitk.WriteImage(image2, f'{self.output_path}/MC/data/Source_normalized.mhd')
#
#        folder = f'{self.output_path}/MC/output'
#        image_path = os.path.join(folder, 'TotalAccA.txt')
#        with open(image_path, 'w') as fileID:
#            fileID.write('%.2f' % self.total_acc_A)

# %%
