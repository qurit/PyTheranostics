import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

from doodle.fits.fits import (find_a_initial, fit_biexp_uptake, fit_monoexp,
                              fit_trapezoid)
from doodle.plots.plots import biexp_fit_plots, monoexp_fit_plots


# %%
class BaseDosimetry:
    """Base class for Image-based dosimetry calculations"""
    def __init__(self, patient_id: str, cycle: int, isotope, 
                 CT, SPECTMBq, roi_masks_resampled,
                 injected_activity_MBq, injection_datetime, 
                 activity_tp1_df, 
                 scan1_datetime, delta1_t_days, 
                 activity_tp2_df = None, 
                 scan2_datetime = None, delta2_t_days = None):
        
        self.patient_id = patient_id
        self.cycle = int(cycle)
        self.isotope = isotope
        self.activity_tp1_df = activity_tp1_df  
        self.activity_tp2_df = activity_tp2_df 
        self.injected_activity_MBq = injected_activity_MBq
        self.injection_datetime = injection_datetime   
        self.scan1_datetime = scan1_datetime  
        self.delta1_t_days = delta1_t_days  
        self.scan2_datetime = scan2_datetime 
        self.delta2_t_days = delta2_t_days  
        self.CT = CT 
        self.SPECTMBq = SPECTMBq 
        self.roi_masks_resampled = roi_masks_resampled

    def dataframe(self):
        #################################### Output dataframe #########################################            
        self.new_data = pd.DataFrame({
            'patient_id': [self.patient_id] * len(self.activity_tp1_df),
            'cycle': [self.cycle] * len(self.activity_tp1_df),
            'organ': self.activity_tp1_df['Contour'].tolist(),
            'volume_ml': self.activity_tp1_df['Volume (ml)'].tolist()
        })

        self.organslist = self.activity_tp1_df['Contour'].unique()
        return self.organslist        
    
    # TODO: automate the categorisatioon of the lesions into these groups
    def subtruct_lesions_from_healthy_vois(self, bone_metastases, liver_metastases, lymph_metastases):
        if bone_metastases != None:
            meta = self.activity_tp1_df[self.activity_tp1_df['Contour'].isin(bone_metastases)]
            self.activity_tp1_df.loc[self.activity_tp1_df['Contour'] == 'Skeleton', 'Integral Total (BQML*ml)'] = self.activity_tp1_df.loc[self.activity_tp1_df['Contour'] == 'Skeleton', 'Integral Total (BQML*ml)'] - meta['Integral Total (BQML*ml)'].sum()
        
        if liver_metastases != None:
            meta = self.activity_tp1_df[self.activity_tp1_df['Contour'].isin(liver_metastases)]
            self.activity_tp1_df.loc[self.activity_tp1_df['Contour'] == 'Liver', 'Integral Total (BQML*ml)'] = self.activity_tp1_df.loc[self.activity_tp1_df['Contour'] == 'Liver', 'Integral Total (BQML*ml)'] - meta['Integral Total (BQML*ml)'].sum()
                
    def fitting(self, fitting_methods):
        ################################# Isotope information #########################################
        if self.isotope == '177Lu':
            half_life = 159.528 * 3600 #s
            decayconst = np.log(2)/half_life
        else:
            print('Isotope currently not implemented in the code. Add it in the class code.')        
        ###################################### Variables ##############################################          
        t = [self.delta1_t_days, self.delta2_t_days]
        ts = np.array([i * 24 * 3600 for i in t]) # seconds
        inj_activity = self.injected_activity_MBq # MBq
        #inj_activity = inj_activity * 10**6 # Bq
        ###################################### Fitting ##############################################
        for index,row in self.new_data.iterrows():
            organ = row['organ']
            method = fitting_methods.get(organ)
            
            a_tp1 = self.activity_tp1_df.iloc[index][['Integral Total (BQML*ml)']].values
            a_tp2 = self.activity_tp2_df.iloc[index][['Integral Total (BQML*ml)']].values
            activity = [a_tp1, a_tp2]
            activity = np.array([float(arr[0]) for arr in activity]) # reduce dimensons and extract values
            
            if method == 'monoexp':
                popt,tt,yy,residuals = fit_monoexp(ts,activity,decayconst,monoguess=(1e6,1e-5))
                monoexp_fit_plots(ts  / (3600 * 24) ,activity, tt  / (3600 * 24),yy,row['organ'],popt[2],residuals)
            elif method == 'biexp':
                popt, tt, yy, residuals = fit_biexp_uptake(ts, activity, decayconst, uptakeguess=(6, 1e-3, 1e-3))
                biexp_fit_plots(ts / (3600 * 24), activity / 10**6, tt / (3600 * 24), yy / 10**6, organ, popt[2], residuals)
            elif method == 'trapezoid':
                auc = fit_trapezoid(ts, activity, decayconst)
            else:
                print(f'Unknown method for organ {organ}')
                continue
            plt.show()

            self.new_data.at[index, 'lamda_eff_1/s'] = popt[1]
            self.new_data.at[index, 'a0_MBq'] = popt[0]
            self.new_data.at[index, 'tia_MBqs'] = popt[0]/popt[1]
            self.new_data.at[index, 'TIA_h'] = (popt[0]/popt[1])/(inj_activity * 3600)
            
        ########################### Dictionary with lamda eff of each VOIs ############################
        self.lamda_eff_dict = {}
        for organ in self.organslist:
            organ_data = self.new_data[self.new_data['organ'] == organ]
            lamda_eff = organ_data['lamda_eff_1/s'].iloc[0]
            self.lamda_eff_dict[organ] = lamda_eff
            
        return self.new_data, self.lamda_eff_dict

    def takefitfromcycle1_andintegrate(self):
        cycle1_df = pd.read_csv(f"/mnt/y/Sara/PR21_dosimetry/output/{self.patient_id}_cycle01_dosimetry_output.csv")
        t = [self.inj_timepoint1.loc[0, 'delta_t_days']]
        ts = np.array([i * 24 * 3600 for i in t]) # seconds
        inj_activity = self.inj_timepoint1.loc[0, 'injected_activity_MBq'] # MBq
        #inj_activity = inj_activity * 10**6 # Bq
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
        self.new_data['a0_MBq'] = self.new_data['organ'].map(a0_Bq_dict)
        self.new_data['tia_MBqs'] = self.new_data['a0_Bq'] / self.new_data['lamda_eff_1/s']
        self.new_data['TIA_h'] = (self.new_data['a0_Bq'] / self.new_data['lamda_eff_1/s'])/(inj_activity * 3600)

        return self.new_data, self.lamda_eff_dict
   
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

        if self.cycle == 1:
            time = self.inj_timepoint2.loc[0, 'delta_t_days'] * 24 * 3600 # s
        else:
            time = self.inj_timepoint1.loc[0, 'delta_t_days'] * 24 * 3600 # s
                            
        TIA_WB = np.zeros((self.SPECTMBq.shape[0], self.SPECTMBq.shape[1], self.SPECTMBq.shape[2]))
        TIA_organs = np.zeros((self.SPECTMBq.shape[0], self.SPECTMBq.shape[1], self.SPECTMBq.shape[2]))
            
        TIA_WB[self.roi_masks_resampled['WBCT']] = self.SPECTMBq[self.roi_masks_resampled['WBCT']] * np.exp(self.lamda_eff_dict['WBCT'] * time) / self.lamda_eff_dict['WBCT']
        organs = [organ for organ in self.organslist if organ not in ["WBCT", "ROB", "Liver_Reference", "TotalTumorBurden", "Kidney_L_m", 'Kidney_R_m']]
        index_skeleton = organs.index('Skeleton')
        organs.pop(index_skeleton)

        # Insert 'Skeleton' at the beginning of the list
        organs.insert(0, 'Skeleton')


        for organ in organs:
            TIA_organs[self.roi_masks_resampled[organ]] = self.SPECTMBq[self.roi_masks_resampled[organ]] * np.exp(self.lamda_eff_dict[organ] * time) / self.lamda_eff_dict[organ]
            

        TIA_ROB = TIA_WB - TIA_organs
        self.TIA = TIA_ROB + TIA_organs

        

    def flip_images(self):
        self.CTp = np.transpose(self.CT, (2, 0, 1))
        self.TIAp = np.transpose(self.TIA, (2, 0, 1))
        self.SPECTMBqp = np.transpose(self.SPECTMBq, (2, 0, 1))
        print('CT image:')
        self.image_visualisation(self.CTp)
        print('TIA image:')
        self.image_visualisation(self.TIAp)
        
        return self.TIAp, self.SPECTMBqp 
        
    def normalise_TIA(self):
        self.total_acc_A = np.sum(np.sum(np.sum(self.TIAp)))
        self.source_normalized = self.TIAp / self.total_acc_A

    def save_images(self):
        self.CTp = np.array(self.CTp, dtype=np.float32)
        image = sitk.GetImageFromArray(self.CTp)
        image.SetSpacing([4.7952, 4.7952, 4.7952]) 
        sitk.WriteImage(image, f'/mnt/y/Sara/PR21_dosimetry/{self.patient_id}/cycle0{self.cycle}/MC/data/CT.mhd')
        
        self.source_normalized = np.array(self.source_normalized, dtype=np.float32)
        image2 = sitk.GetImageFromArray(self.source_normalized)
        image2.SetSpacing([4.7952, 4.7952, 4.7952]) 
        sitk.WriteImage(image2, f'/mnt/y/Sara/PR21_dosimetry/{self.patient_id}/cycle0{self.cycle}/MC/data/Source_normalized.mhd')

        folder = f'/mnt/y/Sara/PR21_dosimetry/{self.patient_id}/cycle0{self.cycle}/MC/output'
        image_path = os.path.join(folder, 'TotalAccA.txt')
        with open(image_path, 'w') as fileID:
            fileID.write('%.2f' % self.total_acc_A)
            
        
# %%
