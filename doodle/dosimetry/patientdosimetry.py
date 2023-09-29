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
import SimpleITK as sitk
import itk
from skimage import io
from skimage.transform import resize
from skimage import img_as_bool
from doodle.fits.fits import monoexp_fun, fit_monoexp, find_a_initial, fit_biexp_uptake
from doodle.plots.plots import monoexp_fit_plots, biexp_fit_plots

# %%
itk.LabelStatisticsImageFilter.GetTypes()
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
        # Calculate the sum of 'volume_ml' for observations other than 'WBCT'
        filtered_df = self.activity_tp1_df[(self.activity_tp1_df['Contour'] == 'TotalTumorBurden') | 
                                   (self.activity_tp1_df['Contour'] == 'Kidney_L_a') |
                                   (self.activity_tp1_df['Contour'] == 'Kidney_R_a') |
                                   (self.activity_tp1_df['Contour'] == 'Liver') |
                                   (self.activity_tp1_df['Contour'] == 'ParotidglandL') |
                                   (self.activity_tp1_df['Contour'] == 'ParotidglandR') |
                                   (self.activity_tp1_df['Contour'] == 'Spleen') |
                                   (self.activity_tp1_df['Contour'] == 'Skeleton') |
                                   (self.activity_tp1_df['Contour'] == 'Bladder_Experimental') |
                                   (self.activity_tp1_df['Contour'] == 'SubmandibularglandL') |
                                   (self.activity_tp1_df['Contour'] == 'SubmandibularglandR')]


        # Specify the values for the new observation
        rob_observation = {
            'Contour': 'ROB',
            'Series Date': 'x',
            'Integral Total (BQML*ml)': self.activity_tp1_df.loc[self.activity_tp1_df['Contour'] == 'WBCT', 'Integral Total (BQML*ml)'] - filtered_df['Integral Total (BQML*ml)'].sum(),
            'Max (BQML)': 'x',
            'Mean Ratio (-)': 'x',
            'Total (BQML)': 'x',
            'Volume (ml)': self.activity_tp1_df.loc[self.activity_tp1_df['Contour'] == 'WBCT', 'Volume (ml)'] - filtered_df['Volume (ml)'].sum(),
            'Voxel Count (#)': 'x'
        }
        #activity_tp1_df[len(activity_tp1_df)] = rob_observation
        rob_observation = pd.DataFrame([rob_observation])
        self.activity_tp1_df = pd.concat([self.activity_tp1_df, rob_observation], ignore_index=True)

        # Specify the values for the new observation
        rob_observation = {
            'Contour': 'ROB',
            'Series Date': 'x',
            'Integral Total (BQML*ml)': self.activity_tp2_df.loc[self.activity_tp2_df['Contour'] == 'WBCT', 'Integral Total (BQML*ml)'] - filtered_df['Integral Total (BQML*ml)'].sum(),
            'Max (BQML)': 'x',
            'Mean Ratio (-)': 'x',
            'Total (BQML)': 'x',
            'Volume (ml)': self.activity_tp2_df.loc[self.activity_tp2_df['Contour'] == 'WBCT', 'Volume (ml)'] - filtered_df['Volume (ml)'].sum(),
            'Voxel Count (#)': 'x'
        }
        rob_observation = pd.DataFrame([rob_observation])
        self.activity_tp2_df = pd.concat([self.activity_tp2_df, rob_observation], ignore_index=True)
        
        self.new_data = pd.DataFrame()
        #################################### Output dataframe #########################################            
        if self.patient_id in self.df['patient_id'].values:
            if self.cycle in self.df['cycle'].values:
                print(f"Patient {self.patient_id} cycle0{self.cycle} is already on the list!")
            elif self.cycle not in self.df['cycle'].values:
                self.new_data = pd.DataFrame({
                    'patient_id': [self.patient_id] * len(self.activity_tp1_df),
                    'cycle': [self.cycle] * len(self.activity_tp1_df),
                    'organ': self.activity_tp1_df['Contour'].tolist(),
                    'volume_ml': self.activity_tp1_df['Volume (ml)'].tolist()
                    })
        else:
            self.new_data = pd.DataFrame({
                'patient_id': [self.patient_id] * len(self.activity_tp1_df),
                'cycle': [self.cycle] * len(self.activity_tp1_df),
                'organ': self.activity_tp1_df['Contour'].tolist(),
                'volume_ml': self.activity_tp1_df['Volume (ml)'].tolist()
                })

#        # Calculate the sum of 'volume_ml' for observations other than 'WBCT'
#        sum_without_wbct = self.new_data[self.new_data['organ'] != 'WBCT']['volume_ml'].sum()
#
#        # Specify the values for the new observation
#        rob_observation = {
#            'patient_id': self.patient_id,
#            'cycle': self.cycle,
#            'organ': 'ROB',
#            'volume_ml': self.new_data.loc[self.new_data['organ'] == 'WBCT', 'volume_ml'].values[0] - sum_without_wbct
#        }
#        self.new_data.loc[len(self.new_data)] = rob_observation



        print(self.new_data)
        self.organslist = self.activity_tp1_df['Contour'].unique()
        return self.organslist        
        
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
#            popt,tt,yy,residuals = fit_monoexp(ts,activity,deayconst,monoguess=(1e6,1e-5))
#            monoexp_fit_plots(ts  / (3600 * 24) ,activity / 10**6,tt  / (3600 * 24),yy / 10**6,row['organ'],popt[2],residuals)
#            plt.show()
#            elif function == 2:
#                popt,tt,yy,residuals = fit_biexp_uptake(ts,activity,decayconst,uptakeguess=(6,1e-3 , 1e-3))
#                biexp_fit_plots(ts / (3600 * 24), activity / 10**6, tt  / (3600 * 24),yy / 10**6,row['organ'],popt[2],residuals)
#                #monoexp_fit_plots(ts, activity, tt,yy,row['organ'],popt[2],residuals)
#                #plt.show()
            #else:
            #    print('Function unknown')
            popt,tt,yy,residuals = fit_monoexp(ts,activity,decayconst,monoguess=(1e6,1e-5))
            monoexp_fit_plots(ts  / (3600 * 24) ,activity / 10**6,tt  / (3600 * 24),yy / 10**6,row['organ'],popt[2],residuals)
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

        ##############################################################################################
        ####################################### OPTION 1 #############################################
        ##############################################################################################
#        self.TIA = np.zeros((self.SPECTMBq.shape[0], self.SPECTMBq.shape[1], self.SPECTMBq.shape[2]))
#        if self.cycle == 1:
#            time = self.inj_timepoint2.loc[0, 'delta_t_days'] * 24 * 3600 # s
#        else:
#            time = self.inj_timepoint1.loc[0, 'delta_t_days'] * 24 * 3600 # s
#
#        organs = [organ for organ in self.organslist if organ != "ROB"]
#        print(organs)
#        for organ in organs:
#            self.TIA[self.roi_masks_resampled[organ]] = self.SPECTMBq[self.roi_masks_resampled[organ]] * np.exp(self.lamda_eff_dict[organ] * time) / self.lamda_eff_dict[organ]
                     
        ##############################################################################################
        ####################################### OPTION 1 #############################################
        ##############################################################################################
#        self.TIA = np.zeros((self.SPECTMBq.shape[0], self.SPECTMBq.shape[1], self.SPECTMBq.shape[2]))
#        if self.cycle == 1:
#            time = self.inj_timepoint2.loc[0, 'delta_t_days'] * 24 * 3600 # s
#        else:
#            time = self.inj_timepoint1.loc[0, 'delta_t_days'] * 24 * 3600 # s

                                    
        #organs = [organ for organ in self.organslist if organ != "WBCT"]
        #last_element = organs.pop()  # Remove the ROB
        #organs.insert(0, last_element)  # Insert it at the beginning
        #organs = np.array(organs)
        #print(organs)
        #for organ in organs:
        #    self.TIA[self.roi_masks_resampled[organ]] = self.SPECTMBq[self.roi_masks_resampled[organ]] * np.exp(self.lamda_eff_dict[organ] * time) / self.lamda_eff_dict[organ]


        ##############################################################################################
        ####################################### OPTION 2 #############################################
        ##############################################################################################

#        if self.cycle == 1:
#            time = self.inj_timepoint2.loc[0, 'delta_t_days'] * 24 * 3600 # s
#        else:
#            time = self.inj_timepoint1.loc[0, 'delta_t_days'] * 24 * 3600 # s
#                            
#        TIA_WB = np.zeros((self.SPECTMBq.shape[0], self.SPECTMBq.shape[1], self.SPECTMBq.shape[2]))
#        TIA_organs = np.zeros((self.SPECTMBq.shape[0], self.SPECTMBq.shape[1], self.SPECTMBq.shape[2]))
#            
#        TIA_WB[self.roi_masks_resampled['WBCT']] = self.SPECTMBq[self.roi_masks_resampled['WBCT']] * np.exp(self.lamda_eff_dict['WBCT'] * time) / self.lamda_eff_dict['WBCT']
#        organs = [organ for organ in self.organslist if organ not in ["WBCT", "ROB", "Liver_Reference", "TotalTumorBurden", "Kidney_L_m", 'Kidney_R_m']]
#        index_skeleton = organs.index('Skeleton')
#        organs.pop(index_skeleton)
#
#        # Insert 'Skeleton' at the beginning of the list
#        organs.insert(0, 'Skeleton')
#
#        print(organs)
#        for organ in organs:
#            TIA_organs[self.roi_masks_resampled[organ]] = self.SPECTMBq[self.roi_masks_resampled[organ]] * np.exp(self.lamda_eff_dict[organ] * time) / self.lamda_eff_dict[organ]
#            
#        print(TIA_WB.shape)
#        print(TIA_organs.shape)
#        TIA_ROB = TIA_WB - TIA_organs
#        self.TIA = TIA_ROB + TIA_organs

        ##############################################################################################
        ####################################### OPTION 3 #############################################
        ##############################################################################################

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
        print(organs)
        for organ in organs:
            TIA_organs[self.roi_masks_resampled[organ]] = self.SPECTMBq[self.roi_masks_resampled[organ]] * np.exp(self.lamda_eff_dict[organ] * time) / self.lamda_eff_dict[organ]
            
        TIA_ROB = TIA_WB - TIA_organs
        TIA_ROB_value = np.sum(np.sum(np.sum(TIA_ROB)))
        ROB_no_voxels = np.sum(np.sum(np.sum(self.roi_masks_resampled['ROB'])))

        TIA_value_per_voxel = TIA_ROB_value / ROB_no_voxels
        TIA_ROB[self.roi_masks_resampled['ROB']] = TIA_value_per_voxel
        
        self.TIA = TIA_ROB + TIA_organs
        

        
        return self.TIA

    def flip_images(self):
        self.CTp = np.transpose(self.CT, (2, 0, 1))
        self.TIAp = np.transpose(self.TIA, (2, 0, 1))
        print('CT image:')
        self.image_visualisation(self.CTp)
        print('TIA image:')
        self.image_visualisation(self.TIAp)
        
        return self.TIAp
        
    def normalise_TIA(self):
        self.total_acc_A = np.sum(np.sum(np.sum(self.TIAp)))
        self.source_normalized = self.TIAp / self.total_acc_A

    def save_images_and_df(self):
        self.df = pd.concat([self.df, self.new_data], ignore_index=True)
        self.df = self.df.dropna(subset=['patient_id'])
        print(self.df)
        self.df.to_csv("/mnt/y/Sara/PR21_dosimetry/df.csv")
        
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
