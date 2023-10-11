import matplotlib.pyplot as plt
import gatetools as gt
import itk
import numpy as np
import SimpleITK as sitk
import argparse
from pathlib import Path
import json

this_dir=Path(__file__).resolve().parent.parent
RADIOBIOLOGY_DATA_FILE = Path(this_dir,"data","radiobiology.json")


class Dosemap:
    def __init__(self, df, patient_id, cycle, dosemap, roi_masks_resampled, organlist):
        self.df = df
        self.patient_id = patient_id
        self.cycle = int(cycle)
        self.dosemap = dosemap 
        self.roi_masks_resampled = roi_masks_resampled
        self.organlist = organlist
        print(RADIOBIOLOGY_DATA_FILE)
        if RADIOBIOLOGY_DATA_FILE.exists():
            with open(RADIOBIOLOGY_DATA_FILE) as f:
                self.radiobiology_dic = json.load(f)
        else:
            print("The JSON file does not exist.")

        #with open(RADIOBIOLOGY_DATA_FILE) as f:
        #    self.radiobiology_dic = json.load(f)
        
    def image_visualisation(self, image):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
        axs[0].imshow(image[:, :, 50])
        axs[0].set_title('Slice at index 50')
        axs[1].imshow(image[65, :, :].T)
        axs[1].set_title('Slice at index 100')
        axs[2].imshow(image[:,47, :])
        axs[2].set_title('Slice at index 47')

        plt.tight_layout()
        plt.show()
        

    def show_mean_statistics(self):
        self.ad_mean = {}
        for organ in self.organlist:
            mask = self.roi_masks_resampled[organ]
            self.ad_mean[organ] = self.dosemap[mask].mean()
            print(f'{organ}', self.ad_mean[organ])
        #print(self.ad_mean)
            
    def show_max_statistics(self):
        for organ in self.organlist:
            mask = self.roi_masks_resampled[organ]
            x = self.dosemap[mask].max()
            print(f'{organ}', x)
            
    def dose_volume_histogram(self):
        doseimage = self.dosemap.astype(float)
        doseimage = itk.image_from_array(doseimage)

        for organ in self.organlist:
            x = self.roi_masks_resampled[organ]
            itkVol = x.astype(float)
            itkVol = itk.GetImageFromArray(itkVol)
            print(type(itkVol))
            zip =itk.LabelStatisticsImageFilter.New(doseimage)
            zip = zip.SetLabelInput()
            organ_data = gt.createDVH(doseimage, itkVol)
            print(organ_data)
            
    # equation based on the paper Bodei et al. "Long-term evaluation of renal toxicity after peptide receptor radionuclide therapy with 90Y-DOTATOC 
    # and 177Lu-DOTATATE: the role of associated risk factors"
    def calculate_bed(self, abs_dose_per_cycle, n_cycles, alpha_beta, t_repair):
        bed = 0



        for i in self.organlist:
            t_eff['i'] = self.df.loc[self.df['organ'] == 'i']['lamda_eff_1/s'].values[0]
            for j in range(1, n_cycles + 1):
                abs_dose = abs_dose_per_cycle[f'{j}']
                bed['i'] += abs_dose + 1/alpha_beta * t_repair/(t_repair + t_eff['i']) * abs_dose**2

        return bed


    def dataf(self):
        #print(self.df)
        print(self.df.loc[self.df['organ'] == 'Liver']['lamda_eff_1/s'].values[0])
        intersection = {key: self.organlist[key] for key in self.organlist if key in self.radiobiology_dic and self.organlist[key] == self.radiobiology_dic[key]}

        print(intersection)
    
    def save_dataframe(self):
        print(self.df)
        patient_data = self.df[(self.df['patient_id'] == self.patient_id) & (self.df['cycle'] == self.cycle)]
        patient_data['mean_ad[Gy]']= patient_data['organ'].map(self.ad_mean)
        print(patient_data)
        self.df.loc[(self.df['patient_id'] == self.patient_id) & (self.df['cycle'] == self.cycle)] = patient_data
        print(self.df)
        self.df.to_csv("/mnt/y/Sara/PR21_dosimetry/df.csv")
