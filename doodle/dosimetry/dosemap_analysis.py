import matplotlib.pyplot as plt
import gatetools as gt
import itk
import numpy as np
import SimpleITK as sitk
import argparse
from pathlib import Path
import json

this_dir=Path(__file__).resolve().parent.parent
RADIOBIOLOGY_DATA_FILE = Path(this_dir,"data","radiobiology.json") #important to remember that in dictionary, alpha_beta is in Gy, and t_repair in h


class Dosemap:
    def __init__(self, df, patient_id, cycle, dosemap, roi_masks_resampled, organlist):
        self.df = df
        self.patient_id = patient_id
        self.cycle = int(cycle)
        self.dosemap = dosemap 
        self.roi_masks_resampled = roi_masks_resampled
        self.organlist = organlist
        with open(RADIOBIOLOGY_DATA_FILE) as f:
            self.radiobiology_dic = json.load(f)
        
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
            self.ad_mean[organ] = self.dosemap[mask].mean() / 1000
            #print(f'{organ}', self.ad_mean[organ])

        self.df['mean_ad[Gy]']= self.df['organ'].map(self.ad_mean)


            
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
    def calculate_bed(self):
        bed_df = self.df[self.df['organ'].isin(list(self.radiobiology_dic.keys()))] # only organs that we know the radiobiology parameters
        organs = np.array(bed_df['organ'].unique())
        bed = {}
        for organ in organs:
            t_repair = self.radiobiology_dic[organ]['t_repair']
            alpha_beta = self.radiobiology_dic[organ]['alpha_beta']
            t_eff = (np.log(2) / bed_df.loc[bed_df['organ'] == organ]['lamda_eff_1/s'].values[0]) / 3600
            abs_dose = bed_df.loc[bed_df['organ'] == organ]['mean_ad[Gy]'].values[0]
            bed[organ] = abs_dose + 1/alpha_beta * t_repair/(t_repair + t_eff) * abs_dose**2
            print(f'{organ}', bed[organ])
            
        self.df['bed[Gy]'] = self.df['organ'].map(bed)


