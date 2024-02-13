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


class Image:
#    def __init__(self, df, patient_id, cycle, image, roi_masks_resampled, organlist):
#        self.df = df
#        self.patient_id = patient_id
#        self.cycle = int(cycle)
#        self.image = image 
#        self.roi_masks_resampled = roi_masks_resampled
#        self.organlist = organlist
#        with open(RADIOBIOLOGY_DATA_FILE) as f:
#            self.radiobiology_dic = json.load(f)
    
    def __init__(self, df, patient_id, cycle, image, roi_masks_resampled):
        self.df = df
        self.patient_id = patient_id
        self.cycle = int(cycle)
        self.image = image
        self.roi_masks_resampled = roi_masks_resampled
        self.organlist = self.roi_masks_resampled.keys()
        with open(RADIOBIOLOGY_DATA_FILE) as f:
            self.radiobiology_dic = json.load(f)
            
    def SPECT_image_array(self, SPECT, scalefactor, xspacing, yspacing, zspacing):
        SPECT_image = SPECT.pixel_array
        SPECT_image = np.transpose(SPECT_image, (1, 2, 0)) # from (233,128,128) to (128,128,233)
        SPECT_image.astype(np.float64)
        scalefactor = SPECT.RealWorldValueMappingSequence[0].RealWorldValueSlope
        SPECT_image = SPECT_image * scalefactor # in Bq/ml
        SPECTMBq = SPECT_image * (SPECT.PixelSpacing[0]*SPECT.PixelSpacing[1]*SPECT.SpacingBetweenSlices/1000) / 1E6 # in MBq
        self.image = SPECTMBq
        return SPECTMBq
    
    def image_visualisation(self, image):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
        axs[0].imshow(image[:, :, 50])
        axs[0].set_title('Slice at index 50')
        axs[1].imshow(image[69, :, :].T)
        axs[1].set_title('Slice at index 100')
        axs[2].imshow(image[:,60, :])
        axs[2].set_title('Slice at index 47')

        plt.tight_layout()
        plt.show()
        

    def show_mean_statistics(self, output):
        
        self.ad_mean = {}
        for organ in self.organlist:
            mask = self.roi_masks_resampled[organ]
            self.ad_mean[organ] = self.image[mask].mean() 
            print(f'{organ}', self.ad_mean[organ])

        self.df[output]= self.df['Contour'].map(self.ad_mean)
            
    def show_max_statistics(self):
        for organ in self.organlist:
            mask = self.roi_masks_resampled[organ]
            x = self.image[mask].max()
            print(f'{organ}', x)
            
    def add(self, output):
        self.sum = {}
        for organ in self.organlist:
            mask = self.roi_masks_resampled[organ]
            self.sum[organ] = self.image[mask].sum()
            print(f'{organ}', self.sum[organ])
            
        self.df[output]= self.df['Contour'].map(self.sum)
        
    def voxels_and_volume(self, output1, output2, voxel_volume):
        self.no_voxels = {}
        self.volume = {}
        for organ in self.organlist:
            mask = self.roi_masks_resampled[organ]
            self.no_voxels[organ] = np.sum(self.image[mask] != 0)
            self.volume[organ] = self.no_voxels[organ]  * voxel_volume

        self.df[output1]= self.df['Contour'].map(self.no_voxels)
        self.df[output2]= self.df['Contour'].map(self.volume)
                        
    def dose_volume_histogram(self):
        doseimage = self.image.astype(float)
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


