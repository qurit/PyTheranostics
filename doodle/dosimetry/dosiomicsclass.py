from __future__ import print_function
import sys
import os
import six
from radiomics import featureextractor
import radiomics
import SimpleITK as sitk
import pandas as pd

class Dosiomics:
    def __init__(self, patient_id, cycle, image, mask, organslist):
        self.patient_id = patient_id
        self.cycle = cycle
        self.image = image
        self.mask = mask
        self.organslist = organslist

    def prepareimages(self):
        dosemap = sitk.GetImageFromArray(self.image)
        sitk.WriteImage(dosemap, f"/mnt/y/Sara/PR21_dosimetry/{self.patient_id}/cycle0{self.cycle}/dosiomics/dosemap.nrrd")
        for organ in self.organslist:
            self.mask[organ] = self.mask[organ].astype(int)
            img = sitk.GetImageFromArray(self.mask[organ])
            sitk.WriteImage(img, f"/mnt/y/Sara/PR21_dosimetry/{self.patient_id}/cycle0{self.cycle}/dosiomics/{organ}.nrrd")
    
        
    def featureextractor(self):
        paramPath = os.path.join('..', 'data', 'Params.yaml')
        #print('Parameter file, absolute path:', os.path.abspath(paramPath))
        extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
        #print('Extraction parameters:\n\t', extractor.settings)
        #print('Enabled filters:\n\t', extractor.enabledImagetypes)
        #print('Enabled features:\n\t', extractor.enabledFeatures)
        dosiomics_list = []
        for organ in self.organslist:
            imagepath = f"/mnt/y/Sara/PR21_dosimetry/{self.patient_id}/cycle0{self.cycle}/dosiomics/dosemap.nrrd"
            maskpath = f"/mnt/y/Sara/PR21_dosimetry/{self.patient_id}/cycle0{self.cycle}/dosiomics/{organ}.nrrd"
            result = extractor.execute(imagepath, maskpath)
            #print('Result type:', type(result))  # result is returned in a Python ordered dictionary)
            #print('')
            #print('Calculated features')
            data = {'organ': organ}
            for key, value in six.iteritems(result):
                #print('\t', key, ':', value)
                data[key] = value
                
            dosiomics_list.append(data)
            
        dosiomics_df = pd.DataFrame(dosiomics_list)
        dosiomics_df.to_csv(f"/mnt/y/Sara/PR21_dosimetry/output/{self.patient_id}_cycle0{self.cycle}_dosiomics_output.csv")
        
        return dosiomics_df

