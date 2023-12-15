from __future__ import print_function
import sys
import os
import six
from radiomics import featureextractor
import radiomics
import SimpleITK as sitk
import pandas as pd

class Radiomics:
    def __init__(self, imagemodality, patient_id, cycle, image, mask, organslist):
        self.imagemodality = imagemodality
        self.patient_id = patient_id
        self.cycle = cycle
        self.image = image
        self.mask = mask
        self.organslist = organslist

    def prepareimages(self):
        img = sitk.GetImageFromArray(self.image)

        sitk.WriteImage(img, f"/mnt/y/Sara/PR21_dosimetry/{self.patient_id}/cycle0{self.cycle}/radiomics/{self.imagemodality}.nrrd")
        for organ in self.organslist:
            self.mask[organ] = self.mask[organ].astype(int)
            img = sitk.GetImageFromArray(self.mask[organ])
            sitk.WriteImage(img, f"/mnt/y/Sara/PR21_dosimetry/{self.patient_id}/cycle0{self.cycle}/radiomics/{organ}.nrrd")
    
        
    def featureextractor(self):
        paramPath = os.path.join('..', 'data', 'Params.yaml')
        #print('Parameter file, absolute path:', os.path.abspath(paramPath))
        extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
        #print('Extraction parameters:\n\t', extractor.settings)
        #print('Enabled filters:\n\t', extractor.enabledImagetypes)
        #print('Enabled features:\n\t', extractor.enabledFeatures)
        radiomics_list = []
        for organ in self.organslist:
            imagepath = f"/mnt/y/Sara/PR21_dosimetry/{self.patient_id}/cycle0{self.cycle}/dosiomics/{self.imagemodality}.nrrd"
            maskpath = f"/mnt/y/Sara/PR21_dosimetry/{self.patient_id}/cycle0{self.cycle}/dosiomics/{organ}.nrrd"
            result = extractor.execute(imagepath, maskpath)
            #print('Result type:', type(result))  # result is returned in a Python ordered dictionary)
            #print('')
            #print('Calculated features')
            data = {'organ': organ}
            for key, value in six.iteritems(result):
                #print('\t', key, ':', value)
                data[key] = value
                
            radiomics_list.append(data)
            
        radiomics_df = pd.DataFrame(radiomics_list)
        radiomics_df.to_csv(f"/mnt/y/Sara/PR21_dosimetry/output/{self.patient_id}_cycle0{self.cycle}_radiomics_{self.imagemodality}_output.csv")
        
        return radiomics_df

