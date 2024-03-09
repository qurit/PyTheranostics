from doodle.dosimetry.BaseDosimetry import BaseDosimetry
from typing import Any, Dict, List, Optional
from doodle.ImagingDS.LongStudy import LongitudinalStudy

import datetime
import numpy
import pandas
from os import path

class OrganSDosimetry(BaseDosimetry):
    """Organ S Value Dosimetry Class. Takes Nuclear Medicine Data to perform Organ-level 
    patient-specific dosimetry by computing organ time-integrated activity curves and
    leveraging organ level S-values. It supports external software: Olinda/EXM. """
    def __init__(self,
                 config: Dict[str, Any],
                 nm_data: LongitudinalStudy, 
                 ct_data: Optional[LongitudinalStudy],
                 clinical_data: Optional[pandas.DataFrame] = None
                 ) -> None:
        super().__init__(config, nm_data, ct_data, clinical_data)
        
        """Inputs:
            config: Configuration parameters for dosimetry calculations, a Dict.
                    Note: defined VOIs should have the same naming convention as source organs in Olinda.
                    We included method prepare_df() that combines kidneys and salivary glands into one VOI.
            nm_data: longitudinal, quantitative, nuclear-medicine imaging data, type LongitudinalStudy.
                     Note: voxel values should be in units of Bq/mL.
            ct_data: longitudinal CT imaging data, type LongitudinalStudy,
                     Note: voxel values should be in HU units.
            clinical_data: clinical data such as blood sampling, an optional pandas DataFrame. 
                     Note: blood counting should be in units of Bq/mL.
        """    

    def prepare_data(self) -> None:
        self.results_olinda = self.results[['Volume_CT_mL', 'TIAC_h']].copy()
        self.results_olinda['Volume_CT_mL'] = self.results_olinda['Volume_CT_mL'].apply(lambda x: numpy.mean(x))
        self.results_olinda = self.results_olinda[['Volume_CT_mL', 'TIAC_h']]
        
        self.results_olinda.loc['Kidneys'] = self.results_olinda.loc[['Kidney_R_m', 'Kidney_L_m']].sum()
        self.results_olinda = self.results_olinda.drop(['Kidney_R_m', 'Kidney_L_m'])
        
        self.results_olinda.loc['Salivary Glands'] = self.results_olinda.loc[['ParotidglandL', 'ParotidglandR', 'SubmandibularglandL', 'SubmandibularglandR']].sum()
        self.results_olinda = self.results_olinda.drop(['ParotidglandL', 'ParotidglandR', 'SubmandibularglandL', 'SubmandibularglandR'])

        organs = self.results_olinda.index[self.results_olinda.index != 'WBCT']
        self.results_olinda.loc['WBCT'] = self.results_olinda.loc['WBCT'] - numpy.sum(self.results_olinda.loc[organs])

        self.results_olinda = self.results_olinda.rename(index={'Bladder_Experimental': 'Urinary Bladder Contents', 
                                                  'Skeleton': 'Cortical Bone', 
                                                  'WBCT': 'Total Body',
                                                  'BoneMarrow': 'Red Marrow'}) # TODO Cortical Bone vs Trabercular Bone
        
        self.results_olinda.loc['Red Marrow']['Volume_CT_mL'] = 1170 # TODO volume hardcoded, think about alternatives
            
    def create_Olinda_file(self, dirname, savefile=False):
        this_dir=path.dirname(__file__)
        TEMPLATE_PATH = path.join(this_dir,"olindaTemplates")
        template=pandas.read_csv(path.join(TEMPLATE_PATH,'adult_male.cas'))
        template.columns=['Data']
        if self.config["Radionuclide"] == 'Lu177':
            isotope = 'Lu-177'
        else:
            print('Add isotope')
        ind=template[template['Data']=='[BEGIN NUCLIDES]'].index
        template.loc[ind[0]+1,'Data']= isotope + '|'
        
        for org in self.results_olinda.index:  
            temporg = org
            ind=template[template['Data'].str.contains(temporg)].index
            sourceorgan=template.iloc[ind[0]].str.split('|')[0][0]
            print(sourceorgan)
            massorgan=template.iloc[ind[0]].str.split('|')[0][1]
            kineticdata=self.results_olinda.loc[org]['TIAC_h']
            massdata=round(self.results_olinda.loc[org]['Volume_CT_mL'], 1)
            template.iloc[ind[0]]=sourceorgan+'|'+str(massorgan)+'|'+'{:7f}'.format(kineticdata)  
            print(len(ind))       
            if len(ind) == 2:
                template.iloc[ind[1]]=sourceorgan+'|'+'{:7f}'.format(kineticdata)            
            elif len(ind) == 3:
                template.iloc[ind[1]]=sourceorgan+'|'+str(massdata)
                template.iloc[ind[2]]=sourceorgan+'|'+'{:7f}'.format(kineticdata)
            else:
                print('Double check where the organ appears in the template')

        template = template.replace('TARGET_ORGAN_MASSES_ARE_FROM_USER_INPUT|FALSE', 'TARGET_ORGAN_MASSES_ARE_FROM_USER_INPUT|TRUE')

        now = datetime.datetime.now()
        template.columns=['Saved on ' + now.strftime("%m.%d.%Y") +' at ' + now.strftime('%H:%M:%S')]

        if savefile==True:
            if not path.exists(dirname):
                makedirs(dirname)

            template.to_csv(str(dirname) + '/' + f"{self.config['PatientID']}.cas", index=False)

    def read_Olinda_results(self, 
                            olinda_results_path: str
                            ) -> None:
        data = []
        with open(olinda_results_path, 'r') as file:
        
            extract_lines = False
            for line in file:
                stripped_line = line.strip()
                if stripped_line.startswith("Target Organ,Alpha,Beta,Gamma,Total,ICRP-103 ED"):
                    extract_lines = True
                elif stripped_line.startswith("Target Organ Name,Mass [g],"):
                    extract_lines = False
                if extract_lines:
                    if stripped_line.startswith('Effective Dose'):
                        stripped_line = stripped_line.replace('Effective Dose', 'Effective Dose,,,,')
                    stripped_line = stripped_line.rstrip(',')
                    data.append(stripped_line.split(','))
        
        df_ad = pandas.DataFrame(data[1:], columns=data[0])
        df_ad = df_ad.set_index('Target Organ')
        df_ad = df_ad.dropna(axis=0)        
        df_ad['Total'] = pandas.to_numeric(df_ad['Total'], errors='coerce')
        df_ad['Total'].fillna(0, inplace=True)
        df_ad['AD[Gy]'] = df_ad['Total'] * float(self.config['InjectedActivity']) / 1000
        self.df_ad = df_ad
        
    def compute_dose(self):
        self.compute_tiac()
        self.prepare_data()
        # TODO: finish-up the pipeline up to the Olinda File Export.
        
        
        
