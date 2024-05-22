from doodle.dosimetry.BaseDosimetry import BaseDosimetry
from typing import Any, Dict, Optional
from doodle.ImagingDS.LongStudy import LongitudinalStudy

import datetime
import numpy
import pandas
import re
from os import path, makedirs

class OrganSDosimetry(BaseDosimetry):
    """Organ S Value Dosimetry Class. Takes Nuclear Medicine Data to perform Organ-level 
    patient-specific dosimetry by computing organ time-integrated activity curves and
    leveraging organ level S-values. Currently, it supports external software: Olinda/EXM. """ # TODO: add support for MirdCalc and IDAC
    def __init__(self,
                 config: Dict[str, Any],
                 nm_data: LongitudinalStudy, 
                 ct_data: Optional[LongitudinalStudy],
                 clinical_data: Optional[pandas.DataFrame] = None
                 ) -> None:
        super().__init__(config, nm_data, ct_data, clinical_data)
        self.check_mandatory_fields_organ()
        
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
        
    def check_mandatory_fields_organ(self) -> None:

        if "Organ" not in self.config["Level"]:
            print("Verify the level on which dosimetry should be performed.")
        
        return None
        
    def prepare_data(self) -> None:
        """Creates .cas file that can be exported to Olinda/EXM."""
        self.results_olinda = self.results[['Volume_CT_mL', 'TIA_h']].copy()
        self.results_olinda = self.results_olinda.drop(["WholeBody"])
        
        # Average Volume over time points.
        self.results_olinda['Volume_CT_mL'] = self.results_olinda['Volume_CT_mL'].apply(lambda x: numpy.mean(x))  
        
        # Combine Kidneys.
        kidneys = ['Kidney_Left', 'Kidney_Right']
        self.results_olinda.loc['Kidneys'] = self.results_olinda.loc[kidneys].sum()  
        self.results_olinda = self.results_olinda.drop(kidneys)
        
        # Combine Salivary Glands.
        sal_glands = ['ParotidGland_Left', 'ParotidGland_Right', 'SubmandibularGland_Left', 'SubmandibularGland_Right']
        self.results_olinda.loc['Salivary Glands'] = self.results_olinda.loc[sal_glands].sum()
        self.results_olinda = self.results_olinda.drop(sal_glands)

        # Rename
        self.results_olinda = self.results_olinda.rename(index={'Bladder': 'Urinary Bladder Contents', 
                                                                'Skeleton': 'Cortical Bone', 
                                                                'RemainderOfBody': 'Total Body',
                                                                'BoneMarrow': 'Red Marrow'}) # TODO Cortical Bone vs Trabercular Bone
        # BoneMarrow volume.
        self.results_olinda.loc['Red Marrow']['Volume_CT_mL'] = 1170 # TODO volume hardcoded, think about alternatives

        return None
    
    def create_output_file(self, 
                           dirname: str, 
                           savefile: bool = False
                           ) -> None:  
        if self.config["OutputFormat"] == "Olinda":  
            self.create_Olinda_file(dirname, savefile)
        else:
            print("In the current version, we only support Olinda as external organ S-value software.") # TODO: other software case files
                
    def create_Olinda_file(self, 
                           dirname: str, 
                           savefile: bool = False
                           ) -> None:
        """Creates .cas file that can be exported to Olinda/EXM."""
        this_dir=path.dirname(__file__)
        TEMPLATE_PATH = path.join(this_dir,"olindaTemplates")
        
        if self.config["Gender"] == "Male":
            template=pandas.read_csv(path.join(TEMPLATE_PATH,'adult_male.cas'))
        elif self.config["Gender"] == "Female":
            template=pandas.read_csv(path.join(TEMPLATE_PATH,'adult_female.cas'))
        else:
            print('Ensure that you correctly wrote patient gender in config file. Olinda supports: Male and Female.')
        
        template.columns=['Data']
        match = re.match(r"([a-zA-Z]+)([0-9]+)", self.config["Radionuclide"])
        letters, numbers = match.groups()
        formatted_radionuclide = f"{letters}-{numbers}"
        
        ind=template[template['Data']=='[BEGIN NUCLIDES]'].index
        template.loc[ind[0]+1,'Data']= formatted_radionuclide + '|'
             
        for organ in self.results_olinda.index:
            indices = template[template['Data'].str.contains(organ)].index

            source_organ = template.iloc[indices[0]].str.split('|')[0][0]
            mass_phantom = template.iloc[indices[0]].str.split('|')[0][1]
            kinetic_data = self.results_olinda.loc[organ]['TIA_h']
            mass_data = round(self.results_olinda.loc[organ]['Volume_CT_mL'], 1)

            # Update the template DataFrame
            template.iloc[indices[0]] = f"{source_organ}|{mass_phantom}|{'{:7f}'.format(kinetic_data)}"

            if len(indices) == 2:
                template.iloc[indices[1]] = f"{source_organ}|{'{:7f}'.format(kinetic_data)}"
            elif len(indices) == 3:
                template.iloc[indices[1]] = f"{source_organ}|{mass_data}"
                template.iloc[indices[2]] = f"{source_organ}|{'{:7f}'.format(kinetic_data)}"
            else:
                print('Double-check where the organ appears in the template.')

        template = template.replace('TARGET_ORGAN_MASSES_ARE_FROM_USER_INPUT|FALSE', 'TARGET_ORGAN_MASSES_ARE_FROM_USER_INPUT|TRUE')

        template.columns=['Saved on ' + datetime.datetime.now().strftime("%m.%d.%Y") +' at ' + datetime.datetime.now().strftime('%H:%M:%S')]

        if savefile==True:
            if not path.exists(dirname):
                makedirs(dirname)

            template.to_csv(str(dirname) + '/' + f"{self.config['PatientID']}.cas", index=False)

    def read_results(self, 
                     olinda_results_path: str
                     ) -> None:  
        if self.config["OutputFormat"] == "Olinda":  
            self.read_Olinda_results(olinda_results_path)
            
    def read_Olinda_results(self, 
                            olinda_results_path: str
                            ) -> None:
        """Reads .txt results file from Olinda.""" 
        if not olinda_results_path.endswith(".txt"):
            print('Please export result from Olinda in .txt file.') ## TODO: Add .csv extension results
            return
    
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
                
        if not data:
            print('No relevant data found in the file.')
            return
    
        df_ad = pandas.DataFrame(data[1:], columns=data[0])
        df_ad = df_ad.set_index('Target Organ')
        df_ad = df_ad.dropna(axis=0)        
        df_ad['Total'] = pandas.to_numeric(df_ad['Total'], errors='coerce')
        df_ad['Total'].fillna(0, inplace=True)
        df_ad['AD[Gy]'] = df_ad['Total'] * float(self.config['InjectedActivity']) / 1000
        df_ad = df_ad.rename(columns={"Total":"AD[Gy/GBq]"})
        self.df_ad = df_ad
        

    def compute_dose(self):
        self.compute_tia()
        self.prepare_data()
        
        
        
