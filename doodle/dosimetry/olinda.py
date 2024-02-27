from os import path
from pathlib import Path
import datetime
import numpy
import pandas


def load_s_values(gender: str, radionuclide: str) -> pandas.DataFrame:
    """Load S-values Dataframes"""
    path_to_sv = Path(f"./phantomdata/{radionuclide}-{gender}-Svalues.csv")
    if not path_to_sv.exists():
        raise FileExistsError(f"S-values for {gender}, {radionuclide} not found. Please make sure"
                              " gender is ['Male', 'Female'] and radionuclide SymbolMass e.g., Lu177")

    s_df = pandas.read_csv(path_to_sv)
    s_df.set_index(keys=["Target"], drop=True, inplace=True)
    s_df = s_df.drop(labels=["Target"], axis=1)

    return s_df


def load_phantom_mass(gender: str, organ: str) -> float:
    """Load the mass of organs in the standar ICRP Male/Female phantom"""
    masses = pandas.read_csv("./phantomdata/human_phantom_massess.csv")

    if organ not in masses["Organ"].to_list():
        raise ValueError(f"Organ {organ} not found in phantom data.")

    return masses.loc[masses["Organ"] == organ].iloc[0][gender]


class Olinda:
    def __init__(self, config, tiac_for_masks):
        self.config = config
        self.tiac_for_masks = tiac_for_masks

        
    def phantom_data(self):## preparaing data
        lesion_df = self.tiac_for_masks[self.tiac_for_masks.index.str.contains('Lesion', case=False, na=False)]
        
        self.tiac_for_masks['Volume_CT_mL'] = self.tiac_for_masks['Volume_CT_mL'].apply(lambda x: numpy.mean(x))
        phrases_to_exclude = ['Cavity', 'Femur', 'Humerus', 'Reference', 'TotalTumorBurden', 'Kidney_L_a', 'Kidney_R_a', 'Lesion', 'LN_Iliac']
        phrases_to_exclude.extend([f'L{i}' for i in range(10)])  # L(any number)
        self.tiac_for_masks = self.tiac_for_masks[~self.tiac_for_masks.index.str.contains('|'.join(phrases_to_exclude), case=False, na=False)]
        self.tiac_for_masks = self.tiac_for_masks[['Volume_CT_mL', 'TIAC_h']]
        columns_to_sum = ['TIAC_h', 'Volume_CT_mL']

        self.tiac_for_masks.loc['Kidneys'] = self.tiac_for_masks.loc[['Kidney_R_m', 'Kidney_L_m']].sum()
        self.tiac_for_masks.loc['Kidneys', columns_to_sum] = self.tiac_for_masks.loc[['Kidney_R_m', 'Kidney_L_m'], columns_to_sum].sum()
        self.tiac_for_masks = self.tiac_for_masks.drop(['Kidney_R_m', 'Kidney_L_m'])

        self.tiac_for_masks.loc['Salivary Glands'] = self.tiac_for_masks.loc[['ParotidglandL', 'ParotidglandR', 'SubmandibularglandL', 'SubmandibularglandR']].sum()
        self.tiac_for_masks.loc['Salivary Glands', columns_to_sum] = self.tiac_for_masks.loc[['ParotidglandL', 'ParotidglandR', 'SubmandibularglandL', 'SubmandibularglandR'], columns_to_sum].sum()
        self.tiac_for_masks = self.tiac_for_masks.drop(['ParotidglandL', 'ParotidglandR', 'SubmandibularglandL', 'SubmandibularglandR'])

        organs = self.tiac_for_masks.index[self.tiac_for_masks.index != 'WBCT']
        self.tiac_for_masks.loc['WBCT', columns_to_sum] = self.tiac_for_masks.loc['WBCT', columns_to_sum] - self.tiac_for_masks.loc[organs, columns_to_sum].sum()

        self.tiac_for_masks = self.tiac_for_masks.rename(index={'Bladder_Experimental': 'Urinary Bladder Contents', 'Skeleton': 'Cortical Bone', 'WBCT': 'Total Body'}) # TODO Cortical Bone vs Trabercular Bone

            
    def create_case_file(self, dirname, savefile=False):
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
        
        for org in self.tiac_for_masks.index:  #ignore the tumor here
            temporg = org
            ind=template[template['Data'].str.contains(temporg)].index
            sourceorgan=template.iloc[ind[0]].str.split('|')[0][0]
            massorgan=template.iloc[ind[0]].str.split('|')[0][1]
            kineticdata=self.tiac_for_masks.loc[org]['TIAC_h']
            massdata=round(self.tiac_for_masks.loc[org]['Volume_CT_mL'], 1)
            template.iloc[ind[0]]=sourceorgan+'|'+str(massorgan)+'|'+'{:7f}'.format(kineticdata)         
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

