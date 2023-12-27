from os import path
from pathlib import Path

import numpy as np
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
    def __init__(self, df):
        self.df = df

        
    def phantom_data(self):
        lesion_df = self.df[self.df['organ'].str.contains('Lesion', case=False, na=False)]
        
        phrases_to_exclude = ['Femur', 'Humerus', 'Reference', 'TotalTumorBurden', 'Kidney_L_m', 'Kidney_R_m', 'Lesion']
        phrases_to_exclude.extend([f'L{i}' for i in range(10)])  # L(any number)
        # Replace "Kidney_R_a" and "Kidney_L_a" with "Kidneys" in the 'organ' column
        self.df['organ'] = self.df['organ'].replace(['Kidney_R_a', 'Kidney_L_a'], 'Kidneys')
        # Group by 'organ' and aggregate values (e.g., sum)
        self.df = self.df.groupby('organ').agg({'tiac_h': 'sum'}).reset_index()
        
        self.df['organ'] = self.df['organ'].replace(['ParotidglandL', 'ParotidglandR', 'SubmandibularglandL', 'SubmandibularglandR'], 'Salivary Glands')
        # Group by 'organ' and aggregate values (e.g., sum)
        self.df = self.df.groupby('organ').agg({'tiac_h': 'sum'}).reset_index()
        self.df['organ'] = self.df['organ'].replace('Bladder_Experimental', 'Urinary Bladder Contents')



        self.df = self.df[~self.df['organ'].str.contains('|'.join(phrases_to_exclude), case=False, na=False)]
        print(self.df)

        self.organlist = self.df['organ'].unique()
        this_dir=path.dirname(__file__)
        PHANTOM_PATH = path.join(this_dir,'phantomdata')
        self.phantom_mass = pd.read_csv(path.join(PHANTOM_PATH,'human_phantom_masses.csv'))
        print(np.array(self.phantom_mass['Organ']))
        not_inphantom=[]
        for org in self.organlist: 
            if org not in np.array(self.phantom_mass['Organ']):            
                not_inphantom.append(org)
        print('These organs from the biodi are not modelled in the phantom\n{}'.format(not_inphantom))
            
    def template_data(self):
        TEMPLATE_PATH = path.join(this_dir,"olindaTemplates")