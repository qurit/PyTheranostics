from os import path
import pandas as pd
import numpy as np


class Olinda:
    def __init__(self, df, isotope):
        self.df = df
        self.isotope = isotope

        
    def phantom_data(self):
        print(self.df)
        lesion_df = self.df[self.df['organ'].str.contains('Lesion', case=False, na=False)]
        
        ## Prepare df for Olinda 
        phrases_to_exclude = ['Femur', 'Humerus', 'Reference', 'TotalTumorBurden', 'Kidney_L_m', 'Kidney_R_m', 'Lesion', 'WBCT']
        phrases_to_exclude.extend([f'L{i}' for i in range(10)])  # L(any number)
        self.df = self.df[~self.df['organ'].str.contains('|'.join(phrases_to_exclude), case=False, na=False)]
        
        
        # group kidneys 
        self.df['organ'] = self.df['organ'].replace(['Kidney_R_a', 'Kidney_L_a'], 'Kidneys')
        self.df = self.df.groupby('organ').agg({'tiac_h': 'sum'}).reset_index()
        
        # group salivary glands
        self.df['organ'] = self.df['organ'].replace(['ParotidglandL', 'ParotidglandR', 'SubmandibularglandL', 'SubmandibularglandR'], 'Salivary Glands')
        self.df = self.df.groupby('organ').agg({'tiac_h': 'sum'}).reset_index()
        
        # renaming
        self.df['organ'] = self.df['organ'].replace('Bladder_Experimental', 'Urinary Bladder Contents')
        self.df['organ'] = self.df['organ'].replace('ROB', 'Total Body')
        self.df['organ'] = self.df['organ'].replace('Skeleton', 'Cortical Bone')
        

        blankIndex=[''] * len(self.df)
        self.df.index=blankIndex
        print(self.df)


        self.organlist = self.df['organ'].unique()
        this_dir=path.dirname(__file__)
        PHANTOM_PATH = path.join(this_dir,'phantomdata')
        self.phantom_mass = pd.read_csv(path.join(PHANTOM_PATH,'human_phantom_masses.csv'))
        print(np.array(self.phantom_mass['Organ']))
        self.not_inphantom=[]
        for org in self.organlist: 
            if org not in np.array(self.phantom_mass['Organ']):            
                self.not_inphantom.append(org)
        print('These organs are not modelled in the phantom\n{}'.format(self.not_inphantom))
            
    def create_case_file(self):-
        this_dir=path.dirname(__file__)
        TEMPLATE_PATH = path.join(this_dir,"olindaTemplates")
        template=pd.read_csv(path.join(TEMPLATE_PATH,'adult_male.cas'))
        template.columns=['Data']
        
        #modify the isotope in the template
        ind=template[template['Data']=='[BEGIN NUCLIDES]'].index
        template.loc[ind[0]+1,'Data']=self.isotope + '|'
        
        print(self.df.drop(set(self.not_inphantom).intersection(set(self.df.index))).index)
        #change the kinetics for each input organ
        for org in self.df.drop(set(self.not_inphantom).intersection(set(self.df.index))).index:  #ignore the tumor here
            temporg=org
            uptakepos=2
            

                
            ind=template[template['Data'].str.contains(temporg)].index
            sourceorgan=template.iloc[ind[0]].str.split('|')[0][0]
            massorgan=template.iloc[ind[0]].str.split('|')[0][1]
            kineticdata=self.df.loc[org]['tiac_h']
            
            if np.isnan(kineticdata):
                kineticdata=0

            template.iloc[ind[0]]=sourceorgan+'|'+massorgan+'|'+'{:7f}'.format(kineticdata)
            template.iloc[ind[uptakepos]]=sourceorgan+'|'+'{:7f}'.format(kineticdata)

        
        print(template)
        #####