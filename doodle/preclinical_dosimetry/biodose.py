import pandas as pd
import numpy as np

from numpy import exp, log
from os import path,makedirs
import datetime
from copy import deepcopy
from scipy import integrate


import matplotlib.pylab as plt
#from doodle.plots.plots import monoexp_fit_plots, biexp_fit_plots
from doodle.fits.fits import exponential_fit_lmfit, calculate_r_squared, get_exponential, monoexp_fun, biexp_fun, biexp_fun_uptake, triexp_fun
from typing import Any, Callable, Optional, Tuple, Dict
from doodle.plots.plots import plot_tac_residuals
from scipy import integrate
from scipy.optimize import curve_fit
import lmfit
from lmfit import Model



this_dir=path.dirname(__file__)
TEMPLATE_PATH = path.join(this_dir,"olindaTemplates")
PHANTOM_PATH = path.join(this_dir,'phantomdata') # These variables use only lowercases so "PhantomData" is in lowercase


class BioDose():
    '''
    This is a class for the analysis of biodistribution data in preclinical experiments
    '''

    def __init__(self, isotope, half_life, phantom, mouse_mass, sex, uptake=None, timepoints=None):
        'half_life in hours'
        if uptake is None:
            uptake = []
        if timepoints is None:
            timepoints = []

        self.isotope = isotope
        self.half_life = half_life
        self.phantom = phantom
        self.mouse_mass = mouse_mass
        self.uptake = uptake
        self.sex = sex
        self.t = timepoints
        self.biodi = None
        self.wb_m = int(self.mouse_mass[:-1]) 

    def rename_organ(self,oldname,newname):
        ind_list=self.biodi.index.tolist()
        ind_pos = ind_list.index(oldname)
        ind_list[ind_pos] = newname
        self.biodi.index = ind_list
                
    def read_biodi(self, biodi_file):
        '''This method reads a biodi file and sets the data as a pandas dataframe in self.biodi
        '''
        print('Reading biodistribution information from the file: {}'.format(biodi_file))
        biodi = pd.read_csv(biodi_file)
        
        # Generalize the biodi organ names
        biodi['Organ'] = biodi['Organ'].str.title()
        biodi['Organ'] = biodi['Organ'].replace('Adrenal Glands','Adrenals')
        biodi['Organ'] = biodi['Organ'].replace('Adrenal','Adrenals')
        biodi['Organ'] = biodi['Organ'].replace('Gall Bladder','Gallbladder')
        biodi['Organ'] = biodi['Organ'].replace('Bone Marrow','Red Marrow')
        biodi['Organ'] = biodi['Organ'].replace('Muscles','Muscle')
        biodi['Organ'] = biodi['Organ'].replace('Urine','Bladder')
        biodi['Organ'] = biodi['Organ'].replace('Submandibular Glands','Salivary Glands')
        biodi['Organ'] = biodi['Organ'].replace('Submandibular Gland','Salivary Glands')
        biodi['Organ'] = biodi['Organ'].replace('Seminal Glands','Seminals')
        biodi['Organ'] = biodi['Organ'].replace('Seminal','Seminals')
        biodi['Organ'] = biodi['Organ'].replace('Seminal Vesicles','Seminals')
        biodi['Organ'] = biodi['Organ'].replace('Testis','Testes')
        biodi['Organ'] = biodi['Organ'].replace('Lung','Lungs')
        biodi['Organ'] = biodi['Organ'].replace('Large Intestines','Large Intestine')
        biodi['Organ'] = biodi['Organ'].replace('Small Intestines','Small Intestine')
        biodi['Organ'] = biodi['Organ'].replace('Bone','Skeleton')
        biodi['Organ'] = biodi['Organ'].replace('Kidney','Kidneys')
        biodi['Organ'] = biodi['Organ'].replace('Tumour','Tumor')

        biodi.set_index('Organ', inplace=True)
        biodi = pd.concat([biodi.iloc[:, :len(self.t)], biodi.iloc[:, len(self.t):]], axis=1, keys=['%ID/g', 'sigma'])
        biodi.sort_index(inplace=True)

        self.raw_biodi = biodi
        self.biodi = biodi.copy()
        print('Raw biodi (all data at injection time) available in self.raw_biodi')

        decay_factor = np.round(exp(-log(2)/self.half_life*self.t), 5)


        print("Decay factors for this isotope at given times are\n", decay_factor)
        self.biodi['%ID/g'] = np.round(biodi['%ID/g']*decay_factor,5)
        self.biodi['sigma'] = np.round(biodi['sigma']*decay_factor,5)
        print('Decayed biodistribution stored in self.biodi')
        
    def initialize_results_df(self):
        '''
        This method initializes the results dataframe with the organ list and columns for the different fits.
        '''
        columns = ['Mono-Exponential', 'Bi-Exponential', 'Bi-Exponential_uptake', 
                   'Tri-Exponential', 'Perctage_diff - mono vs bi washout', 
                   'Perctage_diff - bi vs tri uptake washout']
        if not hasattr(self, 'area') or self.area is None:
            self.area = pd.DataFrame(index=self.biodi.index, columns=columns)
        if not hasattr(self, 'fit_results') or self.fit_results is None:
            self.fit_results = {}


    def update_fit_results(self, org, mono_params=None, bi_params=None, uptake_params=None,
                           area_mono=None, area_bi=None, area_uptake=None):
        '''
        This method updates the fit results for a given organ.
        '''
        area_mono_val = area_mono[0] if isinstance(area_mono, (tuple, list)) else area_mono
        area_bi_val = area_bi[0] if isinstance(area_bi, (tuple, list)) else area_bi
        
        if mono_params and area_mono:
            self.area.loc[org, 'Mono-Exponential'] = area_mono_val
        
        if bi_params and area_bi:
            self.area.loc[org, 'Bi-Exponential'] = area_bi_val
            if area_mono and area_mono_val != 0:
                self.area.loc[org, 'Perctage_diff - mono vs bi washout'] = (
                    abs(area_mono_val - area_bi_val) / area_mono_val * 100
                )

        if uptake_params and area_uptake:
            self.area.loc[org, 'Bi-Exponential_uptake'] = area_uptake[0]

        self.fit_results[org] = [
            (mono_params['A1'], mono_params['A2']) if mono_params else None,
            area_mono,
            (bi_params['A1'], bi_params['A2'], bi_params['B1'], bi_params['B2']) if bi_params else
            (uptake_params['A1'], uptake_params['A2'], uptake_params['B1'], uptake_params['B2']) if uptake_params else None,
            area_bi if bi_params else area_uptake
        ]

        
    def curve_fits(self, organlist=None, uptake=False, maxev=100000, monoguess=(1,0.1),  
               uptakeguess=(1,1,-1,1), ignore_weights=False, append_zero=True, tps_to_skip_fit=0):
        ''' 
        This method fits the curves using lmfit and stores the results in self.fit_results.  
        The results can be seen in self.area organized in a pandas dataframe.
        '''
        decayconst = log(2)/self.half_life
    
        if organlist is None:
            organlist = self.biodi.index

        # Initialize results DataFrame
        self.initialize_results_df()

        dfs = []
        for org in organlist:
            bio_data = self.biodi.loc[org]['%ID/g']
            activity = np.asarray(bio_data)
            t = np.asarray(self.t)
            sigmas = np.asarray(self.biodi.loc[org]['sigma'])
            ylabel = '%ID/g'

            if not uptake:
                # Mono-exponential fit
                result_mono, fitted_mono = exponential_fit_lmfit(
                    t[tps_to_skip_fit:], activity[tps_to_skip_fit:],
                    num_exponentials=1,
                    sigma=sigmas[tps_to_skip_fit:],
                    params_init={'A1': monoguess[0], 'A2': monoguess[1]},
                    bounds={'A1': (0, None), 'A2': (decayconst, None)}
                )         
                plot_tac_residuals(result=result_mono, region=org, y_label=ylabel)
                
                # Bi-exponential fit
                result_bi, fitted_bi = exponential_fit_lmfit(
                    t[tps_to_skip_fit:], activity[tps_to_skip_fit:],
                    num_exponentials=2,
                    sigma=sigmas[tps_to_skip_fit:],
                    params_init={'A1': result_mono.params['A1'].value * 0.6, 'A2': result_mono.params['A2'].value * 0.8, 
                                 'B1': result_mono.params['A1'].value * 0.4, 'B2': result_mono.params['A2'].value * 1.2},
                    bounds={'A1': (0, None), 'A2': (decayconst, None),
                            'B1': (0, None), 'B2': (decayconst, None)}
                )
                plot_tac_residuals(result=result_bi, region=org, y_label=ylabel)
                # Store results
                mono_params = result_mono.params.valuesdict()
                bi_params = result_bi.params.valuesdict()

                organ_data = {
                    'Organ': [org],
                    'mono_exp:%ID/g': [mono_params['A1']], 
                    'mono_exp:lambda_effective_1/h': [mono_params['A2']],  
                    'bi_exp:1_%ID/g': [bi_params['A1']], 
                    'bi_exp:lambda_effective1_1/h': [bi_params['A2']], 
                    'bi_exp:lambda_effective2_1/h': [bi_params['B2']], 
                    'bi_exp:2_%ID/g': [bi_params['B1']]
                }
                organ_df = pd.DataFrame(organ_data, index=[org])
                dfs.append(organ_df)

                # Calculate areas
                if tps_to_skip_fit == 0:
                    area_mono = integrate.quad(
                        lambda x: monoexp_fun(x, mono_params['A1'], mono_params['A2']), 
                        0, np.inf
                    )
                    area_bi = integrate.quad(
                        lambda x: biexp_fun(x, bi_params['A1'], bi_params['A2'], 
                                          bi_params['B1'], bi_params['B2']), 
                        0, np.inf
                    )
                else:
                    # Handle skipped points by adding trapezoidal areas
                    triangle_area = t[0] * bio_data.iloc[0] / 2
                    trapezoid_area = (bio_data.iloc[0] + bio_data.iloc[1]) * (t[1] - t[0]) / 2

                    monoexp_area = integrate.quad(
                        lambda x: monoexp_fun(x, mono_params['A1'], mono_params['A2']), 
                        t[tps_to_skip_fit], np.inf
                    )
                    biexp_area = integrate.quad(
                        lambda x: biexp_fun(x, bi_params['A1'], bi_params['A2'], 
                                          bi_params['B1'], bi_params['B2']), 
                        t[tps_to_skip_fit], np.inf
                    )

                    if tps_to_skip_fit == 1:
                        area_mono = triangle_area + trapezoid_area + monoexp_area[0]
                        area_bi = triangle_area + trapezoid_area + biexp_area[0]
                    elif tps_to_skip_fit >= 2:
                        trapezoid_area2 = (bio_data.iloc[1] + bio_data.iloc[2]) * (t[2] - t[1]) / 2
                        area_mono = triangle_area + trapezoid_area + trapezoid_area2 + monoexp_area[0]
                        area_bi = triangle_area + trapezoid_area + trapezoid_area2 + biexp_area[0]

                self.update_fit_results(org, mono_params=mono_params, bi_params=bi_params,
                                    area_mono=area_mono, area_bi=area_bi)

            else:
                # Uptake model
                result_uptake, fitted_uptake = exponential_fit_lmfit(
                    t[tps_to_skip_fit:], activity[tps_to_skip_fit:],
                    num_exponentials=2,
                    sigma=sigmas[tps_to_skip_fit:],
                    with_uptake=True,
                    params_init={'A1': uptakeguess[0], 'A2': uptakeguess[1], 
                               'B1': uptakeguess[2], 'B2': uptakeguess[3]},
                    bounds={'A1': (0, None), 'A2': (decayconst, None),
                            'B1': (None, None), 'B2': (decayconst, None)}
                )
                plot_tac_residuals(result=result_uptake, region=org, y_label=ylabel)
                uptake_params = result_uptake.params.valuesdict()
                area_uptake = integrate.quad(
                    lambda x: biexp_fun_uptake(x, uptake_params['A1'], uptake_params['A2'], 
                                            uptake_params['B2']), 
                    0, np.inf
                )

                self.update_fit_results(org, uptake_params=uptake_params, area_uptake=area_uptake)

        try:
            self.fitting_parameters = pd.concat(dfs, ignore_index=True)
        except Exception as e:
            print(f"Error creating fitting parameters DataFrame: {e}")
        
        

#    def get_fit_accepted_dict(self):
#        self.fit_accepted = {}  # Initialize an empty dictionary
#        organs = self.biodi.index  # Assuming self.biodi contains the organ names
#        
#        # Ask the user to provide the numbers for each organ
#        for organ in organs:
#            while True:
#                try:
#                    choice = int(input(f"Enter a number (1, 2, 3, 4) for {organ}: "))
#                    if choice in [1, 2, 3, 4]:
#                        self.fit_accepted[organ] = choice
#                        break
#                    else:
#                        print("Invalid choice. Please choose from 1, 2, 3, or 4.")
#                except ValueError:
#                    print("Invalid input. Please enter a valid number.")


#    def num_decays(self,fit_accepted):
#        ''' Sets the number of decays in each of the organ based on the accepted fit (e.g. exponential or bi-exponential) '''
#        self.fit_accepted=fit_accepted
#        self.disintegrations=pd.DataFrame(index=self.biodi.index,columns=['%ID/g*h'])
#        for key in self.fit_accepted:
#            self.disintegrations.loc[key,'%ID/g*h']=self.fit_results[key][self.fit_accepted[key]*2-1][0]
#       
#        self.disintegrations.loc["Remainder Body"] = np.nan

    def num_decays(self, fit_accepted):
        ''' Sets the number of decays in each of the organ based on the accepted fit (e.g. exponential or bi-exponential) '''
        self.disintegrations=pd.DataFrame(index=self.biodi.index,columns=['%ID/g*h'])
        self.fit_accepted = fit_accepted

        for organ, choice in self.fit_accepted.items():
            if choice == 1:
                column_name = 'Mono-Exponential'
            elif choice == 2:
                column_name = 'Bi-Exponential'
            elif choice == 3:
                column_name = 'Bi-Exponential_uptake'
            elif choice == 4:
                column_name = 'Tri-Exponential'
            else:
                continue  # Skip choices that are not 1, 2, 3, or 4
            
            if organ in self.area.index and column_name in self.area.columns:
                area_value = self.area.loc[organ, column_name]
                self.disintegrations.at[organ, '%ID/g*h'] = area_value

        if 'Right Colon' in self.disintegrations.index:
            if pd.isna(self.disintegrations.loc['Right Colon']).any():
                self.disintegrations.loc['Right Colon'] = self.disintegrations.loc['Left Colon']
                self.disintegrations.loc['Rectum'] = self.disintegrations.loc['Left Colon']
                
        if 'Red Marrow' in self.disintegrations.index:
            if pd.isna(self.disintegrations.loc['Red Marrow']).any():
                self.disintegrations.loc['Red Marrow'] = self.disintegrations.loc['Heart Contents'] * 0.34



        self.disintegrations.loc["Remainder Body"] = np.nan
        
        
    def calculate_tumor_sink_effect(self):
        tumor_value = self.disintegrations['h']['Tumor']
        
        # Calculate the sum of disintegration values for all organs except 'Remainder Body' (Organs included in the ROB are added to WB, so we don't want to add them twice)
        wb_value = self.disintegrations['h'].sum() - self.disintegrations['h']['Remainder Body'] 

        self.tumor_sink_effect_factor = (1 + (tumor_value / (wb_value - tumor_value))) # represents a multiplicative adjustment to the organ's disintegration value to account for normalizing or redistributing activity after subtracting the tumor's share from the whole body
        print(f'Tumor sink effect: {self.tumor_sink_effect_factor}')
    
    def tumor_sink_effect_correction(self, df):
        df_corrected = df.copy()

        for organ in df_corrected.columns:
            if organ != 'Tumor':
                df_corrected[organ] *= self.tumor_sink_effect_factor
        
        return df_corrected



    def phantom_data(self):
        print(PHANTOM_PATH)
        if 'mouse' in self.phantom.lower():
            self.phantom_mass = pd.read_csv(path.join(PHANTOM_PATH,'mouse_phantom_masses.csv'))  # TODO: CHANGE PATH 
#        elif 'human' in self.phantom.lower():
#            self.phantom_mass = pd.read_csv(path.join(PHANTOM_PATH,'human_phantom_masses.csv'))
        self.phantom_mass.set_index('Organ',inplace=True)
        self.phantom_mass.sort_index(inplace=True)
        self.not_inphantom=[]
        for org in self.biodi.index: ########## for org in self.disinteggrations.index:
            if org not in self.phantom_mass.index:            
                self.not_inphantom.append(org)
        rob = ['Remainder Body']
        self.not_inphantom = list(set(self.not_inphantom) - set(rob))
        print('These organs from the biodi are not modelled in the phantom\n{}'.format(self.not_inphantom))
        
        
        self.phantom_mass.loc['Remainder Body']=(self.phantom_mass.loc['Body']-self.phantom_mass.loc[self.phantom_mass.index!='Body'].sum())
        largeintestine = ['Left Colon', 'Right Colon', 'Rectum']
        self.not_inbiodi=[]
        for org in self.phantom_mass.index: 
            if org not in self.disintegrations.index and org != 'Body' and org != 'Remainder Body':            
                self.not_inbiodi.append(org)
        self.not_inbiodi = list(set(self.not_inbiodi) - set(largeintestine))
        print('\nThese organs modelled in the phantom were not included in the biodistribution.\n{}'.format(self.not_inbiodi))
        self.phantom_mass.loc[self.not_inbiodi].sum()
        self.phantom_mass.loc['Remainder Body']=self.phantom_mass.loc['Remainder Body']+self.phantom_mass.loc[self.not_inbiodi].sum()
    
    def remainder_body_uptake(self,tumor_name=None):
        print("At this point we are ignoring the tumor")
        if tumor_name:
            self.not_inphantom_notumor=[org for org in self.not_inphantom if tumor_name not in org]
            tumortemp = self.biodi.loc[tumor_name]
        else:
            self.not_inphantom_notumor=self.not_inphantom
        self.not_inphantom_notumor
        print(self.phantom)
        print('Disintegrations\n')

        # These organs that are not modelled in the phantom are now going to be scaled using mass information from the literature:
        if 'mouse' in self.phantom.lower():
            self.literature_mass = pd.read_csv(path.join(PHANTOM_PATH,'mouse_notinphantom_masses.csv'))  # TODO: CHANGE PATH
        
#        elif 'human' in self.phantom.lower():
#            self.literature_mass = pd.read_csv(path.join(PHANTOM_PATH,'human_notinphantom_masses.csv'))  # TODO: CHANGE PATH
            
        self.literature_mass.set_index('Organ',inplace=True)

        if 'mouse' in self.phantom.lower():
            self.literature_mass.loc['Muscle'] = self.phantom_mass.loc['Remainder Body']-self.literature_mass.sum()
        

        self.literature_mass=self.literature_mass.loc[self.disintegrations.index.intersection(self.literature_mass.index)]
        print('\nLiterature Mass (g)\n')
        print(self.literature_mass)
        print(self.phantom_mass)
        self.not_inphantom_notumor.remove('Tail')
        self.phantom_mass.loc['Residual'] = self.phantom_mass.loc['Remainder Body'] - self.literature_mass.sum()
        if 'mouse' in self.phantom.lower():
            self.disintegrations['%ID*h']=self.disintegrations['%ID/g*h']*self.phantom_mass[self.mouse_mass]
            if 'Residual' in self.disintegrations.index:
                self.not_inphantom_notumor.remove('Residual')
                self.disintegrations.loc['Remainder Body', '%ID*h'] = (self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor].mul(self.literature_mass[self.mouse_mass].loc[self.not_inphantom_notumor]).sum()) + (self.disintegrations.loc['Residual', '%ID/g*h'] * (self.phantom_mass.loc['Residual', self.mouse_mass]))
            else:    
                self.disintegrations.loc['Remainder Body', '%ID*h'] = (self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor].mul(self.literature_mass[self.mouse_mass].loc[self.not_inphantom_notumor]).sum())
            for org in self.not_inphantom_notumor:
                if org != 'Tail':
                    self.disintegrations.loc[org, '%ID*h'] = self.disintegrations.loc[org, '%ID/g*h'] * self.literature_mass.loc[org, self.mouse_mass]
                else:
                    pass
                
                
#        elif 'human' in self.phantom.lower():
#            self.disintegrations['%ID*h Female']=self.disintegrations['%ID/g*h']*self.phantom_mass['Female']
#            self.disintegrations['%ID*h Male']=self.disintegrations['%ID/g*h']*self.phantom_mass['Male']
#            
#            self.disintegrations.loc['Remainder Body', '%ID*h Female'] = (self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor].mul(self.literature_mass['Female']).sum())
#            self.disintegrations.loc['Remainder Body', '%ID*h Male'] = (self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor].mul(self.literature_mass['Male']).sum())
#            #self.disintegrations.loc['Remainder Body']['%ID*h Female']=(self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor]*self.literature_mass['Female']).sum()
#            #self.disintegrations.loc['Remainder Body']['%ID*h Male']=(self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor]*self.literature_mass['Male']).sum()
        

        #self.disintegrations.drop(not_inphantom_notumor,inplace=True)
        self.disintegrations['h']=self.disintegrations['%ID*h']/100
        
        
    def remainder_body_uptake_human(self,tumor_name=None):
        print("At this point we are ignoring the tumor")
        if tumor_name:
            self.not_inphantom_notumor=[org for org in self.not_inphantom if tumor_name not in org]
            tumortemp = self.biodi.loc[tumor_name]
        else:
            self.not_inphantom_notumor=self.not_inphantom
        self.not_inphantom_notumor
        #print(self.phantom)
        #print('Disintegrations\n')

        # These organs that are not modelled in the phantom are now going to be scaled using mass information from the literature:
        #self.literature_mass = pd.read_csv(path.join(PHANTOM_PATH,'human_notinphantom_masses.csv'))  # TODO: CHANGE PATH
            
        #self.literature_mass.set_index('Organ',inplace=True)

        #self.literature_mass=self.literature_mass.loc[self.disintegrations.index.intersection(self.literature_mass.index)]
        #print('\nLiterature Mass (g)\n')
        #print(self.literature_mass)

        #print(self.phantom_mass)

        #self.disintegrations.loc['Remainder Body', 'h Female'] = (self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor].mul(self.literature_mass['Female']).sum())
        #self.disintegrations.loc['Remainder Body', 'h Male'] = (self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor].mul(self.literature_mass['Male']).sum())
            #self.disintegrations.loc['Remainder Body']['%ID*h Female']=(self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor]*self.literature_mass['Female']).sum()
            #self.disintegrations.loc['Remainder Body']['%ID*h Male']=(self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor]*self.literature_mass['Male']).sum()
        

        #self.disintegrations.drop(not_inphantom_notumor,inplace=True)
        #self.disintegrations['h']=self.disintegrations['%ID*h']/100
            
    def not_inphantom_notumor_fun(self):
        self.disintegrations.drop(self.not_inphantom_notumor,inplace=True)
            
    def add_tumor_mass(self,tumor_name,tumor_mass):
        self.phantom_mass.loc[tumor_name]=tumor_mass  #grams   Provided by average of biodi from Etienne
        
            
    def create_mousecase(self, df, savefile=False,dirname='./',):
        
        '''This function creates a pandas dataframe that looks exactly as the case files generated by OLINDA for the g mouse.        
           The result can be viewed under self.mousecase, and the pandas methods can be used to finally save it if wanted.
        '''
        filename = self.phantom.lower() + '.cas'
        template=pd.read_csv(path.join(TEMPLATE_PATH, filename))
        template.columns=['Data']

        #modify the isotope in the template
        ind=template[template['Data']=='[BEGIN NUCLIDES]'].index
        template.loc[ind[0]+1,'Data']=self.isotope + '|'
        input_organs = df.drop(set(self.not_inphantom).intersection(set(df.index))).index.to_list()
        if 'Residual' in input_organs:
            input_organs.remove('Residual')
        else:
            pass
        if 'Tail' in input_organs:
            input_organs.remove('Tail')
        else:
            pass
        print(input_organs)
        #change the kinetics for each input organ
        for org in input_organs:  #ignore the tumor here

            temporg=org
            uptakepos=2
            
            if org=='Large Intestine':
                temporg='LLI'        
            elif org=='Skeleton':
                temporg='Bone' #think about trabecular vs cortical pos
                uptakepos=3
            elif org=='Remainder Body':
                temporg='Body'
            elif org=='Heart':
                temporg='Heart'
                uptakepos=3
                
            ind=template[template['Data'].str.contains(temporg)].index
            print(org)
            sourceorgan=template.iloc[ind[0]].str.split('|')[0][0]
            massorgan=template.iloc[ind[0]].str.split('|')[0][1]
            kineticdata=df.loc[org]['h']

            if np.isnan(kineticdata):
                kineticdata=0

            template.iloc[ind[0]]=sourceorgan+'|'+massorgan+'|'+'{:7f}'.format(kineticdata)
            template.iloc[ind[uptakepos]]=sourceorgan+'|'+'{:7f}'.format(kineticdata)

        now = datetime.datetime.now()
        template.columns=['Saved on ' + now.strftime("%m.%d.%Y") +' at ' + now.strftime('%H:%M:%S')]
            
        self.mousecase=template

        if savefile==True:
            if not path.exists(dirname):
                makedirs(dirname)

            self.mousecase.to_csv(dirname+'/'+filename,index=False)
        
        print(f'The case file {filename} has been saved in\n{format(dirname)}')
        
    def rename_organ(self,oldname,newname):
        ind_list=self.disintegrations.index.tolist()
        ind_pos = ind_list.index(oldname)
        ind_list[ind_pos] = newname
        self.disintegrations.index = ind_list
        
        ind_list=self.biodi.index.tolist()
        ind_pos = ind_list.index(oldname)
        ind_list[ind_pos] = newname
        self.biodi.index = ind_list
        
#    def heart_contents(self):
#        self.disintegrations.loc['Red Marrow'] = self.disintegrations.loc['Heart Contents']*0.34
#        self.disintegrations.sort_index(inplace=True)

    def humanize_tiac(self,small_intestine=False):
        ''' small_intestine: False if it is already in the biodi.
                            True to assume same as Large intestine '''

        human = deepcopy(self)

        human.phantom='AdultHuman'

        human.biodi.loc['Small Intestine'] = human.biodi.loc['Large Intestine']
        human.biodi.sort_index(inplace=True)

        human.rename_organ('Skeleton','Bone Surfaces')
        human.rename_organ('Blood','Heart Contents')
        human.disintegrations = human.disintegrations.drop('Tail', axis=0)

        #human.heart_contents()
        human.phantom_mass = pd.read_csv(path.join(PHANTOM_PATH,'human_phantom_masses.csv'))
        human.phantom_mass.set_index('Organ',inplace=True)
        human.phantom_mass.sort_index(inplace=True)
        human.not_inphantom=[]
        for org in human.disintegrations.index: ########## for org in self.disinteggrations.index:
            if org not in human.phantom_mass.index:            
                human.not_inphantom.append(org)
        rob = ['Remainder Body']
        human.not_inphantom = list(set(human.not_inphantom) - set(rob))
        print('These organs from the biodi are not modelled in the phantom\n{}'.format(human.not_inphantom))
#        tempfitresults=deepcopy(human.fit_results)
#        fit_results={}
#        fit_accepted={}
#        for key in human.fit_results:
#            if key == 'Skeleton':
#                fit_results['Bone Surfaces']=tempfitresults[key]
#                fit_accepted['Bone Surfaces']=human.fit_accepted[key]
#            elif key == 'Large Intestine':
#                fit_results['Left Colon']=tempfitresults[key]
#                fit_accepted['Left Colon']=human.fit_accepted[key]
#
#                fit_results['Right Colon']=tempfitresults[key]
#                fit_accepted['Right Colon']=human.fit_accepted[key]
#
#                fit_results['Rectum']=tempfitresults[key]
#                fit_accepted['Rectum']=human.fit_accepted[key]
#
#                if small_intestine:
#                    fit_results['Small Intestine']=tempfitresults[key]
#                    fit_accepted['Small Intestine']=human.fit_accepted[key]
#
#            elif key == 'Blood':
#                fit_results['Heart Contents']=tempfitresults[key]
#                fit_accepted['Heart Contents']=human.fit_accepted[key]
#
#                fit_results['Red Marrow']=deepcopy(tempfitresults[key])
#                fit_results['Red Marrow'][0]=fit_results['Red Marrow'][0]*0.34
#                fit_results['Red Marrow'][1]=tuple([0.34*x for x in list(fit_results['Red Marrow'][1])])
#                fit_results['Red Marrow'][2]=fit_results['Red Marrow'][2]*0.34
#                fit_results['Red Marrow'][3]=tuple([0.34*x for x in list(fit_results['Red Marrow'][3])])
#                fit_accepted['Red Marrow']=human.fit_accepted[key]
#
#                
#            else:
#                fit_results[key]=tempfitresults[key]
#                fit_accepted[key]=human.fit_accepted[key]
#
#        human.fit_results=fit_results
#        human.fit_accepted=fit_accepted
#
#        human.area.index = human.area.index.str.replace('Skeleton','Bone Surfaces')
#        human.area.index = human.area.index.str.replace('Large Intestine','Left Colon')
#        human.area.index = human.area.index.str.replace('Blood','Heart Contents')
        return human
    

        
    def interspecies_conversion(self):
        organs = self.disintegrations['%ID*h Male'].index
        for organ in organs:
            print(organ)
            print(self.phantom_mass['Male'].loc[organ])
            
        
        self.disintegrations['%ID*h Male']=self.disintegrations['%ID*h Male']*(self.wb_m/self.phantom_mass['Male'].loc['Body'])
        self.disintegrations['%ID*h Female']=self.disintegrations['%ID*h Female']*(self.wb_m/self.phantom_mass['Female'].loc['Body'])

    def m1(self):

        self.disintegrations_m1=self.disintegrations.copy()

        self.disintegrations_m1.drop('%ID/g*h', axis=1, inplace=True)
        self.disintegrations_m1.drop('%ID*h', axis=1, inplace=True)

        self.disintegrations_m1.rename(columns={'h': 'h Male'}, inplace=True)
        self.disintegrations_m1['h Female'] = self.disintegrations_m1['h Male']
            
        self.disintegrations_m1.loc['Rectum', 'h Female'] = (70/360) * self.disintegrations_m1.loc['Large Intestine', 'h Female'] 
        self.disintegrations_m1.loc['Rectum', 'h Male'] = (70/370) * self.disintegrations_m1.loc['Large Intestine', 'h Male']    
        self.disintegrations_m1.loc['Left Colon', 'h Female'] = (145/360) * self.disintegrations_m1.loc['Large Intestine', 'h Female'] 
        self.disintegrations_m1.loc['Left Colon', 'h Male'] = (150/370) * self.disintegrations_m1.loc['Large Intestine', 'h Male']
        self.disintegrations_m1.loc['Right Colon', 'h Female'] = (145/360) * self.disintegrations_m1.loc['Large Intestine', 'h Female'] 
        self.disintegrations_m1.loc['Right Colon', 'h Male'] = (150/370) * self.disintegrations_m1.loc['Large Intestine', 'h Male']
        self.disintegrations_m1 = self.disintegrations_m1.drop('Large Intestine', axis=0)
        try:
            self.disintegrations_m1 = self.disintegrations_m1.drop('Tumor', axis=0)
        except:
            print('No tumor')
        #if 'Red Marrow' != self.disintegrations_m1.index:
        #    self.disintegrations_m1.loc['Red Marrow'] = 0.34 * self.disintegrations_m1.loc['Heart Contents']       
        if 'Red Marrow' in self.disintegrations.index:
            if pd.isna(self.disintegrations.loc['Red Marrow']).any():
                self.disintegrations.loc['Red Marrow'] = self.disintegrations.loc['Heart Contents'] * 0.34
                
        self.disintegrations_m1.sort_index(inplace=True)
        
        
        return self.disintegrations_m1
    
    def m2(self, tumor_name=None):
        all_organ = list(self.disintegrations.index)
        try:
            all_organ.remove('Tumor')
        except:
            print('No tumor')
        #all_organ.remove('Tail')
        if tumor_name:
            self.not_inphantom_notumor=[org for org in self.not_inphantom if tumor_name not in org]
            tumortemp = self.biodi.loc[tumor_name]
        else:
            self.not_inphantom_notumor=self.not_inphantom
        self.not_inphantom_notumor
        try:
            self.not_inphantom_notumor.remove('Tail')
        except:
            print('')
        try:
            self.not_inphantom_notumor.remove('Large Intestine')
        except:
            print('')
        self.phantom_mass = pd.read_csv(path.join(PHANTOM_PATH,'human_phantom_masses.csv'))
        self.phantom_mass.set_index('Organ',inplace=True)
        self.phantom_mass.sort_index(inplace=True)
        
        self.literature_mass = pd.read_csv(path.join(PHANTOM_PATH,'human_notinphantom_masses.csv'))  # TODO: CHANGE PATH
        self.literature_mass.set_index('Organ',inplace=True)
        blood = ['Blood']
        self.not_inphantom_notumor = list(set(self.not_inphantom_notumor) - set(self.phantom_mass.index) - set(blood))

        
        self.in_phantom_organs = [x for x in all_organ if x not in self.not_inphantom_notumor]
        self.in_phantom_organs

        

    def create_humancase(self, df, method, savefile=False,dirname='./'):
        
        '''This function creates a pandas dataframe that looks exactly as the case files generated by OLINDA for the human.        
           The result can be viewed under self.humancas, and the pandas methods can be used to finally save it if wanted.
        '''

        if self.sex=='Male':
            template=pd.read_csv(path.join(TEMPLATE_PATH,'adult_male.cas'))            
        else:
            template=pd.read_csv(path.join(TEMPLATE_PATH,'adult_female.cas'))
        
        template.columns=['Data']

        #modify the isotope in the template
        ind=template[template['Data']=='[BEGIN NUCLIDES]'].index
        template.loc[ind[0]+1,'Data']=self.isotope + '|'

        #change the kinetics for each input organ
        for org in df.drop(set(self.not_inphantom).intersection(set(df.index))).index:  #ignore the tumor here
            temporg=org
            uptakepos=2
            if org=='Left Colon':
                temporg='ULI'        
            elif org=='Right Colon':
                temporg='LLI'
            elif org=='Bone Surfaces':
                temporg='Bone' #think about trabecular vs cortical pos
                uptakepos=3
            elif org=='Remainder Body':
                temporg='Body'
            elif org=='Heart':
                temporg='Heart Wall'        
            elif org=='Heart Contents':
                temporg='Heart Contents'
                uptakepos=1       
                
            ind=template[template['Data'].str.contains(temporg)].index
            sourceorgan=template.iloc[ind[0]].str.split('|')[0][0]
            massorgan=template.iloc[ind[0]].str.split('|')[0][1]
            kineticdata=df.loc[org]['h '+self.sex]
            
            if np.isnan(kineticdata):
                kineticdata=0

            template.iloc[ind[0]]=sourceorgan+'|'+massorgan+'|'+'{:7f}'.format(kineticdata)
            template.iloc[ind[uptakepos]]=sourceorgan+'|'+'{:7f}'.format(kineticdata)

        now = datetime.datetime.now()
        template.columns=['Saved on ' + now.strftime("%m.%d.%Y") +' at ' + now.strftime('%H:%M:%S')]
            
        self.humancase=template

        if savefile==True:
            if not path.exists(dirname):
                makedirs(dirname)

            self.humancase.to_csv(dirname+'/'+self.sex+method+'.cas',index=False)
        
        print('The case file {} has been saved in\n{}'.format(self.sex+'.cas',dirname))
        

    def scale_biexponential_tiac(self, row):
        
        Cm_organ_t0_1 = row['bi_exp1:%ID'] / 100 
        Cm_organ_t0_2 = row['bi_exp2:%ID'] / 100 
       
       
        exp1 = row['bi_exp1:%ID']  / row['bi_exp:lambda_effective1_1/h']
        exp2 = row['bi_exp2:%ID'] / row['bi_exp:lambda_effective2_1/h']
        sum = exp1 + exp2

        frac_exp1 = exp1/sum
        frac_exp2 = exp2/sum
        lambda_effective1 = row['bi_exp:lambda_effective1_1/h']
        lambda_effective2 = row['bi_exp:lambda_effective2_1/h']
        
        
        lambda_physical = log(2) / self.half_life #1/h
        
        lambda_biological1 = lambda_effective1 - lambda_physical
        lambda_biological2 = lambda_effective2 - lambda_physical

        wb_m = int(self.mouse_mass[:-1])
        k_b_male = (73000 / wb_m) ** 0.25
        k_b_female = (60000 / wb_m) ** 0.25

        #TIAC_h_bi_male = ((Cm_organ_t0_1 * k_b_male) / ((lambda_effective1)) )+( (Cm_organ_t0_2 * k_b_male) / ((lambda_effective2)))
        TIAC_h_bi_male = ((Cm_organ_t0_1 ) / (((k_b_male)**(-1)*lambda_biological1 + lambda_physical)) ) + ( (Cm_organ_t0_2 ) / ((k_b_male)**(-1)*(lambda_biological2) + lambda_physical))
        
        
        #TIAC_h_bi_female = ((Cm_organ_t0_1 * k_b_female) / ((lambda_effective1)) )+( (Cm_organ_t0_2 * k_b_female) / ((lambda_effective2)))
        TIAC_h_bi_female = ((Cm_organ_t0_1 * k_b_female) / ((lambda_effective1)) )+( (Cm_organ_t0_2 * k_b_female) / ((lambda_effective2)))
        


        return TIAC_h_bi_male, TIAC_h_bi_female


    def scale_monoexponential_tiac(self, row):

        lambda_effective = row['mono_exp:lambda_effective_1/h']

        
        Cm_organ_t0 = row['mono_exp:%ID'] / 100 
        
        wb_m = int(self.mouse_mass[:-1])
        k_b_male = (73000 / wb_m) ** 0.25
        
        k_b_female = (60000 / wb_m) ** 0.25

        TIAC_h_mono_male = Cm_organ_t0 / (k_b_male**(-1) * (lambda_effective))
        TIAC_h_mono_female = Cm_organ_t0 / (k_b_female**(-1) * (lambda_effective))

        return TIAC_h_mono_male, TIAC_h_mono_female


    def M3(self,human_not_inphantom_notumor, tumor_name=None):
        

        print("At this point we are ignoring the tumor")
        if tumor_name:
            self.not_inphantom_notumor=[org for org in self.not_inphantom if tumor_name not in org]
            tumortemp = self.biodi.loc[tumor_name]
        else:
            self.not_inphantom_notumor=self.not_inphantom
        self.not_inphantom_notumor
        print(self.phantom)
        print('Disintegrations\n')

        # These organs that are not modelled in the phantom are now going to be scaled using mass information from the literature:
        if 'mouse' in self.phantom.lower():
            self.literature_mass = pd.read_csv(path.join(PHANTOM_PATH,'mouse_notinphantom_masses.csv'))  # TODO: CHANGE PATH
        
#        elif 'human' in self.phantom.lower():
#            self.literature_mass = pd.read_csv(path.join(PHANTOM_PATH,'human_notinphantom_masses.csv'))  # TODO: CHANGE PATH
            
        self.literature_mass.set_index('Organ',inplace=True)

        if 'mouse' in self.phantom.lower():
            self.literature_mass.loc['Muscle'] = self.phantom_mass.loc['Remainder Body']-self.literature_mass.sum()
        

        self.literature_mass=self.literature_mass.loc[self.fitting_parameters.index.intersection(self.literature_mass.index)]

        fit_accepted_df = pd.DataFrame(list(self.fit_accepted.items()), columns=['Organ', 'fit_accepted'])
        self.fitting_parameters = self.fitting_parameters.merge(fit_accepted_df, on="Organ")
        self.fitting_parameters.set_index('Organ', inplace=True)

                
        if 'mouse' in self.phantom.lower():
            self.fitting_parameters['mono_exp:%ID']=self.fitting_parameters['mono_exp:%ID/g']*self.phantom_mass[self.mouse_mass]
            self.fitting_parameters['bi_exp1:%ID']=self.fitting_parameters['bi_exp:1_%ID/g']*self.phantom_mass[self.mouse_mass]
            self.fitting_parameters['bi_exp2:%ID']=self.fitting_parameters['bi_exp:2_%ID/g']*self.phantom_mass[self.mouse_mass]
            if 'Residual' in self.fitting_parameters.index:
                self.not_inphantom_notumor.remove('Residual')
                self.fitting_parameters.loc['Remainder Body', 'mono_exp:%ID'] = (self.fitting_parameters['mono_exp:%ID/g'].loc[self.not_inphantom_notumor].mul(self.literature_mass[self.mouse_mass].loc[self.not_inphantom_notumor]).sum()) + (self.disintegrations.loc['Residual', '%ID/g*h'] * (self.phantom_mass.loc['Residual', self.mouse_mass]))
            else:    
                self.fitting_parameters.loc['Remainder Body', 'mono_exp:%ID'] = (self.fitting_parameters['mono_exp:%ID/g'].loc[self.not_inphantom_notumor].mul(self.literature_mass[self.mouse_mass]).sum())
                self.fitting_parameters.loc['Remainder Body', 'bi_exp1:%ID'] = (self.fitting_parameters['bi_exp:1_%ID/g'].loc[self.not_inphantom_notumor].mul(self.literature_mass[self.mouse_mass]).sum())
                self.fitting_parameters.loc['Remainder Body', 'bi_exp2:%ID'] = (self.fitting_parameters['bi_exp:2_%ID/g'].loc[self.not_inphantom_notumor].mul(self.literature_mass[self.mouse_mass]).sum())

            for org in self.not_inphantom_notumor:
                if org != 'Tail':
                    self.fitting_parameters.loc[org, 'mono_exp:%ID'] = self.fitting_parameters.loc[org, 'mono_exp:%ID/g'] * self.literature_mass.loc[org, self.mouse_mass]
                    self.fitting_parameters.loc[org, 'bi_exp1:%ID'] = self.fitting_parameters.loc[org, 'bi_exp:1_%ID/g'] * self.literature_mass.loc[org, self.mouse_mass]
                    self.fitting_parameters.loc[org, 'bi_exp2:%ID'] = self.fitting_parameters.loc[org, 'bi_exp:2_%ID/g'] * self.literature_mass.loc[org, self.mouse_mass]

                else:
                    pass
        print(self.fitting_parameters)
        self.fitting_parameters = self.fitting_parameters.drop('Tail', axis=0)
        self.fitting_parameters = self.fitting_parameters.drop('Tumor', axis=0)
        
        for i, organ in enumerate(self.fitting_parameters.index):
            if self.fitting_parameters.loc[organ, 'fit_accepted'] == 1.0:
                TIAC_h_mono_male, TIAC_h_mono_female = self.scale_monoexponential_tiac(self.fitting_parameters.iloc[i])
                self.fitting_parameters.loc[organ, 'h Male'] = TIAC_h_mono_male
                self.fitting_parameters.loc[organ, 'h Female'] = TIAC_h_mono_female

            elif self.fitting_parameters.loc[organ, 'fit_accepted'] == 2.0:
                TIAC_h_bi_male, TIAC_h_bi_female = self.scale_biexponential_tiac(self.fitting_parameters.iloc[i])
                self.fitting_parameters.loc[organ, 'h Male'] = TIAC_h_bi_male
                self.fitting_parameters.loc[organ, 'h Female'] = TIAC_h_bi_female
                
        self.fitting_parameters.rename({'Skeleton': 'Bone Surfaces'}, inplace=True)
        self.fitting_parameters.rename({'Blood': 'Heart Contents'}, inplace=True)
        self.template_for_m4 =  self.fitting_parameters
        

        self.fitting_parameters.loc['Rectum', 'h Female'] = (70/360) * self.fitting_parameters.loc['Large Intestine', 'h Female'] 
        self.fitting_parameters.loc['Rectum', 'h Male'] = (70/370) * self.fitting_parameters.loc['Large Intestine', 'h Male']    
        self.fitting_parameters.loc['Left Colon', 'h Female'] = (145/360) * self.fitting_parameters.loc['Large Intestine', 'h Female'] 
        self.fitting_parameters.loc['Left Colon', 'h Male'] = (150/370) * self.fitting_parameters.loc['Large Intestine', 'h Male']
        self.fitting_parameters.loc['Right Colon', 'h Female'] = (145/360) * self.fitting_parameters.loc['Large Intestine', 'h Female'] 
        self.fitting_parameters.loc['Right Colon', 'h Male'] = (150/370) * self.fitting_parameters.loc['Large Intestine', 'h Male']
        self.fitting_parameters = self.fitting_parameters.drop('Large Intestine', axis=0)

        self.fitting_parameters.loc['Red Marrow'] = 0.34 * self.fitting_parameters.loc['Heart Contents']  

        self.fitting_parameters.loc['Remainder Body', 'h Female'] = self.fitting_parameters['h Female'].loc[human_not_inphantom_notumor].sum()
        self.fitting_parameters.loc['Remainder Body', 'h Male'] =  self.fitting_parameters['h Male'].loc[human_not_inphantom_notumor].sum()
        
        for org in human_not_inphantom_notumor:
            self.fitting_parameters = self.fitting_parameters.drop(org, axis=0)
        
        self.fitting_parameters.sort_index(inplace=True)
            
def m2(human, mouse,  mouse_mass = '25g'):
    try:
        mouse.not_inphantom_notumor.remove('Tail')
    except:
        print("")
    mouse.not_inphantom_notumor
    human.disintegrations_m2=pd.DataFrame(index=human.in_phantom_organs,columns=['h Female', 'h Male'])
    if 'Residual' in human.not_inphantom_notumor:
        human.not_inphantom_notumor.remove('Residual')
    for org in human.in_phantom_organs:
        
        if org == 'Remainder Body':
            mouse_rof = mouse.literature_mass[mouse_mass].loc[human.not_inphantom_notumor].sum() #we use only organs which constitute for ROB in human. for example Adrenals were ROB in mouse, but not in humans
        elif org == 'Bone Surfaces':
            mouse_mass_org = mouse.phantom_mass.loc['Skeleton', mouse_mass]
        elif org == 'Heart Contents':
            mouse_mass_org = mouse.literature_mass.loc['Blood', mouse_mass]
        elif org in mouse.phantom_mass.index:
            mouse_mass_org = mouse.phantom_mass.loc[org,mouse_mass]
        elif org in mouse.literature_mass.index:
            mouse_mass_org = mouse.literature_mass.loc[org, mouse_mass]
        else:
            print(f'Dont have a mass for {org}')

        if org == 'Remainder Body':
            human_rof_female = human.literature_mass['Female'].loc[human.not_inphantom_notumor].sum()
            human_rof_male = human.literature_mass['Male'].loc[human.not_inphantom_notumor].sum()
        elif org == 'Large Intestine':
            human_mass_org_female = 360
            human_mass_org_male = 370
        elif org in human.phantom_mass.index and org != 'Large Intestine':
            human_mass_org_female = human.phantom_mass['Female'].loc[org]
            human_mass_org_male = human.phantom_mass['Male'].loc[org]
        else:
            print(f'Dont have a mass for {org}')


        human.disintegrations_m2.loc[org, 'h Female']=human.disintegrations.loc[org, 'h']*(int(mouse_mass[:-1]) /human.phantom_mass['Female'].loc['Body'])*(human_mass_org_female/mouse_mass_org)
        human.disintegrations_m2.loc[org, 'h Male']=human.disintegrations.loc[org, 'h']*(int(mouse_mass[:-1]) /human.phantom_mass['Male'].loc['Body'])*(human_mass_org_male/mouse_mass_org)
        if org == 'Remainder Body':
            human.disintegrations_m2.loc['Remainder Body', 'h Female'] = human.disintegrations['h'].loc[human.not_inphantom_notumor].sum()*(int(mouse_mass[:-1]) /human.phantom_mass['Female'].loc['Body'])*(human_rof_female/mouse_rof)
            human.disintegrations_m2.loc['Remainder Body', 'h Male'] = human.disintegrations['h'].loc[human.not_inphantom_notumor].sum()*(int(mouse_mass[:-1]) /human.phantom_mass['Male'].loc['Body'])*(human_rof_female/mouse_rof)

    human.disintegrations_m2.loc['Rectum', 'h Female'] = (70/360) * human.disintegrations_m2.loc['Large Intestine', 'h Female'] 
    human.disintegrations_m2.loc['Rectum', 'h Male'] = (70/370) * human.disintegrations_m2.loc['Large Intestine', 'h Male']    
    human.disintegrations_m2.loc['Left Colon', 'h Female'] = (145/360) * human.disintegrations_m2.loc['Large Intestine', 'h Female'] 
    human.disintegrations_m2.loc['Left Colon', 'h Male'] = (150/370) * human.disintegrations_m2.loc['Large Intestine', 'h Male']
    human.disintegrations_m2.loc['Right Colon', 'h Female'] = (145/360) * human.disintegrations_m2.loc['Large Intestine', 'h Female'] 
    human.disintegrations_m2.loc['Right Colon', 'h Male'] = (150/370) * human.disintegrations_m2.loc['Large Intestine', 'h Male']
    human.disintegrations_m2 = human.disintegrations_m2.drop('Large Intestine', axis=0)

    human.disintegrations_m2.loc['Red Marrow'] = 0.34 * human.disintegrations_m2.loc['Heart Contents']       
    
    human.disintegrations_m2.sort_index(inplace=True)
    

    return human.disintegrations_m2