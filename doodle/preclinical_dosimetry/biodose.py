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

        wb_value = self.disintegrations['h'].sum()  

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
        print(self.not_inphantom_notumor)
        print('Disintegrations\n')

        # These organs that are not modelled in the phantom are now going to be scaled using mass information from the literature:
        if 'mouse' in self.phantom.lower():
            self.literature_mass = pd.read_csv(path.join(PHANTOM_PATH,'mouse_notinphantom_masses.csv'))  # TODO: CHANGE PATH
        
        elif 'human' in self.phantom.lower():
            self.literature_mass = pd.read_csv(path.join(PHANTOM_PATH,'human_notinphantom_masses.csv'))  # TODO: CHANGE PATH
            
        print(self.phantom.lower())
        print(self.literature_mass)
            
        self.literature_mass.set_index('Organ',inplace=True)

        if 'mouse' in self.phantom.lower():
            self.literature_mass.loc['Muscle'] = self.phantom_mass.loc['Remainder Body']-self.literature_mass.sum()
        

        self.literature_mass=self.literature_mass.loc[self.disintegrations.index.intersection(self.literature_mass.index)]
        print('\nLiterature Mass (g)\n')
        print(self.literature_mass)
        print(self.phantom_mass)
        try:
            self.not_inphantom_notumor.remove('Tail')
        except:
            pass
        
        ## Residual is the remaining carcass of the mouse after removing the organs; not all biodi study measure its activity, but some does
        if 'Residual' in self.disintegrations.index:
            self.phantom_mass.loc['Residual'] = self.phantom_mass.loc['Remainder Body'] - self.literature_mass.sum()
        else:
            pass
        
        
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
                
                
        elif 'human' in self.phantom.lower():
            print('x')
            self.disintegrations['%ID*h Female']=self.disintegrations['%ID/g*h']*self.phantom_mass['Female']
            self.disintegrations['%ID*h Male']=self.disintegrations['%ID/g*h']*self.phantom_mass['Male']
            
            self.disintegrations.loc['Remainder Body', '%ID*h Female'] = (self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor].mul(self.literature_mass['Female']).sum())
            self.disintegrations.loc['Remainder Body', '%ID*h Male'] = (self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor].mul(self.literature_mass['Male']).sum())

        
        self.disintegrations['h']=self.disintegrations['%ID*h']/100
        self.disintegrations_all_organs = self.disintegrations.copy()  # Store the original disintegrations before dropping organs
        self.disintegrations.drop(self.not_inphantom_notumor,inplace=True) # Only organs that are in the phantom will be kept in the disintegrations dataframe and passed to olinda
        
        
            
    def not_inphantom_notumor_fun(self):
        self.disintegrations.drop(self.not_inphantom_notumor,inplace=True)
            
    def add_tumor_mass(self,tumor_name,tumor_mass):
        self.phantom_mass.loc[tumor_name]=tumor_mass  #grams   Provided by average of biodi from Etienne
        
            
    def create_mousecase(self, df, method, savefile=False,dirname='./',):
        
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

            self.mousecase.to_csv(dirname+'/'+method+filename,index=False)
        
        print(f'The case file {filename} has been saved in\n{format(dirname)}')
        
    def rename_organ(self,oldname,newname):
        ind_list=self.disintegrations_all_organs.index.tolist()
        ind_pos = ind_list.index(oldname)
        ind_list[ind_pos] = newname
        self.disintegrations_all_organs.index = ind_list
        
        ind_list=self.biodi.index.tolist()
        ind_pos = ind_list.index(oldname)
        ind_list[ind_pos] = newname
        self.biodi.index = ind_list


    def create_human(self, tumor_name = None):
        # We are mostly using the disintegrations_all_organs dataframe, but we adjust the biodi dataframe as well to match the human phantom structure
        human = deepcopy(self)
        human.phantom='AdultHuman'
          
        if 'Small Intestine' not in human.biodi.index:
            human.biodi.loc['Small Intestine'] = human.biodi.loc['Large Intestine']
            human.disintegrations_all_organs.loc['Small Intestine'] = human.disintegrations_all_organs.loc['Large Intestine']
            print('Small Intestine added to the biodi')
        else:
            print('Small Intestine already in the biodi, no need to add it')
            
        if 'Red Marrow' in self.biodi.index:
            if pd.isna(self.biodi.loc['Red Marrow']).any():
                self.biodi.loc['Red Marrow'] = self.biodi.loc['Heart Contents'] * 0.34
                self.disintegrations_all_organs.loc['Red Marrow', 'h'] = self.disintegrations_all_organs.loc['Heart Contents', 'h'] * 0.34

        if 'Skeleton' in human.biodi.index:
            human.rename_organ('Skeleton','Bone Surfaces')

        if 'Blood' in human.biodi.index:
            human.rename_organ('Blood','Heart Contents')

        if 'Tail' in human.disintegrations_all_organs.index:
            print('Tail is not modelled in the human phantom, removing it from the biodi')
            human.biodi = human.biodi.drop('Tail', axis=0)
            human.disintegrations_all_organs = human.disintegrations_all_organs.drop('Tail', axis=0)
            
        if 'Tumor' in human.disintegrations_all_organs.index:
            print('Tumor is not modelled in the human phantom, removing it from the biodi')
            human.biodi = human.biodi.drop('Tumor', axis=0)
            human.disintegrations_all_organs = human.disintegrations_all_organs.drop('Tumor', axis=0)
            
        human.phantom_mass = pd.read_csv(path.join(PHANTOM_PATH,'human_phantom_masses.csv'))
        human.phantom_mass.set_index('Organ',inplace=True)
        human.phantom_mass.sort_index(inplace=True)
        
        human.literature_mass = pd.read_csv(path.join(PHANTOM_PATH, 'human_notinphantom_masses.csv')) 
        human.literature_mass.set_index('Organ', inplace=True)

        human.disintegrations_all_organs.sort_index(inplace=True)
        print(human.disintegrations_all_organs)
        if 'h Female' not in human.disintegrations_all_organs:
            human.disintegrations_all_organs.rename(columns={'h': 'h Male'}, inplace=True)
        if 'h Female' not in human.disintegrations_all_organs:
            human.disintegrations_all_organs['h Female'] = human.disintegrations_all_organs['h Male']
        
        human.disintegrations_all_organs.loc['Rectum', 'h Female'] = (70/360) * human.disintegrations_all_organs.loc['Large Intestine', 'h Female'] 
        human.disintegrations_all_organs.loc['Rectum', 'h Male'] = (70/370) * human.disintegrations_all_organs.loc['Large Intestine', 'h Male']    
        human.disintegrations_all_organs.loc['Left Colon', 'h Female'] = (145/360) * human.disintegrations_all_organs.loc['Large Intestine', 'h Female'] 
        human.disintegrations_all_organs.loc['Left Colon', 'h Male'] = (150/370) * human.disintegrations_all_organs.loc['Large Intestine', 'h Male']
        human.disintegrations_all_organs.loc['Right Colon', 'h Female'] = (145/360) * human.disintegrations_all_organs.loc['Large Intestine', 'h Female'] 
        human.disintegrations_all_organs.loc['Right Colon', 'h Male'] = (150/370) * human.disintegrations_all_organs.loc['Large Intestine', 'h Male']
        human.disintegrations_all_organs = human.disintegrations_all_organs.drop('Large Intestine', axis=0)
        
        human.not_inphantom=[]
        
        for org in human.disintegrations_all_organs.index: ########## for org in self.disinteggrations.index:
            if org not in human.phantom_mass.index:            
                human.not_inphantom.append(org)
                
        human.not_inphantom = list(set(human.not_inphantom) - set(['Remainder Body']))
        print('These organs from the biodi are not modelled in the phantom:\n{}'.format(human.not_inphantom))
        if tumor_name:
            human.not_inphantom_notumor=[org for org in human.not_inphantom if tumor_name not in org]
        else:
            human.not_inphantom_notumor=human.not_inphantom
        
        human.disintegrations_all_organs.loc['Remainder Body', 'h Female'] = sum(human.disintegrations_all_organs.loc[human.not_inphantom_notumor, 'h Female'])
        human.disintegrations_all_organs.loc['Remainder Body', 'h Male'] = sum(human.disintegrations_all_organs.loc[human.not_inphantom_notumor, 'h Male'])
        
        human.disintegrations_all_organs = human.disintegrations_all_organs[['h Female', 'h Male']]
        human.disintegrations_all_organs.drop(human.not_inphantom, inplace=True)  # Only organs that are in the phantom will be kept in the disintegrations dataframe and passed to olinda
        human.disintegrations_all_organs.sort_index(inplace=True)
        return human
    
    def apply_relative_mass_scaling(self, mouse_mass = 25):
        rMSF_data = pd.read_csv(path.join(PHANTOM_PATH,'rMSF_factor.csv'), index_col=0)  # TODO: CHANGE PATH
        
        female_mass_sum = rMSF_data.loc[self.not_inphantom_notumor, 'Female'].sum()
        male_mass_sum   = rMSF_data.loc[self.not_inphantom_notumor, 'Male'].sum()
        mouse_mass_sum = rMSF_data.loc[self.not_inphantom_notumor, f'{mouse_mass}g_mouse'].sum()
        
        mouse_body_mass   = rMSF_data.loc['Body', f'{mouse_mass}g_mouse']
        human_body_female = rMSF_data.loc['Body', 'Female']
        human_body_male   = rMSF_data.loc['Body', 'Male']
        
        remainder_correction_female = (mouse_body_mass / human_body_female) * (female_mass_sum / mouse_mass_sum)
        remainder_correction_male = (mouse_body_mass / human_body_male) * (male_mass_sum / mouse_mass_sum)
        
        for organ in self.disintegrations_all_organs.index:
            if organ != 'Remainder Body':
                rMSF_female = rMSF_data.loc[organ, f'rMSF_F_{mouse_mass}']
                rMSF_male = rMSF_data.loc[organ, f'rMSF_M_{mouse_mass}']

                self.disintegrations_all_organs.loc[organ, 'h Female'] *= rMSF_female
                self.disintegrations_all_organs.loc[organ, 'h Male'] *= rMSF_male
            elif organ == 'Remainder Body':
                self.disintegrations_all_organs.loc[organ, 'h Female'] *= remainder_correction_female
                self.disintegrations_all_organs.loc[organ, 'h Male'] *=  remainder_correction_male
                


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
        

    def scale_biexponential_tiac(self, row, biol_lambda_SF = 0.25):
        
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

        wb_m = self.wb_m
        k_b_male = (73000 / wb_m) ** biol_lambda_SF
        k_b_female = (60000 / wb_m) ** biol_lambda_SF

        TIAC_h_bi_male = ((Cm_organ_t0_1 ) / (((k_b_male)**(-1)*lambda_biological1 + lambda_physical)) ) + ( (Cm_organ_t0_2 ) / ((k_b_male)**(-1)*(lambda_biological2) + lambda_physical))
        
        TIAC_h_bi_female = ((Cm_organ_t0_1 ) / (((k_b_female)**(-1)*lambda_biological1 + lambda_physical)) ) + ( (Cm_organ_t0_2 ) / ((k_b_female)**(-1)*(lambda_biological2) + lambda_physical))
        
        return TIAC_h_bi_male, TIAC_h_bi_female


    def scale_monoexponential_tiac(self, row, biol_lambda_SF = 0.25):

        lambda_effective = row['mono_exp:lambda_effective_1/h']
        lambda_physical = log(2) / self.half_life #1/h
        
        lambda_biological = lambda_effective - lambda_physical
        
        Cm_organ_t0 = row['mono_exp:%ID'] / 100 
        
        wb_m = self.wb_m
        k_b_male = (73000 / wb_m) ** biol_lambda_SF
        
        k_b_female = (60000 / wb_m) ** biol_lambda_SF

        TIAC_h_mono_male = Cm_organ_t0 / (k_b_male**(-1) * (lambda_biological) + lambda_physical)
        TIAC_h_mono_female = Cm_organ_t0 / (k_b_female**(-1) * (lambda_biological) + lambda_physical)

        return TIAC_h_mono_male, TIAC_h_mono_female


    def lambda_biological_scaling(self, biol_lambda_SF = 0.25, tumor_name=None):
        
        print("At this point we are ignoring the tumor")
        if tumor_name:
            self.not_inphantom_notumor=[org for org in self.not_inphantom if tumor_name not in org]
            tumortemp = self.biodi.loc[tumor_name]
        else:
            self.not_inphantom_notumor=self.not_inphantom
        self.not_inphantom_notumor
        print(self.phantom)
        print('Disintegrations\n')


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
                    print(org)
                    self.fitting_parameters.loc[org, 'mono_exp:%ID'] = self.fitting_parameters.loc[org, 'mono_exp:%ID/g'] * self.literature_mass.loc[org, self.mouse_mass]
                    self.fitting_parameters.loc[org, 'bi_exp1:%ID'] = self.fitting_parameters.loc[org, 'bi_exp:1_%ID/g'] * self.literature_mass.loc[org, self.mouse_mass]
                    self.fitting_parameters.loc[org, 'bi_exp2:%ID'] = self.fitting_parameters.loc[org, 'bi_exp:2_%ID/g'] * self.literature_mass.loc[org, self.mouse_mass]
                else:
                    pass
        print(self.fitting_parameters)
        if 'Tail' in self.fitting_parameters.index:
            self.fitting_parameters = self.fitting_parameters.drop('Tail', axis=0)
            
        if 'Tumor' in self.fitting_parameters.index:
            self.fitting_parameters = self.fitting_parameters.drop('Tumor', axis=0)
        
        for i, organ in enumerate(self.fitting_parameters.index):
            if self.fitting_parameters.loc[organ, 'fit_accepted'] == 1.0:
                TIAC_h_mono_male, TIAC_h_mono_female = self.scale_monoexponential_tiac(self.fitting_parameters.iloc[i], biol_lambda_SF)
                self.fitting_parameters.loc[organ, 'h Male'] = TIAC_h_mono_male
                self.fitting_parameters.loc[organ, 'h Female'] = TIAC_h_mono_female

            elif self.fitting_parameters.loc[organ, 'fit_accepted'] == 2.0:
                TIAC_h_bi_male, TIAC_h_bi_female = self.scale_biexponential_tiac(self.fitting_parameters.iloc[i], biol_lambda_SF)
                self.fitting_parameters.loc[organ, 'h Male'] = TIAC_h_bi_male
                self.fitting_parameters.loc[organ, 'h Female'] = TIAC_h_bi_female
                
        self.disintegrations_all_organs = self.fitting_parameters[['h Female', 'h Male']]
                

