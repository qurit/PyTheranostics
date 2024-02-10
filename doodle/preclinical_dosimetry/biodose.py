import pandas as pd
import numpy as np

from numpy import exp, log
from os import path,makedirs
import datetime
from copy import deepcopy
from scipy import integrate


import matplotlib.pylab as plt
from doodle.fits.fits import monoexp_fun, biexp_fun, fit_monoexp, find_a_initial, fit_biexp_uptake, fit_biexp, fit_triexp
from doodle.plots.plots import monoexp_fit_plots, biexp_fit_plots


this_dir=path.dirname(__file__)
TEMPLATE_PATH = path.join(this_dir,"olindaTemplates")
PHANTOM_PATH = path.join(this_dir,'phantomdata') # These variables use only lowercases so "PhantomData" is in lowercase


class BioDose():
    '''
    This is a class for the analysis of biodistribution data in preclinical experiments
    '''

    def __init__(self, isotope, half_life, phantom,  sex, uptake=None, timepoints=None):
        'half_life in hours'
        if uptake is None:
            uptake = []
        if timepoints is None:
            timepoints = []

        self.isotope = isotope
        self.half_life = half_life
        self.phantom = phantom
        self.uptake = uptake
        self.sex = sex
        self.t = timepoints
        self.biodi = None

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
        

        

    def curve_fits(self, organlist=None, uptake=False, maxev=100000, monoguess=(1,1), biguess=(1,1,1,1), uptakeguess=(1,1,-1,1), ignore_weights=False,append_zero=True,skip_points=0):
        ''' This method fits the curves and stores the results in self.fit_results.  The results can be seen in self.area organized in a pandas dataframe.'''

        decayconst = log(2)/self.half_life
        if organlist is None:
            self.area = pd.DataFrame(index=organlist, columns=['Mono-Exponential', 'Bi-Exponential', 'Bi-Exponential_uptake', 'Tri-Exponential', 'Perctage_diff - mono vs bi washout', 'Perctage_diff - bi vs tri uptake washout'])
            self.fit_results = {}
            organlist = self.biodi.index
            
        #self.monoexp_parameters = pd.DataFrame()
        dfs = []
        for org in organlist:
            bio_data = self.biodi.loc[org]['%ID/g']
            activity = np.asarray(bio_data)
            t = np.asarray((self.t))
            sigmas = self.biodi.loc[org]['sigma']
            sigmas = np.asarray(sigmas)
            xlabel = 't (h)'
            ylabel = '%ID/g'
            
            if uptake == False:
                popt1,tt1,yy1,residuals1 = fit_monoexp(t, activity,decayconst=decayconst,skip_points=skip_points, monoguess=monoguess,   maxev=maxev, limit_bounds=False, sigmas = sigmas) 
                monoexp_fit_plots(t, activity, tt1, yy1, org,  popt1[2], residuals1, xlabel, ylabel, skip_points, sigmas)
                popt2,tt2,yy2,residuals2 = fit_biexp(t, activity, maxev=maxev, biguess=biguess, decayconst=decayconst, ignore_weights=ignore_weights,  skip_points=skip_points, sigmas = sigmas)
                biexp_fit_plots(t, activity, tt2, yy2, org, popt2[4], residuals2, xlabel, ylabel, skip_points, sigmas)
                organ_data = {
                    'Organ': [org],
                    '%ID/g': [popt1[0]], 
                    'lambda_effective_1/h': [popt1[1]],  
                }
                # Creating a temporary DataFrame for the current organ
                organ_df = pd.DataFrame(organ_data, index=[org])
                dfs.append(organ_df)

                # Appending the temporary DataFrame to the main DataFrame
                #self.monoexp_parameters = self.monoexp_parameters.append(organ_df, ignore_index=True)

                if skip_points == 0:
                    area1 = integrate.quad(monoexp_fun, 0, np.inf, args=(popt1[0], popt1[1]))
                    area2 = integrate.quad(biexp_fun, 0, np.inf, args=(popt2[0], popt2[1], popt2[2], popt2[3]))
                elif skip_points == 1:
                    # calculate triangle on the left, made by first point and (0,0) with height the height of the first point
                    triangle_area = t[0] * bio_data.iloc[0] / 2  # base is time of first point, height is height of first point
                    trapezoid_area = (bio_data.iloc[0] + bio_data.iloc[1]) * (t[1] - t[0]) / 2  # base1 is height of first point, base2 is height of second point, height is difference in times

                    monoexp_area = integrate.quad(monoexp_fun, t[skip_points], np.inf, args=(popt1[0], popt1[1]))
                    biexp_area = integrate.quad(biexp_fun, t[skip_points], np.inf, args=(popt2[0], popt2[1], popt2[2], popt2[3]))

                    area1 = triangle_area + trapezoid_area + monoexp_area
                    area2 = triangle_area + trapezoid_area + biexp_area
                elif skip_points >= 2:
                    # calculate triangle on the left, made by first point and (0,0) with height the height of the first point
                    triangle_area = t[0] * bio_data.iloc[0] / 2  # base is time of first point, height is height of first point
                    trapezoid_area = (bio_data.iloc[0] + bio_data.iloc[1]) * (t[1] - t[0]) / 2  # base1 is height of first point, base2 is height of second point, height is difference in times
                    trapezoid_area2 = (bio_data.iloc[1] + bio_data.iloc[2]) * (t[2] - t[1]) / 2 
                    monoexp_area = integrate.quad(monoexp_fun, t[skip_points], np.inf, args=(popt1[0], popt1[1]))
                    biexp_area = integrate.quad(biexp_fun, t[skip_points], np.inf, args=(popt2[0], popt2[1], popt2[2], popt2[3]))

                    area1 = triangle_area + trapezoid_area + trapezoid_area2 + monoexp_area
                    area2 = triangle_area + trapezoid_area + trapezoid_area2 + biexp_area                    
                    
                elif skip_points >= 3:    
                    print('Error: Value must be equal to 0 or 1.') 
                
                self.fit_results[org]=[popt1,area1,popt2,area2]

                if np.isnan(area1).all():
                    self.area.loc[org,'Bi-Exponential']=area2[0]
                else:
                    self.area.loc[org,'Mono-Exponential']=area1[0]
                    self.area.loc[org,'Bi-Exponential']=area2[0]
                    self.area.loc[org,'Perctage_diff - mono vs bi washout']= abs(area1[0] - area2[0]) / area1[0] * 100     
                    #self.area.loc[org]=[area1[0],area2[0]]

            else:
                popt3,tt3,yy3,residuals3 = fit_biexp_uptake(t, activity, maxev=maxev, uptakeguess=uptakeguess, decayconst=decayconst, ignore_weights=ignore_weights,  skip_points=skip_points, sigmas = sigmas)                
                biexp_fit_plots(t, activity, tt3, yy3, org, popt3[4], residuals3, xlabel, ylabel, skip_points, sigmas)
                area3 = integrate.quad(biexp_fun, 0, np.inf, args=(popt3[0], popt3[1], popt3[2], popt3[3]))

                self.area.loc[org,'Bi-Exponential_uptake'] = area3[0]
                #self.area.loc[org,'Tri-Exponential'] = area4[0]
                #self.area.loc[org,'Perctage_diff - bi vs tri uptake washout'] = abs(area3[0] - area4[0]) / area3[0] * 100                
        
        try:
            self.monoexp_parameters = pd.concat(dfs, ignore_index=True)   
        except:
            print("")

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
        
        
    def tumor_sink_effect(self):
        not_organ = ['Tumor']
        
        # Get the disintegration value for the 'Tumor'
        tumor_value = self.disintegrations['h']['Tumor']
        
        # Calculate the sum of disintegration values for all organs except 'Remainder Body' (Organs included in the ROB are added to WB, so we don't want to add them twice)
        wb_value = self.disintegrations['h'].sum() - self.disintegrations['h']['Remainder Body'] 

        # Update disintegration values for each organ considering the tumor sink effect
        for organ in self.disintegrations['h'].index:
            if organ not in not_organ:
                self.disintegrations['h'][organ] = self.disintegrations['h'][organ] + (tumor_value * ((self.disintegrations['h'][organ]) / (wb_value - tumor_value)))
        
        # Create a deep copy of the disintegrations to avoid modifying the original data
        new_disintegrations = deepcopy(self.disintegrations)
        
        return new_disintegrations
        
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
            self.disintegrations['%ID*h']=self.disintegrations['%ID/g*h']*self.phantom_mass['25g']
            if 'Residual' in self.disintegrations.index:
                self.not_inphantom_notumor.remove('Residual')
                self.disintegrations.loc['Remainder Body', '%ID*h'] = (self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor].mul(self.literature_mass['25g'].loc[self.not_inphantom_notumor]).sum()) + (self.disintegrations.loc['Residual', '%ID/g*h'] * (self.phantom_mass.loc['Residual', '25g']))
            else:    
                self.disintegrations.loc['Remainder Body', '%ID*h'] = (self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor].mul(self.literature_mass['25g'].loc[self.not_inphantom_notumor]).sum())
            for org in self.not_inphantom_notumor:
                if org != 'Tail':
                    self.disintegrations.loc[org, '%ID*h'] = self.disintegrations.loc[org, '%ID/g*h'] * self.literature_mass.loc[org, '25g']
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
        
            
    def create_mouse25case(self, savefile=False,dirname='./',):
        
        '''This function creates a pandas dataframe that looks exactly as the case files generated by OLINDA for the 25g mouse.        
           The result can be viewed under self.mouse25case, and the pandas methods can be used to finally save it if wanted.
        '''

        template=pd.read_csv(path.join(TEMPLATE_PATH,'mouse25g.cas'))
        template.columns=['Data']

        #modify the isotope in the template
        ind=template[template['Data']=='[BEGIN NUCLIDES]'].index
        template.loc[ind[0]+1,'Data']=self.isotope + '|'
        input_organs = self.disintegrations.drop(set(self.not_inphantom).intersection(set(self.disintegrations.index))).index.to_list()
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
            kineticdata=self.disintegrations.loc[org]['h']

            if np.isnan(kineticdata):
                kineticdata=0

            template.iloc[ind[0]]=sourceorgan+'|'+massorgan+'|'+'{:7f}'.format(kineticdata)
            template.iloc[ind[uptakepos]]=sourceorgan+'|'+'{:7f}'.format(kineticdata)

        now = datetime.datetime.now()
        template.columns=['Saved on ' + now.strftime("%m.%d.%Y") +' at ' + now.strftime('%H:%M:%S')]
            
        self.mouse25case=template

        if savefile==True:
            if not path.exists(dirname):
                makedirs(dirname)

            self.mouse25case.to_csv(dirname+'/'+'mouse25g.cas',index=False)
        
        print('The case file mouse25g.cas has been saved in\n{}'.format(dirname))
        
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
    

        
    def interspecies_conversion(self,mouse_mass=25):
        organs = self.disintegrations['%ID*h Male'].index
        for organ in organs:
            print(organ)
            print(self.phantom_mass['Male'].loc[organ])
            
        
        self.disintegrations['%ID*h Male']=self.disintegrations['%ID*h Male']*(mouse_mass/self.phantom_mass['Male'].loc['Body'])
        self.disintegrations['%ID*h Female']=self.disintegrations['%ID*h Female']*(mouse_mass/self.phantom_mass['Female'].loc['Body'])

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
    
    def m2(self, mouse_mass=25, tumor_name=None):
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

    def M3(self,human_not_inphantom_notumor, tumor_name=None):
        #self.monoexp_parameters.set_index('Organ', inplace=True)
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
        

        self.literature_mass=self.literature_mass.loc[self.monoexp_parameters.index.intersection(self.literature_mass.index)]
        print('\nLiterature Mass (g)\n')
        print(self.literature_mass)

        print(self.phantom_mass)
        if 'mouse' in self.phantom.lower():
            self.monoexp_parameters['%ID']=self.monoexp_parameters['%ID/g']*self.phantom_mass['25g']
            self.monoexp_parameters.loc['Remainder Body', '%ID'] = (self.monoexp_parameters['%ID/g'].loc[self.not_inphantom_notumor].mul(self.literature_mass['25g']).sum())
            for org in self.not_inphantom_notumor:
                if org != 'Tail':
                    self.monoexp_parameters.loc[org, '%ID'] = self.monoexp_parameters.loc[org, '%ID/g'] * self.literature_mass.loc[org, '25g']
                else:
                    pass
        
        self.monoexp_parameters['fraction'] = self.monoexp_parameters['%ID'] / 100 
        self.monoexp_parameters['effective_halflife_h'] = log(2) / self.monoexp_parameters['lambda_effective_1/h']
        self.monoexp_parameters['physical_halflife_h'] = 6.647*24
        self.monoexp_parameters['biological_halflife_h'] = 1 / ((1 / self.monoexp_parameters['effective_halflife_h']) - (1 / self.monoexp_parameters['physical_halflife_h']))
        self.monoexp_parameters['lambda_physical_1/h'] = log(2) / self.monoexp_parameters['physical_halflife_h'] 
        self.monoexp_parameters['lambda_biological_1/h'] = log(2) / self.monoexp_parameters['biological_halflife_h']
        k_female = (60000 / 25) ** (0.25)
        k_male = (73000 / 25) ** (0.25)
        self.monoexp_parameters['h Female'] = (self.monoexp_parameters['fraction'] / (k_female**(-1) * self.monoexp_parameters['lambda_biological_1/h'] + self.monoexp_parameters['lambda_physical_1/h'])) 
        self.monoexp_parameters['h Male'] = (self.monoexp_parameters['fraction'] / (k_male**(-1) * self.monoexp_parameters['lambda_biological_1/h'] + self.monoexp_parameters['lambda_physical_1/h']))       

        self.monoexp_parameters.rename({'Skeleton': 'Bone Surfaces'}, inplace=True)
        self.monoexp_parameters.rename({'Blood': 'Heart Contents'}, inplace=True)

        self.monoexp_parameters.loc['Rectum', 'h Female'] = (70/360) * self.monoexp_parameters.loc['Large Intestine', 'h Female'] 
        self.monoexp_parameters.loc['Rectum', 'h Male'] = (70/370) * self.monoexp_parameters.loc['Large Intestine', 'h Male']    
        self.monoexp_parameters.loc['Left Colon', 'h Female'] = (145/360) * self.monoexp_parameters.loc['Large Intestine', 'h Female'] 
        self.monoexp_parameters.loc['Left Colon', 'h Male'] = (150/370) * self.monoexp_parameters.loc['Large Intestine', 'h Male']
        self.monoexp_parameters.loc['Right Colon', 'h Female'] = (145/360) * self.monoexp_parameters.loc['Large Intestine', 'h Female'] 
        self.monoexp_parameters.loc['Right Colon', 'h Male'] = (150/370) * self.monoexp_parameters.loc['Large Intestine', 'h Male']
        self.monoexp_parameters = self.monoexp_parameters.drop('Large Intestine', axis=0)
        self.monoexp_parameters = self.monoexp_parameters.drop('Tail', axis=0)
        self.monoexp_parameters = self.monoexp_parameters.drop('Tumor', axis=0)
        self.monoexp_parameters.loc['Red Marrow'] = 0.34 * self.monoexp_parameters.loc['Heart Contents']  

        self.monoexp_parameters.loc['Remainder Body', 'h Female'] = self.monoexp_parameters['h Female'].loc[human_not_inphantom_notumor].sum()
        self.monoexp_parameters.loc['Remainder Body', 'h Male'] =  self.monoexp_parameters['h Male'].loc[human_not_inphantom_notumor].sum()
        
        for org in human_not_inphantom_notumor:
            self.monoexp_parameters = self.monoexp_parameters.drop(org, axis=0)
        
        self.monoexp_parameters.sort_index(inplace=True)
            
def m2(human, mouse):
    try:
        mouse.not_inphantom_notumor.remove('Tail')
    except:
        print("")
    mouse.not_inphantom_notumor
    mouse_mass = 25
    human.disintegrations_m2=pd.DataFrame(index=human.in_phantom_organs,columns=['h Female', 'h Male'])
    if 'Residual' in human.not_inphantom_notumor:
        human.not_inphantom_notumor.remove('Residual')
    for org in human.in_phantom_organs:
        
        if org == 'Remainder Body':
            mouse_rof = mouse.literature_mass['25g'].loc[human.not_inphantom_notumor].sum() #we use only organs which constitute for ROB in human. for example Adrenals were ROB in mouse, but not in humans
        elif org == 'Bone Surfaces':
            mouse_mass_org = mouse.phantom_mass.loc['Skeleton', '25g']
        elif org == 'Heart Contents':
            mouse_mass_org = mouse.literature_mass.loc['Blood', '25g']
        elif org in mouse.phantom_mass.index:
            mouse_mass_org = mouse.phantom_mass.loc[org, '25g']
        elif org in mouse.literature_mass.index:
            mouse_mass_org = mouse.literature_mass.loc[org, '25g']
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

        human.disintegrations_m2.loc[org, 'h Female']=human.disintegrations.loc[org, 'h']*(mouse_mass/human.phantom_mass['Female'].loc['Body'])*(human_mass_org_female/mouse_mass_org)
        human.disintegrations_m2.loc[org, 'h Male']=human.disintegrations.loc[org, 'h']*(mouse_mass/human.phantom_mass['Male'].loc['Body'])*(human_mass_org_male/mouse_mass_org)
        if org == 'Remainder Body':
            human.disintegrations_m2.loc['Remainder Body', 'h Female'] = human.disintegrations['h'].loc[human.not_inphantom_notumor].sum()*(mouse_mass/human.phantom_mass['Female'].loc['Body'])*(human_rof_female/mouse_rof)
            human.disintegrations_m2.loc['Remainder Body', 'h Male'] = human.disintegrations['h'].loc[human.not_inphantom_notumor].sum()*(mouse_mass/human.phantom_mass['Male'].loc['Body'])*(human_rof_female/mouse_rof)

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