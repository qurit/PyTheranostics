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

        for org in organlist:
            bio_data = self.biodi.loc[org]['%ID/g']
            activity = np.asarray(bio_data)
            t = np.asarray((self.t))
            sigmas = self.biodi.loc[org]['sigma']
            sigmas = np.asarray(sigmas)
            
            if uptake == False:
                popt1,tt1,yy1,residuals1 = fit_monoexp(t, activity,decayconst=decayconst,skip_points=skip_points, monoguess=monoguess,   maxev=maxev, limit_bounds=False, sigmas = sigmas) 
                monoexp_fit_plots(t, activity, tt1, yy1, org, popt1[2], residuals1, skip_points, sigmas)
                popt2,tt2,yy2,residuals2 = fit_biexp(t, activity, maxev=maxev, biguess=biguess, decayconst=decayconst, ignore_weights=ignore_weights,  skip_points=skip_points, sigmas = sigmas)
                biexp_fit_plots(t, activity, tt2, yy2, org, popt2[4], residuals2, skip_points, sigmas)
                
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
                biexp_fit_plots(t, activity, tt3, yy3, org, popt3[4], residuals3, skip_points, sigmas)
                area3 = integrate.quad(biexp_fun, 0, np.inf, args=(popt3[0], popt3[1], popt3[2], popt3[3]))

                self.area.loc[org,'Bi-Exponential_uptake'] = area3[0]
                #self.area.loc[org,'Tri-Exponential'] = area4[0]
                #self.area.loc[org,'Perctage_diff - bi vs tri uptake washout'] = abs(area3[0] - area4[0]) / area3[0] * 100                
            

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
        print(type(self.disintegrations['%ID*h']))
        not_organ = ['Tumor']
        tumor_value = self.disintegrations['%ID*h']['Tumor']
        wb_value = self.disintegrations['%ID*h'].sum()

        for index_value in self.disintegrations['%ID*h'].index:
            if index_value not in not_organ:
                self.disintegrations['%ID*h'][index_value] = self.disintegrations['%ID*h'][index_value] + (tumor_value * ((self.disintegrations['%ID*h'][index_value]) / (wb_value - tumor_value)))

        
    def phantom_data(self):
        print(PHANTOM_PATH)
        if 'mouse' in self.phantom.lower():
            self.phantom_mass = pd.read_csv(path.join(PHANTOM_PATH,'mouse_phantom_masses.csv'))  # TODO: CHANGE PATH 
        elif 'human' in self.phantom.lower():
            self.phantom_mass = pd.read_csv(path.join(PHANTOM_PATH,'human_phantom_masses.csv'))
        self.phantom_mass.set_index('Organ',inplace=True)
        self.phantom_mass.sort_index(inplace=True)
        
        self.not_inphantom=[]
        for org in self.biodi.index: 
            if org not in self.phantom_mass.index:            
                self.not_inphantom.append(org)
        print('These organs from the biodi are not modelled in the phantom\n{}'.format(self.not_inphantom))
        
        self.phantom_mass.loc['Remainder Body']=(self.phantom_mass.loc['Body']-self.phantom_mass.loc[self.phantom_mass.index!='Body'].sum())

        self.not_inbiodi=[]
        for org in self.phantom_mass.index: 
            if org not in self.biodi.index and org != 'Body' and org != 'Remainder Body':            
                self.not_inbiodi.append(org)
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
            literature_mass = pd.read_csv(path.join(PHANTOM_PATH,'mouse_notinphantom_masses.csv'))  # TODO: CHANGE PATH
        
        elif 'human' in self.phantom.lower():
            literature_mass = pd.read_csv(path.join(PHANTOM_PATH,'human_notinphantom_masses.csv'))  # TODO: CHANGE PATH
            
        literature_mass.set_index('Organ',inplace=True)

        if 'mouse' in self.phantom.lower():
            literature_mass.loc['Muscle'] = self.phantom_mass.loc['Remainder Body']-literature_mass.sum()
        

        literature_mass=literature_mass.loc[self.disintegrations.index.intersection(literature_mass.index)]
        print('\nLiterature Mass (g)\n')
        print(literature_mass)

        print(self.phantom_mass)
        if 'mouse' in self.phantom.lower():
            self.disintegrations['%ID*h']=self.disintegrations['%ID/g*h']*self.phantom_mass['25g']
            self.disintegrations.loc['Remainder Body', '%ID*h'] = (self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor].mul(literature_mass['25g']).sum())
            for org in self.not_inphantom_notumor:
                if org != 'Tail':
                    self.disintegrations.loc[org, '%ID*h'] = self.disintegrations.loc[org, '%ID/g*h'] * literature_mass.loc[org, '25g']
                else:
                    pass
                
                
        elif 'human' in self.phantom.lower():
            self.disintegrations['%ID*h Female']=self.disintegrations['%ID/g*h']*self.phantom_mass['Female']
            self.disintegrations['%ID*h Male']=self.disintegrations['%ID/g*h']*self.phantom_mass['Male']
            
            self.disintegrations.loc['Remainder Body', '%ID*h Female'] = (self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor].mul(literature_mass['Female']).sum())
            self.disintegrations.loc['Remainder Body', '%ID*h Male'] = (self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor].mul(literature_mass['Male']).sum())
            #self.disintegrations.loc['Remainder Body']['%ID*h Female']=(self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor]*literature_mass['Female']).sum()
            #self.disintegrations.loc['Remainder Body']['%ID*h Male']=(self.disintegrations['%ID/g*h'].loc[self.not_inphantom_notumor]*literature_mass['Male']).sum()
        

        #self.disintegrations.drop(not_inphantom_notumor,inplace=True)
        self.disintegrations=self.disintegrations/100
        
    def not_inphantom_notumor_fun(self):
        self.disintegrations.drop(self.not_inphantom_notumor,inplace=True)
            
    def add_tumor_mass(self,tumor_name,tumor_mass):
        self.phantom_mass.loc[tumor_name]=tumor_mass  #grams   Provided by average of biodi from Etienne
        
            
    def create_mouse25case(self,savefile=False,dirname='./',):
        
        '''This function creates a pandas dataframe that looks exactly as the case files generated by OLINDA for the 25g mouse.        
           The result can be viewed under self.mouse25case, and the pandas methods can be used to finally save it if wanted.
        '''

        template=pd.read_csv(path.join(TEMPLATE_PATH,'mouse25g.cas'))
        template.columns=['Data']

        #modify the isotope in the template
        ind=template[template['Data']=='[BEGIN NUCLIDES]'].index
        template.loc[ind[0]+1,'Data']=self.isotope + '|'

        #change the kinetics for each input organ
        for org in self.disintegrations.drop(set(self.not_inphantom).intersection(set(self.disintegrations.index))).index:  #ignore the tumor here
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
            sourceorgan=template.iloc[ind[0]].str.split('|')[0][0]
            massorgan=template.iloc[ind[0]].str.split('|')[0][1]
            kineticdata=self.disintegrations.loc[org]['%ID*h']
            
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
        ind_list=self.biodi.index.tolist()
        ind_pos = ind_list.index(oldname)
        ind_list[ind_pos] = newname
        self.biodi.index = ind_list
        
    def heart_contents(self):
        self.biodi.loc['Red Marrow'] = self.biodi.loc['Heart Contents']*0.34
        self.biodi.sort_index(inplace=True)

    def humanize_biodi(self,small_intestine=False):
        ''' small_intestine: False if it is already in the biodi.
                            True to assume same as Large intestine '''

        human = deepcopy(self)

        human.phantom='AdultHuman'

        human.biodi.loc['Small Intestine'] = human.biodi.loc['Large Intestine']
        human.biodi.sort_index(inplace=True)

        human.rename_organ('Skeleton','Bone Surfaces')
        human.rename_organ('Large Intestine','Left Colon')
        human.rename_organ('Blood','Heart Contents')

        human.biodi.loc['Right Colon'] = human.biodi.loc['Left Colon']
        human.biodi.loc['Rectum'] = human.biodi.loc['Left Colon']

        human.heart_contents()
        print(human.biodi)
        tempfitresults=deepcopy(human.fit_results)
        fit_results={}
        fit_accepted={}
        for key in human.fit_results:
            if key == 'Skeleton':
                fit_results['Bone Surfaces']=tempfitresults[key]
                fit_accepted['Bone Surfaces']=human.fit_accepted[key]
            elif key == 'Large Intestine':
                fit_results['Left Colon']=tempfitresults[key]
                fit_accepted['Left Colon']=human.fit_accepted[key]

                fit_results['Right Colon']=tempfitresults[key]
                fit_accepted['Right Colon']=human.fit_accepted[key]

                fit_results['Rectum']=tempfitresults[key]
                fit_accepted['Rectum']=human.fit_accepted[key]

                if small_intestine:
                    fit_results['Small Intestine']=tempfitresults[key]
                    fit_accepted['Small Intestine']=human.fit_accepted[key]

            elif key == 'Blood':
                fit_results['Heart Contents']=tempfitresults[key]
                fit_accepted['Heart Contents']=human.fit_accepted[key]

                fit_results['Red Marrow']=deepcopy(tempfitresults[key])
                fit_results['Red Marrow'][0]=fit_results['Red Marrow'][0]*0.34
                fit_results['Red Marrow'][1]=tuple([0.34*x for x in list(fit_results['Red Marrow'][1])])
                fit_results['Red Marrow'][2]=fit_results['Red Marrow'][2]*0.34
                fit_results['Red Marrow'][3]=tuple([0.34*x for x in list(fit_results['Red Marrow'][3])])
                fit_accepted['Red Marrow']=human.fit_accepted[key]

                
            else:
                fit_results[key]=tempfitresults[key]
                fit_accepted[key]=human.fit_accepted[key]

        human.fit_results=fit_results
        human.fit_accepted=fit_accepted

        human.area.index = human.area.index.str.replace('Skeleton','Bone Surfaces')
        human.area.index = human.area.index.str.replace('Large Intestine','Left Colon')
        human.area.index = human.area.index.str.replace('Blood','Heart Contents')
        return human
    
    
    def interspecies_conversion(self,mouse_mass=25):
        organs = self.disintegrations['%ID*h Male'].index
        for organ in organs:
            print(organ)
            print(self.phantom_mass['Male'].loc[organ])
            
        
        self.disintegrations['%ID*h Male']=self.disintegrations['%ID*h Male']*(mouse_mass/self.phantom_mass['Male'].loc['Body'])
        self.disintegrations['%ID*h Female']=self.disintegrations['%ID*h Female']*(mouse_mass/self.phantom_mass['Female'].loc['Body'])




    def create_humancase(self,savefile=False,dirname='./'):
        
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
        for org in self.disintegrations.drop(set(self.not_inphantom).intersection(set(self.disintegrations.index))).index:  #ignore the tumor here
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
            kineticdata=self.disintegrations.loc[org]['%ID*h '+self.sex]
            
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

            self.humancase.to_csv(dirname+'/'+self.sex+'.cas',index=False)
        
        print('The case file {} has been saved in\n{}'.format(self.sex+'.cas',dirname))



