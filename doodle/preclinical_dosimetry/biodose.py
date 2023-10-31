import pandas as pd
import numpy as np

from numpy import exp, log
from fits import tiac_organ
from os import path,makedirs
import datetime
from copy import deepcopy

#import matplotlib.pylab as plt
#from doodle.fits.fits import monoexp_fun, fit_monoexp, find_a_initial, fit_biexp_uptake, fit_biexp, fit_triexp
#from doodle.plots.plots import monoexp_fit_plots, biexp_fit_plots

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
        
        
    def read_df(self, df: pd.DataFrame):
        """
        Reads the a df from biodi as input
        :param df: df
        :return: none
        """
        df = df.copy(deep=True)
        df = df.swaplevel(axis=1)
        df = df.sort_index(axis=1)
        df = df.drop('count', axis=1)

        self.t = np.array(df.columns.levels[1])
        decay_factor = exp(-log(2) / self.half_life * self.t)

        df['mean'] = df['mean'] * decay_factor
        df['std'] = df['std'] * decay_factor
        df.rename(columns={'mean': '%ID/g', 'std': 'sigma'}, inplace=True)

        self.biodi = df


    def read_biodi(self, biodi_file):
        '''This method reads a biodi file and sets the data as a pandas dataframe in self.biodi
        '''
        print('Reading biodistribution information from the file: {}'.format(biodi_file))
        biodi = pd.read_csv(biodi_file)
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