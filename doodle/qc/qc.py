from pathlib import Path
import json
import numpy as np
import pandas as pd
from io import StringIO


this_dir=Path(__file__).resolve().parent.parent
ISOTOPE_DATA_FILE = Path(this_dir,"isotope_data","isotopes.json")


class QC:
    def __init__(self,isotope,**kwargs):

        '''
        
        **kwargs:
            db_dic: containing three keys
                    db_file
                    sheet_names (list)
                    header 
                    site_id       
        '''

        with open(ISOTOPE_DATA_FILE) as f:
            self.isotope_dic = json.load(f)

        self.isotope = isotope
        self.isotope_dic = self.isotope_dic[isotope]
        self.summary = ''
        
        if 'db_dic' in kwargs:
            db_file = kwargs['db_dic']['db_file']
            sheet_names = kwargs['db_dic']['sheet_names']
            header = kwargs['db_dic']['header']
            site_id = kwargs['db_dic']['site_id']

            cal_type = kwargs['cal_type']

            db_df = pd.read_excel(db_file,sheet_name=sheet_names,header=header)

            cal_forms = db_df[sheet_names[0]]
            ref_shipped = db_df[sheet_names[1]]

             #convert columns of date and time to datetime
            cal_forms['measurement_datetime'] = pd.to_datetime(cal_forms['measurement_datetime'], format='%Y%m%d %H:%M')
            ref_shipped['ref_datetime'] = pd.to_datetime(ref_shipped['ref_datetime'], format='%Y%m%d %H:%M')

            # only look at the center being qualified and the type of calibration necessary
            self.db_df['cal_data'] = cal_forms[(cal_forms.site_id == site_id) & (cal_forms.cal_type == cal_type)]
            self.db_df['shipped_data'] = ref_shipped[(ref_shipped.site_id == site_id)]

    
    def window_check(self,win_perdiff_max=2,type='planar'):

        if type == 'planar':
            ds = self.ds
        elif type == 'spect':
            ds = self.recon_ds
        elif type == 'raw':
            ds = self.proj_ds

        win = []
        for i in range(len(ds.EnergyWindowInformationSequence)):
            low = ds.EnergyWindowInformationSequence[i].EnergyWindowRangeSequence[0].EnergyWindowLowerLimit
            upper = ds.EnergyWindowInformationSequence[i].EnergyWindowRangeSequence[0].EnergyWindowUpperLimit
            center = (low + upper) / 2

            win.append((low,center,upper))

        # check if the expected windows are found in the image being analyzed
        win_check = {}
        # first find if the centers correspond to the different expected windows and labeled those windows accordingly
        for el in win:
            for k,w in self.isotope_dic['windows_kev'].items():
                if int(el[1]) in range(round(w[0]),round(w[2])):
                    win_check[k] = el

        # find which expected windows are not in the current dataset, and which windows are in both
        missing_win_keys = list(set(self.isotope_dic['windows_kev'].keys()) - set(win_check))
        common_win_keys = set(self.isotope_dic['windows_kev'].keys()).intersection(set(win_check))
        
        if type != 'spect':
            if missing_win_keys:
                self.append_to_summary(f'The dataset is missing a window for {missing_win_keys}.\tMISMATCH\n')
            else:
                self.append_to_summary('The dataset contains all the expected energy windows.\tOK\n')
        else:
            if 'photopeak' in common_win_keys:
                self.append_to_summary(f"The reconstructed image is for the correct photopeak, centered at {win_check['photopeak'][1]} keV.\tOK\n")


        # find differences between common windows
        win_perc_dif = {}
        for k in common_win_keys:
            expected = np.array(self.isotope_dic['windows_kev'][k])
            configured = np.array(win_check[k])

            win_perc_dif[k] = np.round((configured - expected) / expected * 100,2)

              
        win_perc_dif.update(dict.fromkeys(missing_win_keys,np.nan))

        protocol_df = pd.DataFrame.from_dict(self.isotope_dic['windows_kev']).T

        win_df = pd.DataFrame.from_dict(win_check).T

        perc_diff_df = pd.DataFrame.from_dict(win_perc_dif).T
        
        arrays = [['expected','expected','expected','configured','configured','configured','perc_diff','perc_diff','perc_diff'],['min','center','upper','min','center','upper','min','center','upper']]
        
        if type != 'spect':
            # check if any perc_diff are higher than 2%
            if (perc_diff_df.abs()>win_perdiff_max).any(axis=None):
                self.append_to_summary(f'There are differences in the energy window settings that are higher than {win_perdiff_max} %.\nPlease see below:\t       MISMATCH\n\n')
                # self.append_to_summary(f'{self.window_check_df.to_string()}')
            elif (perc_diff_df.abs()!=0).any(axis=None):
                self.append_to_summary(f'There are differences in the energy window settings but are minimal and acceptable.\t       VERIFY\n\n')
                # self.append_to_summary(f'{self.window_check_df.to_string()}')
            else:
                self.append_to_summary(f'The energy window settings are set as expected.\t       OK\n\n')
                # self.append_to_summary(f'{self.window_check_df.to_string()}')

        window_check_df = pd.concat([protocol_df,win_df, perc_diff_df],axis=1)
        window_check_df.columns = pd.MultiIndex.from_arrays(arrays)

        if type == 'spect':
            window_check_df.dropna(inplace=True)

        return window_check_df
    
    
    def append_to_summary(self,text):
        self.summary = self.summary + text
    
    def print_summary(self):
        # print(self.summary)
        summary = StringIO(self.summary)
        self.summary_df = pd.read_csv(summary,sep='\t')
        # print(self.summary_df.columns[0])
        cols = self.summary_df.columns[0]
        # print(cols)
        self.summary_df.style.set_properties(subset=[cols],**{'width': '500px'},**{'text-align': 'left'}).hide_index()
        # print(self.summary)