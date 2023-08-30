from pathlib import Path
import json
import numpy as np
import pandas as pd
from io import StringIO
from doodle.shared.radioactive_decay import decay_act
from doodle.shared.evaluation_metrics import perc_diff


this_dir=Path(__file__).resolve().parent.parent
ISOTOPE_DATA_FILE = Path(this_dir,"data","isotopes.json")


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

        self.db_df = {}

        with open(ISOTOPE_DATA_FILE) as f:
            self.isotope_dic = json.load(f)

        self.isotope = isotope
        self.isotope_dic = self.isotope_dic[isotope]
        self.summary = ''
        
        if 'db_dic' in kwargs:
            db_file = kwargs['db_dic']['db_file']
            sheet_names = kwargs['db_dic']['sheet_names']
            header = kwargs['db_dic']['header']
            if 'site_id' in kwargs['db_dic']:
                site_id = kwargs['db_dic']['site_id']

            if 'cal_type' in kwargs:
                cal_type = kwargs['cal_type']

            db_df = pd.read_excel(db_file,sheet_name=sheet_names,header=header)
            
            if 'calibration_data' in db_df.keys():
                cal_forms = db_df[sheet_names[0]]

                #convert columns of date and time to datetime
                cols = ['measurement_datetime', 'ref_time']

                for c in cols:
                    cal_forms[c] = pd.to_datetime(cal_forms[c], format='%Y%m%d %H:%M')

                    self.db_df['cal_data'] = cal_forms

            else:
                cal_forms = db_df[sheet_names[0]]
                ref_shipped = db_df[sheet_names[1]]

                #convert columns of date and time to datetime
                cal_forms['measurement_datetime'] = pd.to_datetime(cal_forms['measurement_datetime'], format='%Y%m%d %H:%M')
                ref_shipped['ref_datetime'] = pd.to_datetime(ref_shipped['ref_datetime'], format='%Y%m%d %H:%M')

                # only look at the center being qualified and the type of calibration necessary
                if 'site_id' in kwargs['db_dic']: 
                    self.db_df['cal_data'] = cal_forms[(cal_forms.site_id == site_id) & (cal_forms.cal_type == cal_type)]
                    self.db_df['shipped_data'] = ref_shipped[(ref_shipped.site_id == site_id)]
                else:
                    self.db_df['cal_data'] = cal_forms
                    self.db_df['shipped_data'] = ref_shipped

    
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
        #self.summary_df.style.set_properties(subset=[cols],**{'width': '500px'},**{'text-align': 'left'}).hide_index() # https://stackoverflow.com/questions/55051920/pandas-styler-object-has-no-attribute-hide-index-error
        self.summary_df.style.set_properties(subset=[cols],**{'width': '500px'},**{'text-align': 'left'}).hide()
        # print(self.summary)


    def update_db(self,syringe_name='syringe_20_mL'):
        sources = self.db_df['shipped_data'].source_id.unique()
        centres = self.db_df['shipped_data'].site_id.unique()

        
        for s in sources:
            for c in centres:
                #find the reference time of the shipped source
                ref_act = self.db_df['shipped_data'][(self.db_df['shipped_data'].source_id == s) & (self.db_df['shipped_data'].site_id == c)]['A_ref_MBq']
                ref_time = self.db_df['shipped_data'][(self.db_df['shipped_data'].source_id == s) & (self.db_df['shipped_data'].site_id == c)]['ref_datetime'] 
                # print(self.db_df['cal_data'].loc[(self.db_df['cal_data'].source_id == s) &  (self.db_df['shipped_data'].site_id == c)])
               
                self.db_df['cal_data'].loc[((self.db_df['cal_data'].source_id == s) &  (self.db_df['cal_data'].site_id == c)),'ref_act_MBq'] = ref_act.values[0]
                self.db_df['cal_data'].loc[((self.db_df['cal_data'].source_id == s) &  (self.db_df['cal_data'].site_id == c)),'ref_time'] = ref_time.values[0]   
            
     
        # calculate the delta time of the calibration of the shipped source and the measurement
        self.db_df['cal_data']['delta_t'] = (self.db_df['cal_data']['measurement_datetime'] - self.db_df['cal_data']['ref_time']) / np.timedelta64(1,'D')
        
        self.db_df['cal_data']['decayed_ref'] = np.nan

        # decay correct reference sources to measurement time series
        self.db_df['cal_data'].decayed_ref = self.db_df['cal_data'].apply(lambda row: decay_act(row.ref_act_MBq,row.delta_t,self.isotope_dic['half_life']),axis=1)

        # # Check that decay has been applied correctly. calculate perc_diff
        self.db_df['cal_data']['decay_perc_diff'] = perc_diff(self.db_df['cal_data']['Ae_MBq'],self.db_df['cal_data'].decayed_ref)

        # #check recovery with the calculated decayed activity
        self.db_df['cal_data']['recovery_calculated'] = (self.db_df['cal_data'].Am_MBq / self.db_df['cal_data']['decayed_ref'] *100).round(1)

        
        #check the syringe
        self.db_df['cal_data'].loc[self.db_df['cal_data'].source_id == syringe_name,'syringe_activity_calculated'] = self.db_df['cal_data']['Ai_syr_MBq'] - self.db_df['cal_data']['Af_syr_MBq']
        self.db_df['cal_data'].loc[self.db_df['cal_data'].source_id == syringe_name,'recovery_calculated'] = (self.db_df['cal_data']['Am_syr_MBq'] / self.db_df['cal_data']['syringe_activity_calculated'] * 100).round(1)

        # reorder columns
        cols = ['site_id', 'department', 'city', 'performed_by', 'cal_type','manufacturer', 'model', 'collimator', 'initial_ cal_num', 'source_id','measurement_datetime', 'time_zone', 'ref_time','ref_act_MBq', 'delta_t','decayed_ref', 'Ae_MBq', 'decay_perc_diff', 'Am_MBq', 'Ai_syr_MBq','Af_syr_MBq', 'As_syr_MBq','syringe_activity_calculated', 'Am_syr_MBq', 'reported_recovery','recovery_calculated','final_cal_num', 'comments']

        self.db_df['cal_data'] = self.db_df['cal_data'][cols]