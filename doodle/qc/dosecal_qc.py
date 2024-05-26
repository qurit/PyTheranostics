from doodle.qc.qc import QC
from doodle.shared.radioactive_decay import decay_act
from doodle.shared.evaluation_metrics import perc_diff

import pandas as pd
import numpy as np

class DosecalQC(QC):
    
    def __init__(self,isotope,db_dic,cal_type='dc'):
        super().__init__(isotope,db_dic=db_dic,cal_type=cal_type)
                

    def check_calibration(self,accepted_percent=1.5,accepted_recovery=(97,103)):

        # keep a flag to accept or reject depending on the different tests. Default is to accept (1):
        # If something fails it will be changed to 2 if needs to verify and 3 if it completely fails

        self.accepted_calibration = 1

        self.append_to_summary(f'QC for Dose Calibrator Calibration of {self.isotope}:\t \n\n')

        #check that the expected activity matches the decay of the:
        
        #  Shipped Sources
        self.check_source_decay(accepted_percent=accepted_percent)

        # 20 ml syringe
        self.check_syringe_recovery()

        # check if reported recovery matches the calculated one and if they fall within the approved range
 
        if (self.db_df['cal_data']['recovery_calculated'] == self.db_df['cal_data']['reported_recovery']).all(axis=None):
            if ((accepted_recovery[0] <= self.db_df['cal_data']['recovery_calculated']) & (accepted_recovery[1] >= self.db_df['cal_data']['recovery_calculated'])).all(axis=None):
                self.append_to_summary(f'The reported recovery matches the calculated recovery and falls within {accepted_recovery[0]} % and {accepted_recovery[1]} %.\t            OK\n\n')

            else:
                self.accepted_calibration = 3
                self.append_to_summary(f'The reported recovery matches the calculated recovery but is out of the acceptable range of {accepted_recovery[0]} % to {accepted_recovery[1]} %.\t             FAIL\n')
                self.append_to_summary(f'\nPlease see below:\t \n\n')
                mismatch_df = self.db_df['cal_data'][self.db_df['cal_data']['recovery_calculated'] != self.db_df['cal_data']['reported_recovery']]
                cols_show = ['performed_by','manufacturer','model','source_id','ref_act_MBq','decayed_ref','Ai_syr_MBq','Af_syr_MBq','As_syr_MBq','Am_syr_MBq','reported_recovery','recovery_calculated']
                self.append_to_summary(f"{mismatch_df[cols_show].to_string()}\t ")

        else:        
            if ((accepted_recovery[0] <= self.db_df['cal_data']['recovery_calculated']) & (accepted_recovery[1] >= self.db_df['cal_data']['recovery_calculated'])).all(axis=None):
                self.accepted_calibration = 2
                self.append_to_summary(f'One or more reported recovery does not match the calculated recovery. However, the calculated recovery is within the accepted range of {accepted_recovery[0]} % and {accepted_recovery[1]} %.\t             VERIFY\n')
                # self.append_to_summary(f'Please see below:\t \n\n')
            else:
                self.accepted_calibration = 3
                self.append_to_summary(f'One or more reported recovery does not match the calculated recovery and the calculated recovery is out of the acceptable range of {accepted_recovery[0]} % to {accepted_recovery[1]} %.\t             FAIL\n')
                # self.append_to_summary(f'Please see below:\t \n\n')
            # print(self.db_df['cal_data'].columns)
            mismatch_df = self.db_df['cal_data'][self.db_df['cal_data']['recovery_calculated'] != self.db_df['cal_data']['reported_recovery']]
            cols_show = ['manufacturer','model','source_id','delta_t','ref_act_MBq','decayed_ref','Ae_MBq','decay_perc_diff','Am_MBq','Ai_syr_MBq','Af_syr_MBq','As_syr_MBq','Am_syr_MBq','reported_recovery','recovery_calculated', 'syringe_activity_calculated']
            #print(f"\n\n{mismatch_df[cols_show]}")
            print(f"\n\n{self.db_df['cal_data'][cols_show]}")


        # if self.accepted_calibration == 1:
            # self.append_to_summary(f"The following are the accepted calibration numbers for the dose calibrators submitted:\n\n")
        self.accepted_calnum_df = self.db_df['cal_data'][['manufacturer','model','source_id','final_cal_num']].groupby(['manufacturer','model','source_id']).sum()
            # self.append_to_summary(f"{summary_df.to_string()}")

        self.print_summary()


    def check_source_decay(self,accepted_percent):
         # find the shipped sources
        sources = self.db_df['shipped_data'].source_id.unique()

        for s in sources:
            #find the reference time of the shipped source
            ref_act = self.db_df['shipped_data'][self.db_df['shipped_data'].source_id == s]['A_ref_MBq']
            ref_time = self.db_df['shipped_data'][self.db_df['shipped_data'].source_id == s]['ref_datetime']
            
         
            # calculate the delta time of the calibration of the shipped source and the measurement
            self.db_df['cal_data'].loc[self.db_df['cal_data'].source_id == s,'delta_t'] = (self.db_df['cal_data'][self.db_df['cal_data']['source_id']==s]['measurement_datetime'] - ref_time.values[0]) / np.timedelta64(1,'D')
            self.db_df['cal_data'].loc[self.db_df['cal_data'].source_id == s,'ref_act_MBq'] = ref_act.values[0]


        self.db_df['cal_data']['decayed_ref'] = np.nan

        # decay correct reference sources to measurement time series
        self.db_df['cal_data'].decayed_ref = self.db_df['cal_data'].apply(lambda row: decay_act(row.ref_act_MBq,row.delta_t,self.isotope_dic['half_life']),axis=1)

        # Check that decay has been applied correctly. calculate perc_diff
        self.db_df['cal_data']['decay_perc_diff'] = perc_diff(self.db_df['cal_data']['Ae_MBq'],self.db_df['cal_data'].decayed_ref)

        #check recovery with the calculated decayed activity
        self.db_df['cal_data']['recovery_calculated'] = (self.db_df['cal_data'].Am_MBq / self.db_df['cal_data']['decayed_ref'] *100).round(1)

        # check if any perc_diff are higher than accepted
        if (self.db_df['cal_data']['decay_perc_diff'].abs()>accepted_percent).any(axis=None):
            self.accepted_calibration = 2
            self.append_to_summary(f'One or more of the sources decay correction are off by more than {accepted_percent} %.\t      MISMATCH\n\n')
            # print(f"{self.db_df['cal_data'][['source_id','measurement_datetime','delta_t','ref_act_MBq','Am_MBq','Ae_MBq','decayed_ref','decay_perc_diff']].to_string()}")

        else:
            self.append_to_summary(f'All the sources decay correction are within {accepted_percent} % of the expected.\t                               OK\n\n')

        
    def check_syringe_recovery(self,syringe_name='syringe_20_mL'):
       
        self.db_df['cal_data'].loc[self.db_df['cal_data'].source_id == syringe_name,'syringe_activity_calculated'] = self.db_df['cal_data']['Ai_syr_MBq'] - self.db_df['cal_data']['Af_syr_MBq']
        self.db_df['cal_data'].loc[self.db_df['cal_data'].source_id == syringe_name,'recovery_calculated'] = (self.db_df['cal_data']['Am_syr_MBq'] / self.db_df['cal_data']['syringe_activity_calculated'] * 100).round(1)
        
        # check if the expected activity in syringe is correct
        syr_df = self.db_df['cal_data'][self.db_df['cal_data'].source_id == syringe_name]
       

        if (syr_df.syringe_activity_calculated == syr_df.As_syr_MBq).all(axis=None):
            self.append_to_summary(f'The reported activity expected in the syringe matches the calculation.\t                           OK\n\n')
        else:
            self.accepted_calibration = 2
            self.append_to_summary(f'The reported activity expected in the syringe does not match the calculation.\t                MISMATCH\n\n')
            # mismatch_df = syr_df[syr_df['syringe_activity_calculated'] != syr_df['As_syr_MBq']]
            # cols_show = ['performed_by','manufacturer','model','source_id','Ai_syr_MBq','Af_syr_MBq','As_syr_MBq','syringe_activity_calculated']
            # self.append_to_summary(f"{mismatch_df[cols_show].to_string()} \n\n")