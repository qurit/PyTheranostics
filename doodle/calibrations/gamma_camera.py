from doodle.qc.planar_qc import PlanarQC
from doodle.plots.plots import ewin_montage
from doodle.shared.corrections import tew_scatt
from doodle.shared.radioactive_decay import decay_act

from pathlib import Path
import json
from datetime import datetime


# import pydicom
import numpy as np
import pprint
import math

this_dir=Path(__file__).resolve().parent.parent
CALIBRATIONS_DATA_FILE = Path(this_dir,"data","gamma_camera_sensitivities.json")

class GammaCamera(PlanarQC):

    def __init__(self,isotope,dicomfile,db_dic,cal_type='planar'):
        super().__init__(isotope,dicomfile,db_dic=db_dic,cal_type=cal_type)
    

    def get_sensitivity(self,source_id = 'C',**kwargs):
        # ser_date = self.ds.SeriesDate
        # ser_time = self.ds.SeriesTime

        pix_space = self.ds.PixelSpacing

        #duration of scan in seconds
        acq_duration = self.ds.ActualFrameDuration/1000 

        #number of Detectors
        ndet = self.ds.NumberOfDetectors

       
        # find camera model
        if 'site_id' in kwargs:
            if kwargs['site_id'] == 'CAVA':
                if hasattr(self.ds,'DeviceSerialNumber'):
                    camera_model = 'Intevo'
                else:
                    camera_model = 'Symbia T'
            elif kwargs['site_id'] == 'CAHJ':
                if self.ds.ManufacturerModelName == 'Tandem_870_DR':
                    camera_model = 'Discovery 870'
            elif kwargs['site_id'] == 'CAGQ':
                if self.ds.ManufacturerModelName == 'Encore2':
                    camera_model = 'Intevo T6'
            elif kwargs['site_id'] == 'CAGA':
                if self.ds.ManufacturerModelName == 'Encore2':
                    camera_model = 'Intevo T6'
            elif kwargs['site_id'] == 'CAHN':
                if self.ds.ManufacturerModelName == 'Tandem_Discovery_670_ES':
                    camera_model = 'Discovery 670'
            elif kwargs['site_id'] == 'CAGH':
                if self.ds.ManufacturerModelName == 'Tandem_Discovery_670':
                    camera_model = 'Discovery 670'
            elif kwargs['site_id'] == 'CANL':
                if self.ds.ManufacturerModelName == 'Tandem_Discovery_670_Pro':
                    camera_model = 'Discovery 670'
            elif kwargs['site_id'] == 'CAMN':
                if self.ds.ManufacturerModelName == 'Tandem_Optima_640':
                    camera_model = 'Optima 640'
            elif kwargs['site_id'] == 'CAMP':
                if self.ds.ManufacturerModelName == 'Encore2':
                    camera_model = 'Symbia'
            elif kwargs['site_id'] == 'CATW':
                if self.ds.ManufacturerModelName == 'Encore2':
                    camera_model = 'Symbia T-16'
            elif kwargs['site_id'] == 'CATC':
                if self.ds.ManufacturerModelName == 'Tandem_Discovery_670':
                    camera_model = 'Discovery 670'

         # find activity of source        
        if 'site_id' in kwargs:
            df = self.db_df['cal_data']

            df2 = df[(df.site_id == kwargs['site_id']) & (df.source_id == source_id) & (df.cal_type == 'planar') & (df.model == camera_model)]
            A_ref = df2.ref_act_MBq.values[0]
            ref_time = df2.ref_time.values[0]

            acq_time = self.ds.AcquisitionTime
            acq_date = self.ds.AcquisitionDate

            if '.' in acq_time:
                acq_time = np.datetime64(datetime.strptime(f'{acq_date} {acq_time}','%Y%m%d %H%M%S.%f'))
            else:
                acq_time = np.datetime64(datetime.strptime(f'{acq_date} {acq_time}','%Y%m%d %H%M%S'))

            
            # find the time difference in days
            if self.isotope_dic['half_life_units'] == 'days':
                delta_t = (acq_time - ref_time) / np.timedelta64(1, 'D')
            else:
                print('check units of decay correction')

            
            # Decay the reference activity
            self.A_decayed = decay_act(A_ref,delta_t,self.isotope_dic['half_life'])
            print(f"The activity of the source at the time of the scan was {self.A_decayed} MBq\n")

       #deal with energy windows
        nwin = self.ds.NumberOfEnergyWindows

        ewin = {}

        img = self.ds.pixel_array

        for w in range(nwin):          
            lower = self.ds.EnergyWindowInformationSequence[w].EnergyWindowRangeSequence[0].EnergyWindowLowerLimit
            upper = self.ds.EnergyWindowInformationSequence[w].EnergyWindowRangeSequence[0].EnergyWindowUpperLimit
            center = round((upper + lower)/2,2)
            cent = str(int(center))

            ewin[cent] = {}

            ewin[cent]['lower'] = lower
            ewin[cent]['upper'] = upper
            ewin[cent]['center'] = center
            ewin[cent]['width'] = upper - lower
            ewin[cent]['counts'] = {}
         

        # get counts in each window and detector        
        for ind,i in enumerate(range(0,int(img.shape[0]),2)):
            keys = list(ewin.keys())

            ewin[keys[ind]]['counts']['Detector1'] = np.sum(img[i,:,:])
            ewin[keys[ind]]['counts']['Detector2'] = np.sum(img[i+1,:,:])
        
        
        self.win_check = {}

        # TODO: Improve this part
        for el in ewin:
            for k,w in self.isotope_dic['windows_kev'].items():
                if int(ewin[el]['center']) in range(round(w[0]),round(w[2])):
                    self.win_check[k] = ewin[el]
 

        ewin_montage(img, ewin)

        print("Info about the different energy windows and detectors:")
        pprint.pprint(self.win_check)
        print('\n')

        self.Cp = tew_scatt(self.win_check)

        print("Primary counts in each detector")
        pprint.pprint(self.Cp)     

        # print(self.A_decayed,delta_t,ref_time,acq_time)
        # Calculate sensitivity in cps/MBq
        sensitivity = {k: v / (self.A_decayed*acq_duration) for k, v in self.Cp.items()}
        sensitivity['Average'] = sum(sensitivity.values()) / len(sensitivity)

        #calculate calibration factor in units of MBq/cps
        calibration_factor = {k: 1 / v for k, v in sensitivity.items()}

        self.cal_dic = {}
        self.cal_dic[kwargs['site_id']] = {}
        self.cal_dic[kwargs['site_id']][camera_model] = {}
        self.cal_dic[kwargs['site_id']][camera_model]['manufacturer'] = df2.manufacturer.values[0]
        self.cal_dic[kwargs['site_id']][camera_model]['collimator'] = df2.collimator.values[0]
        self.cal_dic[kwargs['site_id']][camera_model]['sensitivity'] = sensitivity
        self.cal_dic[kwargs['site_id']][camera_model]['calibration_factor'] = calibration_factor

        print('\n')
        print("Calibration Results. Sensitivity in cps/MBq. Calibration factor in MBq/cps")
        pprint.pprint(self.cal_dic)

    def calculate_uncertainty(self, site_id, camera_model):
        u_prim_list = []
        for detector in ['Detector1', 'Detector2']:
            u_pw = math.sqrt(self.win_check['photopeak']['counts'][detector])
            u_sc = math.sqrt(self.win_check['low_scatter']['counts'][detector] + self.win_check['upper_scatter']['counts'][detector])
            u_prim = math.sqrt(u_pw**2 + u_sc**2)
            u_prim_list.append(u_prim)

        final_u_counts = 1/2 * math.sqrt(u_prim_list[0]**2 + u_prim_list[1]**2) + np.std(u_prim_list)
        counts = (self.Cp['Detector1'] + self.Cp['Detector2']) / 2
        u_activity = self.A_decayed * 0.03747
        u_time = 0.001
        
        
        
        uncertainty_cf = self.cal_dic[site_id][camera_model]['calibration_factor']['Average'] * math.sqrt((final_u_counts/counts)**2 + (u_activity/ self.A_decayed)**2 + (u_time/self.ds.ActualFrameDuration/1000 )**2)
        uncertainty_sensitivity = self.cal_dic[site_id][camera_model]['sensitivity']['Average'] * math.sqrt((final_u_counts/counts)**2 + (u_activity/ self.A_decayed)**2 + (u_time/self.ds.ActualFrameDuration/1000 )**2)
        
        return uncertainty_cf, uncertainty_sensitivity
        
            
    def calfactor_to_database(self,**kwargs):
        if 'site_id' in kwargs:
            site_id = kwargs['site_id']

        with open(CALIBRATIONS_DATA_FILE,'r+') as f:
            self.calfactors_dic = json.load(f)
            f.seek(0)
            if site_id in self.calfactors_dic.keys():
                self.calfactors_dic[site_id].update(self.cal_dic[site_id])
            else:
                self.calfactors_dic.update(self.cal_dic)
            json.dump(self.calfactors_dic,f,indent=2)