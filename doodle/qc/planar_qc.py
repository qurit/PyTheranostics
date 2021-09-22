from doodle.qc.qc import QC
import pydicom
import numpy as np
import pandas as pd

class PlanarQC(QC):

    def __init__(self,isotope,dicomfile):
        super().__init__(isotope)
        self.ds = pydicom.dcmread(dicomfile)


    def check_windows_energy(self,win_perdiff_max=2):        

        self.append_to_summary(f'QC for planar scan of {self.isotope}:\n\n')

        self.check_camera_parameters()


        win = []
        for i in range(len(self.ds.EnergyWindowInformationSequence)):
            low = self.ds.EnergyWindowInformationSequence[i].EnergyWindowRangeSequence[0].EnergyWindowLowerLimit
            upper = self.ds.EnergyWindowInformationSequence[i].EnergyWindowRangeSequence[0].EnergyWindowUpperLimit
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

        if missing_win_keys:
            self.append_to_summary(f'The dataset is missing a window for {missing_win_keys}     MISMATCH\n\n')
        else:
            self.append_to_summary('The dataset contains all the expected energy windows.      OK\n\n')

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
        
        self.window_check_df = pd.concat([protocol_df,win_df, perc_diff_df],axis=1)
        self.window_check_df.columns = pd.MultiIndex.from_arrays(arrays)

        # check if any perc_diff are higher than 2%
        if (perc_diff_df.abs()>win_perdiff_max).any(axis=None):
            self.append_to_summary(f'There are differences in the energy window settings that are higher than {win_perdiff_max} %.\nPlease see below:      MISMATCH\n\n')
            self.append_to_summary(f'{self.window_check_df.to_string()}')
        elif (perc_diff_df.abs()!=0).any(axis=None):
            self.append_to_summary(f'There are differences in the energy window settings but are minimal and acceptable.\nPlease see below:      VERIFY\n\n')
            self.append_to_summary(f'{self.window_check_df.to_string()}')
        else:
            self.append_to_summary(f'The energy window settings are set as expected      OK\n\n')
            self.append_to_summary(f'{self.window_check_df.to_string()}')

        self.print_summary()


    def check_camera_parameters(self):
        camera_manufacturer = self.ds.Manufacturer
        camera_model = self.ds.ManufacturerModelName
        acquisition_date = self.ds.AcquisitionDate
        acquisition_time = self.ds.AcquisitionTime
        modality = self.ds.Modality
        duration =(self.ds.ActualFrameDuration / 1000) /60
        rows = self.ds.Rows
        cols = self.ds.Columns
        zoom = self.ds.DetectorInformationSequence[0].ZoomFactor

        self.append_to_summary(f'CAMERA: {camera_manufacturer} {camera_model}\n')
        self.append_to_summary(f'MODALITY: {modality}\n')
        self.append_to_summary(f'Scan performed on {acquisition_date} at {acquisition_time}\n\n\n')

        #check duration
        if round(duration) == self.isotope_dic['planar']['duration']:
            self.append_to_summary(f'SCAN DURATION: {round(duration)} minutes       OK\n')
        else:
            self.append_to_summary(f'SCAN DURATION: {duration} minutes     MISMATCH (should be {self.isotope_dic["planar"]["duration"]} mins)\n')

        #check matrix size
        if [rows,cols] == self.isotope_dic['planar']['matrix']:
            self.append_to_summary(f'MATRIX SIZE: {rows} x {cols}         OK\n')
        else:
            self.append_to_summary(f'MATRIX SIZE: {rows} x {cols}     MISMATCH (should be {self.isotope_dic["planar"]["matrix"][0]} x {self.isotope_dic["planar"]["matrix"][1]}\n')

        #check zoom
        if zoom == self.isotope_dic['planar']['zoom']:
            self.append_to_summary(f'ZOOM: {zoom}                   OK\n\n')
        else:
            self.append_to_summary(f'ZOOM: {zoom}              MISMATCH (should be {self.isotope_dic["planar"]["zoom"]}\n\n')