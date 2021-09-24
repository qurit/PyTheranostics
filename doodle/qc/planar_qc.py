from doodle.qc.qc import QC
import pydicom
import numpy as np
import pandas as pd

class PlanarQC(QC):

    def __init__(self,isotope,dicomfile,db_dic,cal_type='planar'):
        self.db_df = {}
        super().__init__(isotope,db_dic=db_dic,cal_type=cal_type)
        self.ds = pydicom.dcmread(dicomfile)


    def check_windows_energy(self):        

        self.append_to_summary(f'QC for planar scan of {self.isotope}:\n\n')

        self.check_camera_parameters()

        

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
        self.append_to_summary(f'Scan performed on {acquisition_date} at {acquisition_time}\n\n')

        #check duration
        if round(duration) == self.isotope_dic['planar']['duration']:
            self.append_to_summary(f'SCAN DURATION: {round(duration)} minutes       OK\n\n')
        else:
            self.append_to_summary(f'SCAN DURATION: {duration} minutes     MISMATCH (should be {self.isotope_dic["planar"]["duration"]} mins)\n\n')
                
        #check windows
        self.window_check_df = self.window_check(type='planar') 

        #check collimator
        accepted_collimators = self.isotope_dic['accepted_collimators']
        if (len(self.db_df['cal_data']['collimator'].unique()) == 1) & (self.db_df['cal_data']['collimator'].unique()[0] in accepted_collimators):
            self.append_to_summary(f"COLLIMATOR: {self.db_df['cal_data']['collimator'].unique()[0]}         OK\n\n")
        else:
            self.append_to_summary(f"COLLIMATOR: {self.db_df['cal_data']['collimator'].unique()[0]} is not in the accepted collimator list.         VERIFY\n\n")

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