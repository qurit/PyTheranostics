from doodle.qc.qc import QC
import pydicom
import numpy as np
import pandas as pd

class SPECTQC(QC):

    def __init__(self,isotope,projections_file,recon_file,db_dic,cal_type='spect'):
        super().__init__(isotope,db_dic=db_dic,cal_type=cal_type)
        self.proj_ds = pydicom.dcmread(projections_file)
        self.recon_ds = pydicom.dcmread(recon_file)

    
    def check_projs(self):

        self.window_check_df = {}

        self.append_to_summary(f'QC for SPECT RAW DATA of {self.isotope}:\n\n')

        # check acquisition parameters (i.e. projections)
        self.check_camera_parameters(self.proj_ds,projs=True)

        # check image parameters (i.e.reconsturcte)
        self.check_camera_parameters(self.recon_ds,projs=False)
        

        self.print_summary()
    
    
    
    def check_camera_parameters(self,ds,projs=True):
        camera_manufacturer = ds.Manufacturer
        camera_model = ds.ManufacturerModelName
        acquisition_date = ds.AcquisitionDate
        acquisition_time = ds.AcquisitionTime
        modality = ds.Modality
        try:
            duration =(ds.RotationInformationSequence[0].ActualFrameDuration / 1000)
        except:
            duration = np.nan
        rows = ds.Rows
        cols = ds.Columns
        try:
            zoom = ds.DetectorInformationSequence[0].ZoomFactor
        except:
            zoom = None
        try:
            number_projections = ds.RotationInformationSequence[0].NumberOfFramesInRotation * ds.NumberOfDetectors
        except:
            number_projections = np.nan

       

        if projs:
            self.append_to_summary(f'CAMERA: {camera_manufacturer} {camera_model}\t \n')
            self.append_to_summary(f'MODALITY: {modality}\t \n')
            self.append_to_summary(f'Scan performed on: {acquisition_date} at {acquisition_time}\t \n\n')
            self.append_to_summary(f' \t \n\n')

            self.append_to_summary(f'PROJECTIONS:\t \n')

            #check number of projections
            if number_projections == self.isotope_dic['raw']['n_proj'][camera_manufacturer.lower().split(' ')[0]]:
                self.append_to_summary(f"NUMBER OF PROJECTIONS: {number_projections} of {self.isotope_dic['raw']['n_proj'][camera_manufacturer.lower().split(' ')[0]]}\t         OK\n\n")
            else:
                self.append_to_summary(f"NUMBER OF PROJECTIONS: {number_projections} of {self.isotope_dic['raw']['n_proj'][camera_manufacturer.lower().split(' ')[0]]}\t         VERIFY\n\n")

            #check collimator
            accepted_collimators = self.isotope_dic['accepted_collimators']
            if (len(self.db_df['cal_data']['collimator'].unique()) == 1) & (self.db_df['cal_data']['collimator'].unique()[0] in accepted_collimators):
                self.append_to_summary(f"COLLIMATOR: {self.db_df['cal_data']['collimator'].unique()[0]}\t         OK\n\n")
            else:
                self.append_to_summary(f"COLLIMATOR: {self.db_df['cal_data']['collimator'].unique()[0]} is not in the accepted collimator list.\t         VERIFY\n\n")
            
            #check duration
            if round(duration) == self.isotope_dic['raw']['duration_per_proj']:
                self.append_to_summary(f'DURATION PER PROJECTION: {round(duration)} seconds\t       OK\n\n')
            else:
                self.append_to_summary(f'DURATION PER PROJECTION: {duration} seconds (should be {self.isotope_dic["planar"]["duration"]} seconds)\t       MISMATCH \n\n')
            
            #check energy windows
            self.window_check_df['projections'] = self.window_check(type='raw')

             #check non-circular orbit
            try:
                if len(set(self.proj_ds.DetectorInformationSequence[0].RadialPosition)) > 1:
                    self.append_to_summary(f'ORBIT: Non-circular\t      OK\n\n')
            except:
                    self.append_to_summary(f'ORBIT: Circular\t       VERIFY\n\n')
                          
            
            #check zoom
            if zoom == self.isotope_dic['spect']['zoom']:
                self.append_to_summary(f'ZOOM: {zoom}\t                   OK\n\n')
            else:
                self.append_to_summary(f'ZOOM: {zoom} (should be {self.isotope_dic["planar"]["zoom"]}\t MISMATCH\n\n')

            #check matrix size
            if [rows,cols] == self.isotope_dic['raw']['matrix']:
                self.append_to_summary(f'MATRIX SIZE: {rows} x {cols}\t         OK\n\n')
            else:
                self.append_to_summary(f'MATRIX SIZE: {rows} x {cols} (should be {self.isotope_dic["spect"]["matrix"][0]} x {self.isotope_dic["spect"]["matrix"][1]}\tMISMATCH\n\n')

            #check detector motion
            if ds.TypeOfDetectorMotion == self.isotope_dic['raw']['detector_motion']:
                self.append_to_summary(f"DETECTOR MOTION: {self.isotope_dic['raw']['detector_motion']}\t       OK\n\n")
            else:
                self.append_to_summary(f"DETECTOR MOTION: {ds.TypeOfDetectorMotion} (should be {self.isotope_dic['raw']['detector_motion']})\t        MISMATCH \n\n")

        else:
            self.append_to_summary(f' \t \n\n')
            self.append_to_summary(f'\nRECONSTRUCTED IMAGE:\t \n')

            #check energy windows
            self.window_check_df['reconstructed_image'] = self.window_check(type='spect')

            #check applied corrections
            try:
                if set(self.isotope_dic['spect']['corrections']).issubset(set(list(ds.CorrectedImage.split())).intersection(set(self.isotope_dic['spect']['corrections']))) :
                    self.append_to_summary(f"CORRECTIONS APPLIED: {self.isotope_dic['spect']['corrections']}\t                   OK\n\n")
                else:
                    self.append_to_summary(f"CORRECTIONS APPLIED: {ds.CorrectedImage}. This image is not quantitative. Is missing {set(self.isotope_dic['spect']['corrections']) - set(list(ds.CorrectedImage))} correction.\t         FAIL\n\n")
            except:
                if set(self.isotope_dic['spect']['corrections']).issubset(set(list(ds.CorrectedImage)).intersection(set(self.isotope_dic['spect']['corrections']))) :
                    self.append_to_summary(f"CORRECTIONS APPLIED: {self.isotope_dic['spect']['corrections']}\t                   OK\n\n")
                else:
                    self.append_to_summary(f"CORRECTIONS APPLIED: {ds.CorrectedImage}. This image is not quantitative. Is missing {set(self.isotope_dic['spect']['corrections']) - set(list(ds.CorrectedImage))} correction.\t         FAIL\n\n")
                          
            #check matrix size
            if [rows,cols] == self.isotope_dic['spect']['matrix']:
                self.append_to_summary(f'MATRIX SIZE: {rows} x {cols}\t         OK\n\n')
            else:
                self.append_to_summary(f'MATRIX SIZE: {rows} x {cols} (should be {self.isotope_dic["spect"]["matrix"][0]} x {self.isotope_dic["spect"]["matrix"][1]}\tMISMATCH\n\n')

            #check zoom
            if zoom == self.isotope_dic['spect']['zoom']:
                self.append_to_summary(f'ZOOM: {zoom}\t                   OK\n\n')
            else:
                self.append_to_summary(f'ZOOM: {zoom} (should be {self.isotope_dic["planar"]["zoom"]}\tMISMATCH\n\n')
