import pydicom
import numpy as np
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid
from doodle.shared.radioactive_decay import get_activity_at_injection
import pandas as pd
from datetime import datetime

class DicomModify():

    def __init__(self,fname,CF):
        self.ds=pydicom.read_file(fname)
        self.CF=CF
        self.fname = fname

    def make_bqml_suv(self,weight,height,injection_date,pre_inj_activity,pre_inj_time,post_inj_activity,post_inj_time,injection_time,activity_meter_scale_factor,half_life=574300,radiopharmaceutical='Lutetium-PSMA-617',n_detectors = 2):
        #Half-life is in seconds


        
        # Siemens has an issue setting up the times. We are using the Acquisition time which is the time of the start of the last bed to harmonize.
        if 'siemens' in self.ds.Manufacturer.lower():
            self.ds.SeriesTime = self.ds.AcquisitionTime
            self.ds.ContentTime = self.ds.AcquisitionTime
        
        # Get the frame duration in seconds
        frame_duration = self.ds.RotationInformationSequence[0].ActualFrameDuration/1000
        # get number of projections because manufacturers scale by this in the dicomfile
        n_proj = self.ds.RotationInformationSequence[0].NumberOfFramesInRotation*n_detectors
        #get voxel volume in ml
        vox_vol =np.append(np.asarray(self.ds.PixelSpacing),float(self.ds.SliceThickness))
        vox_vol = np.prod(vox_vol/10)

        # Get image in Bq/ml
        A = self.ds.pixel_array
        A.astype(np.float64)
        A = A / (frame_duration * n_proj) * self.CF * 1e6 / vox_vol

        slope,intercept = dicom_slope_intercept(A)

        # update the PixelData
        A = np.int16((A - intercept)/slope)  # GE dicom is signed so np.int16 

        #bring the new image to the pixel bytes
        self.ds.PixelData = A.tobytes()

        self.ds.PixelData

        #update DICOM tags
        # self.ds.Units = 'BQML'
        self.ds.SeriesDescription = 'QSPECT_' + self.ds.SeriesDescription

        # add the RealWorldValueMappingSequence tag [0040,9096]
        self.ds.add_new([0x0040, 0x9096], 'SQ',[])
        self.ds.RealWorldValueMappingSequence += [Dataset(),Dataset()]

        for i in range(2):
            self.ds.RealWorldValueMappingSequence[i].RealWorldValueIntercept = intercept
            self.ds.RealWorldValueMappingSequence[i].RealWorldValueSlope = slope
            self.ds.RealWorldValueMappingSequence[i].RealWorldValueLastValueMapped = int(A.max())
            self.ds.RealWorldValueMappingSequence[i].RealWorldValueFirstValueMapped = int(A.min())

            self.ds.RealWorldValueMappingSequence[i].LUTLabel = 'BQML'
            self.ds.RealWorldValueMappingSequence[i].add_new([0x0040,0x08EA],'SQ',[])
            self.ds.RealWorldValueMappingSequence[i].MeasurementUnitsCodeSequence += [Dataset()]
            self.ds.RealWorldValueMappingSequence[i].MeasurementUnitsCodeSequence[0].CodeValue='Bq/ml'
            
        #add info for SUV
        self.ds.PatientWeight= str(weight) # in kg
        self.ds.PatientSize = str(height/100) # in m

        self.ds.DecayCorrection = 'START'
        self.ds.CorrectedImage.insert(0,'DECY')

        self.ds.add_new([0x0054, 0x0016], 'SQ',[])
        self.ds.RadiopharmaceuticalInformationSequence += [Dataset()]


        # values for net injected activity and injection date and time
        start_datetime, total_injected_activity = get_activity_at_injection(injection_date,pre_inj_activity,pre_inj_time,post_inj_activity,post_inj_time,injection_time,half_life=half_life)
        total_injected_activity = total_injected_activity * activity_meter_scale_factor


        scan_datetime = datetime.strptime(self.ds.SeriesDate + self.ds.SeriesTime,'%Y%m%d%H%M%S.%f')
        delta_scan_inj = (scan_datetime - start_datetime).total_seconds()/(60*60*24)

        pre_inj_datetime = datetime.strptime(injection_date + pre_inj_time,'%Y%m%d%H%M')
        post_inj_datetime = datetime.strptime(injection_date + post_inj_time,'%Y%m%d%H%M')

        inj_dic = {'patient_id':[self.ds.PatientID],'weight_kg':[weight],'height_m':[height],'pre_inj_activity_MBq':[pre_inj_activity],'pre_inj_datetime':[pre_inj_datetime],'post_inj_activity_MBq':[post_inj_activity],'post_inj_datetime':[post_inj_datetime],'injected_activity_MBq':[total_injected_activity],'injection_datetime':[start_datetime],'scan_datetime':[scan_datetime],'delta_t_days':[delta_scan_inj]}
        inj_df = pd.DataFrame(data=inj_dic)
        

        self.ds.RadiopharmaceuticalInformationSequence[0].Radiopharmaceutical=radiopharmaceutical
        self.ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalVolume=""
        self.ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime=start_datetime.strftime("%H%M%S.%f")
        self.ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose=str(round(total_injected_activity, 4))
        self.ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife=str(half_life)
        self.ds.RadiopharmaceuticalInformationSequence[0].RadionuclidePositronFraction=''
        self.ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartDateTime=start_datetime.strftime('%Y%m%d%H%M%S.%f')
        
        # for storing as new series data
        sop_ins_uid = self.ds.SOPInstanceUID 
        b = sop_ins_uid.split('.')
        b[-1] = str(int(b[-1]) + 1)
        self.ds.SOPInstanceUID = '.'.join(b)

        ser_ins_uid = self.ds.SeriesInstanceUID
        b = ser_ins_uid.split('.')
        b.pop()
        prefix = '.'.join(b) + '.'
        self.ds.SeriesInstanceUID = generate_uid(prefix=prefix)

        # self.ds.MediaStorageSOPInstaceUID
        return inj_df


    def save(self):
        self.ds.save_as(f"{self.fname.split('.dcm')[0]}_out.dcm")




    



def dicom_slope_intercept(img):
    '''This function calculates the slope and intercept for a DICOM image in the way that GE does it.
        GE PET images are stored in DICOM files that are signed int16.  This allows for a maximum value of 32767.  
        The slope is calculated such that the maximum value in the pixel array (before multiplying by slope) is 32767. 
    
    Parameters
    ----------
        img: numpy array
          contains the float values of the image (e.g. MBq/ml in our case)

    Returns
    -------
        slope: float
         the slope to be set in the dicom header

        intercept: float
         the intercept for the dicom header    '''


    max_val = np.max(img)
    min_val = np.min(img)

    slope = np.float32(max(max_val,-min_val)/32767)
    intercept = 0  #GE has assigned it to zero

    return float(slope),float(intercept)