# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
from datetime import datetime
import re
import glob
import pydicom 
from rt_utils import RTStructBuilder
import gatetools as gt
import scipy
import SimpleITK as sitk
from skimage import io
from skimage.transform import resize
from skimage import img_as_bool
from  doodle.fits.fits import monoexp_fun, fit_monoexp
from doodle.plots.plots import monoexp_fit_plots
from pathlib import Path
import sys

 

# %%
def getinputdata(patient_id, cycle):
    folder = "/mnt/y/Sara/PR21_dosimetry"
    
    #######################################################################################
    # Activities from Patient statistics from MIM
    mimcsv_tp1 = f"{folder}/activity/{patient_id}_cycle0{cycle}_tp1.csv"
    mimcsv_tp2 = f"{folder}/activity/{patient_id}_cycle0{cycle}_tp2.csv"
    activity_tp1_df = pd.read_csv(mimcsv_tp1)
    activity_tp2_df = pd.read_csv(mimcsv_tp2)
    
    activity_tp1_df['Contour'] = activity_tp1_df['Contour'].str.replace(' ', '')
    activity_tp1_df['Contour'] = activity_tp1_df['Contour'].str.replace('-', '_')
    activity_tp1_df['Contour'] = activity_tp1_df['Contour'].str.replace('(', '_')
    activity_tp1_df['Contour'] = activity_tp1_df['Contour'].str.replace(')', '')
    #######################################################################################
    
    
    # Retrieve information about date of acquisition
    # It is crucial, because name of files in incjection .csv are named with acq date
    acq1 = activity_tp1_df.loc[1, 'Series Date']
    date_object = datetime.strptime(acq1, "%Y-%m-%d")
    data1 = date_object.strftime("%Y%m%d")
    acq2 = activity_tp2_df.loc[1, 'Series Date']
    date_object = datetime.strptime(acq2, "%Y-%m-%d")
    data2 = date_object.strftime("%Y%m%d")
    
    # injection timepoint data have in their name hyphen CAVA-00X, so here i create patient id with hyphen
    # Use regular expression to insert a hyphen after every group of letters
    x_with_hyphen = re.sub(r'([A-Za-z]+)', r'\1-', patient_id)
    # Remove the trailing hyphen (if any)
    if x_with_hyphen.endswith('-'):
        x_with_hyphen = x_with_hyphen[:-1]
    
    #######################################################################################    
    # Load injection timepoint data
    injection_folder = f"{folder}/injection_timepoints"
    file_path = f"{injection_folder}/PR21-{x_with_hyphen}.{data1}.injection.info.csv"
    inj_timepoint1 = pd.read_csv(file_path)
    file_path = f"{injection_folder}/PR21-{x_with_hyphen}.{data2}.injection.info.csv"
    inj_timepoint2 = pd.read_csv(file_path)    
    #######################################################################################
    
    #######################################################################################
    # Load CT
    CT_folder = f"{folder}/{patient_id}/cycle0{cycle}/CT" # this CT is resampled
    files = []
    for fname in glob.glob(CT_folder + "/*.dcm", recursive=False):
        files.append(pydicom.dcmread(fname))
        pixel_arrays = [slice.pixel_array for slice in files]
        CT = np.stack(pixel_arrays, axis=-1)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, "SliceLocation"):
            slices.append(f)
        else:
            skipcount = skipcount + 1
    slices = sorted(slices, key=lambda s: s.SliceLocation)
    
    CT_xysize = slices[0].PixelSpacing
    CT_zsize = slices[0].SliceThickness
    sp = [CT_xysize[0], CT_xysize[1], CT_zsize]
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img3d[:, :, i] = img2d
    CT = np.squeeze(img3d)
    CT = CT - 1024  # Caution! Loaded CT has not accounted for shift in HUs! 'Rescale Intercept': -1024    
    #######################################################################################
    
    
    #######################################################################################
    # Load SPECT
    SPECT_folder = f"{folder}/{patient_id}/cycle0{cycle}/SPECT"
    SPECT_hdr = pydicom.read_file(glob.glob(os.path.join(SPECT_folder, '*.dcm'))[0]) # I added [0] because glob.glob(os.path.join(folder_path, '*.dcm')), produce a list. not a path, but the 1st element of the list is a path
    SPECT_xdim=SPECT_hdr.Rows
    SPECT_ydim=SPECT_hdr.Columns
    SPECT_zdim=SPECT_hdr.NumberOfFrames
    SPECT_xsize=SPECT_hdr.PixelSpacing[0]
    SPECT_ysize=SPECT_hdr.PixelSpacing[1]
    SPECT_zsize=SPECT_hdr.SpacingBetweenSlices
    SPECT_volume_voxel=SPECT_xsize*SPECT_ysize*SPECT_zsize/1000
    
    SPECT = SPECT_hdr.pixel_array
    SPECT = np.transpose(SPECT, (1, 2, 0)) # from (233,128,128) to (128,128,233)
    SPECT.astype(np.float64)
    scalefactor = SPECT_hdr.RealWorldValueMappingSequence[0].RealWorldValueSlope
    SPECT = SPECT * scalefactor # in Bq/ml
    SPECTMBq = SPECT * SPECT_volume_voxel / 1E6 # in MBq
    #######################################################################################
    
    
    #######################################################################################
    # Load RT
    patient_path = f"{folder}/{patient_id}/cycle0{cycle}"
    RT = RTStructBuilder.create_from(
        dicom_series_path = glob.glob(os.path.join(patient_path, 'CT2'))[0], # this CT is not resampled, the one to which RT structures are attached
        rt_struct_path = glob.glob(os.path.join(patient_path, 'CT2/rt-struct.dcm'))[0]
        )
    roi_masks = {} 
    roi_names = RT.get_roi_names()
    # Adjusting names so they much dataframe
    for roi_name in roi_names:
        cleaned_roi_name = roi_name.replace(" ", "")
        cleaned_roi_name = cleaned_roi_name.replace('-', '_')
        cleaned_roi_name = cleaned_roi_name.replace('(', '_')
        cleaned_roi_name = cleaned_roi_name.replace(')', '')
        mask = RT.get_roi_mask_by_name(roi_name)
        roi_masks[f"{cleaned_roi_name}"] = mask
    # Resampling to 128, 128, 233 
    roi_masks_resampled = {}
    organslist = activity_tp1_df['Contour'].unique()
    for organ in organslist:
        mask_image = roi_masks[organ]  # Access the mask using the 'organ' variable
        resized = img_as_bool(resize(mask_image, (128, 128, 233)))  # Resize to (128, 128, 233)
        roi_masks_resampled[organ] = resized
        RT_xdim = resized.shape[0]
        RT_ydim = resized.shape[1]
        RT_zdim = resized.shape[2]
    #######################################################################################
    

    # Define your data for three observations: SPECT, CT, RT
    data = [
        {'observation': 'SPECT', 'xdim': SPECT_xdim, 'ydim': SPECT_ydim, 'zdim': SPECT_zdim, 'xsize': SPECT_xsize, 'ysize': SPECT_ysize, 'zsize': SPECT_zsize},
        {'observation': 'CT', 'xdim': CT.shape[0], 'ydim': CT.shape[1], 'zdim': CT.shape[2], 'xsize': CT_xysize[0], 'ysize': CT_xysize[1], 'zsize': CT_zsize},
        {'observation': 'RT', 'xdim': RT_xdim, 'ydim': RT_ydim, 'zdim': RT_zdim, 'xsize': "", 'ysize': "", 'zsize': ""}
    ]

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Print the summary table
    print(df)

    return activity_tp1_df, activity_tp2_df, inj_timepoint1, inj_timepoint2, CT, SPECTMBq, roi_masks_resampled
# %%
