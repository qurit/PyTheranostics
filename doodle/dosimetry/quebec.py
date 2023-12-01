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
import tempfile
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID

# %%

def getinputdata(patient_id, cycle):
    folder = "/mnt/y/Sara/Xinchi_Quebec_kidney_dosimetry"
    
    #######################################################################################
    # Activities from Patient statistics from MIM
    mimcsv_tp1 = f"{folder}/activity/{patient_id}_cycle0{cycle}.csv"
    activity_df = pd.read_csv(mimcsv_tp1)

    #######################################################################################
    # Load CT
    CT_folder = f"{folder}/{patient_id}/cycle0{cycle}/CTresampled" # this CT is resampled
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
    files = []
    for fname in glob.glob(SPECT_folder + "/*.dcm", recursive=False):
        files.append(pydicom.dcmread(fname))
        pixel_arrays = [slice.pixel_array for slice in files]
        SPECT = np.stack(pixel_arrays, axis=-1)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, "SliceLocation"):
            slices.append(f)
        else:
            skipcount = skipcount + 1
    slices = sorted(slices, key=lambda s: s.SliceLocation)

    SPECT_xdim=slices[0].Rows
    SPECT_ydim=slices[0].Columns
    SPECT_zdim=slices[0].NumberOfSlices
    SPECT_xsize=slices[0].PixelSpacing[0]
    SPECT_ysize=slices[0].PixelSpacing[1]
    SPECT_zsize=slices[0].SliceThickness
    SPECT_volume_voxel=SPECT_xsize*SPECT_ysize*SPECT_zsize/1000
    scalefactor = slices[0].RescaleSlope
    #injactivity = slices[0].RadionuclideTotalDose/10**6

    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)
    for i, s in enumerate(slices):
        img2d = s.pixel_array
        img2d = img2d * scalefactor# in Bq/ml
        img2d = img2d * SPECT_volume_voxel# in    MBq
        img3d[:, :, i] = img2d
    SPECT = np.squeeze(img3d)


    #######################################################################################
    # Load RT
    patient_path = f"{folder}/{patient_id}/cycle0{cycle}"
    RT = RTStructBuilder.create_from(
        dicom_series_path = glob.glob(os.path.join(patient_path, 'CT2'))[0], # this CT is not resampled, the one to which RT structures are attached
        rt_struct_path = glob.glob(os.path.join(patient_path, 'CT2/rt-struct.dcm'))[0]
        )
    roi_masks = {} 
    roi_names = RT.get_roi_names()
    # Resampling to 128, 128, 81 
    roi_masks_resampled = {}
    organslist = activity_df['Contour'].unique()
    for organ in organslist:
        mask_image = RT.get_roi_mask_by_name(organ)  # Access the mask using the 'organ' variable
        resized = img_as_bool(resize(mask_image, (128, 128, 81)))  # Resize to (128, 128, 81)
        roi_masks_resampled[organ] = resized
        RT_xdim = resized.shape[0]
        RT_ydim = resized.shape[1]
        RT_zdim = resized.shape[2]
        
        
    # Define your data for three observations: SPECT, CT, RT
    data = [
        {' ': 'SPECT', 'xdim': SPECT_xdim, 'ydim': SPECT_ydim, 'zdim': SPECT_zdim, 'xsize': SPECT_xsize, 'ysize': SPECT_ysize, 'zsize': SPECT_zsize},
        {' ': 'CT', 'xdim': CT.shape[0], 'ydim': CT.shape[1], 'zdim': CT.shape[2], 'xsize': CT_xysize[0], 'ysize': CT_xysize[1], 'zsize': CT_zsize},
        {' ': 'RT', 'xdim': RT_xdim, 'ydim': RT_ydim, 'zdim': RT_zdim, 'xsize': "", 'ysize': "", 'zsize': ""}
    ]

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Print the summary table
    print(df)


    return activity_df, CT, SPECT, roi_masks_resampled
    
    
    
# %%
# Luke's function from qurit github

def find_first_entry_containing_substring(list_of_attributes, substring, dtype=np.float32):
    line = list_of_attributes[np.char.find(list_of_attributes, substring)>=0][0]
    if dtype == np.float32:
        return np.float32(line.replace('\n', '').split(':=')[-1])
    elif dtype == str:
        return (line.replace('\n', '').split(':=')[-1].replace(' ', ''))
    elif dtype == int:
        return int(line.replace('\n', '').split(':=')[-1].replace(' ', ''))

def intf2dcm(headerfile, folder, patient_id, cycle):
    # Interfile attributes   
    with open(headerfile) as f:
        headerdata = f.readlines()
    headerdata = np.array(headerdata)
    dim1 = find_first_entry_containing_substring(headerdata, 'matrix size [1]', int)
    dim2 = find_first_entry_containing_substring(headerdata, 'matrix size [2]', int)
    dim3 = find_first_entry_containing_substring(headerdata, 'matrix size [3]', int)
    dx = find_first_entry_containing_substring(headerdata, 'scaling factor (mm/pixel) [1]', np.float32) 
    dy = find_first_entry_containing_substring(headerdata, 'scaling factor (mm/pixel) [2]', np.float32) 
    dz = find_first_entry_containing_substring(headerdata, 'scaling factor (mm/pixel) [3]', np.float32) 
    number_format = find_first_entry_containing_substring(headerdata, 'number format', str)
    num_bytes_per_pixel = find_first_entry_containing_substring(headerdata, 'number of bytes per pixel', np.float32)
    imagefile = find_first_entry_containing_substring(headerdata, 'name of data file', str)
    pixeldata = np.fromfile(os.path.join(str(Path(headerfile).parent), imagefile), dtype=np.float32)
    dose_scaling_factor = np.max(pixeldata) / (2**16 - 1) 
    pixeldata /= dose_scaling_factor
    pixeldata = pixeldata.astype(np.int32)
    path = f'{folder}/{patient_id}/cycle0{cycle}/MC'
    #ds = pydicom.read_file(glob.glob(os.path.join(path, 'spect.dcm'))[0])
    ds = pydicom.read_file(glob.glob(os.path.join(path, 'template.dcm'))[0])
    #ds = pydicom.read_file(os.path.join(str(Path(os.path.realpath(__file__)).parent), "template.dcm"))
    ds.BitsAllocated = 32
    ds.Rows = dim1
    ds.Columns = dim2
    ds.PixelRepresentation = 0
    ds.NumberOfFrames = dim3
    ds.PatientName = f'PR21-CAVA-0004'
    ds.PatientID = f'PR21-CAVA-0004'
    ds.PixelSpacing = [dx, dy]
    ds.SliceThickness = dz
    ds.Modality = 'NM'
    #ds.ReferencedImageSequence[0].ReferencedSOPClassUID = 'UN'
    ds.ReferencedImageSequence[0].ReferencedSOPInstanceUID = '1.2.826.0.1.3680043.10.740.6048439480987740658090176328874005953'
    ds.FrameOfReferenceUID = '1.2.826.0.1.3680043.10.740.2362138874319727035222927285105155066'
    #ds.ReferencedRTPlanSequence[0].ReferencedSOPClassUID = 'UN'
    #ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID = 'UN'
    ds.PixelData = pixeldata.tobytes()
    
    
    print(ds.ReferencedImageSequence[0].ReferencedSOPClassUID)
    print(ds.ReferencedImageSequence[0].ReferencedSOPInstanceUID)
    print(ds.FrameOfReferenceUID)
    print(ds.ReferencedRTPlanSequence[0].ReferencedSOPClassUID)
    print(ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID)
    ds.save_as("doseimage_try.dcm")
    return ds
# %%
def image_visualisation(image):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))    
    axs[0].imshow(image[:, :, 10])
    axs[0].set_title('Slice at index 50')
    axs[1].imshow(image[20, :, :].T)
    axs[1].set_title('Slice at index 100')
    axs[2].imshow(image[:,20, :])
    axs[2].set_title('Slice at index 47')
    plt.tight_layout()
    plt.show()
    
def getdosemapdata(patient_id, cycle):
    folder = "/mnt/y/Sara/PR21_dosimetry"
    
    ###### TO DO: merge files
    
    headerfile_path = f'{folder}/{patient_id}/cycle0{cycle}/MC/output/DoseimageGy_MC.hdr'
    print(headerfile_path)
    ds = intf2dcm(headerfile_path, folder, patient_id, cycle)
    Dosemap = ds.pixel_array
    Dosemap = np.rot90(Dosemap)
    Dosemap = np.transpose(Dosemap, (0,2,1))
    return Dosemap
    

# %%
