from typing import Tuple, Callable, List, Dict
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
import scipy
import SimpleITK as sitk
from skimage import io
from skimage.transform import resize
from skimage import img_as_bool
from pathlib import Path
import sys
import tempfile
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID

def load_CT(
    folder: str, 
    patient_id: str, 
    cycle: str
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load CT data from DICOM files in the specified folder."""
    
    CT_folder = f"{folder}/{patient_id}/cycle0{cycle}/CT1"
    dicom_slices  = [pydicom.dcmread(fname) for fname in glob.glob(CT_folder + "/*.dcm", recursive=False)]

    pixel_arrays = [slice.pixel_array for slice in dicom_slices if hasattr(slice, "pixel_array")]
    CT = np.stack(pixel_arrays, axis=-1)

    dicom_slices = [f for f in dicom_slices if hasattr(f, "SliceLocation")]
    sorted_slices = sorted(dicom_slices, key=lambda s: s.SliceLocation)

    voxel_size_x, voxel_size_y = sorted_slices[0].PixelSpacing
    voxel_size_z = sorted_slices[0].SliceThickness
    voxel_sizes = (voxel_size_x, voxel_size_y, voxel_size_z)

    img_shape = list(sorted_slices[0].pixel_array.shape) + [len(sorted_slices)]
    img3d = np.zeros(img_shape)

    for i, slice_obj in enumerate(sorted_slices):
        img2d = slice_obj.pixel_array
        img3d[:, :, i] = img2d

    CT = np.squeeze(img3d)
    CT = CT - 1024

    return CT, tuple(voxel_sizes)

def voxel_sizes(
    header: pydicom.dataset.FileDataset
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Calculate shape and voxel size."""
    
    xdim=header.Rows
    ydim=header.Columns
    zdim=header.NumberOfFrames
    shape = (xdim, ydim, zdim)

    xsize=header.PixelSpacing[0]
    ysize=header.PixelSpacing[1]
    zsize=header.SpacingBetweenSlices
    voxel_sizes = (xsize, ysize, zsize)

    return shape, voxel_sizes

def load_and_resample_RT(
    folder: str, 
    patient_id: str, 
    cycle: str,
    no: int,
    ) -> Tuple[Dict[str, Tuple[int, int, int]], Dict[str, np.ndarray]]:
    """Load and resample RT data from DICOM files in the specified folder."""
    
    def clean_roi_name(roi_name: str) -> str:
        cleaned_roi_name = roi_name.replace(" ", "").replace('-', '_').replace('(', '_').replace(')', '')
        return cleaned_roi_name

    def resample_mask(mask: np.ndarray) -> np.ndarray:
        return img_as_bool(resize(mask, (128, 128, 233)))

    patient_path = f"{folder}/{patient_id}/cycle0{cycle}"

    masks_shape = {}
    masks_resampled = {}

    RT = RTStructBuilder.create_from(
        dicom_series_path=glob.glob(os.path.join(patient_path, f'CT{no}_RT'))[0],
        rt_struct_path=glob.glob(os.path.join(patient_path, f'CT{no}_RT/rt-struct.dcm'))[0]
    )
    roi_masks = {}
    roi_names = RT.get_roi_names()
    
    # Adjusting names
    for roi_name in roi_names:
        cleaned_roi_name = clean_roi_name(roi_name)
        mask = RT.get_roi_mask_by_name(roi_name)
        roi_masks[cleaned_roi_name] = mask
        
    # Resampling to (128, 128, 233)
    masks_resampled = {organ: resample_mask(mask) for organ, mask in roi_masks.items()}
    masks_shape = masks_resampled[list(roi_masks.keys())[0]].shape

    return masks_shape, masks_resampled

def remainder(
    shape: Tuple[int, int, int], 
    masks: dict
    ) -> dict:
    # Create th ROI ROB (WBCT-everything else)
    all_organs = np.zeros((shape[0], shape[1], shape[2]))
    organs = [organ for organ in masks.keys() if (organ != "WBCT" and organ not in ["Kidney_L_m", "Kidney_R_m", 
                                                                                    'Liver_Reference', 
                                                                                    'L1', 'L2', 'L3', 'L4', 'L5',
                                                                                    'Humerus_R', 'Humerus_L',
                                                                                    'Femur_L', 'Femur_R',
                                                                                    'TotalTumorBurden'])]
    for organ in organs:
        all_organs +=  masks[organ]
    rob = masks['WBCT'] - all_organs
    rob = rob.astype(bool)
    masks['ROB'] = rob
    
    return masks

def getinputdata(
    patient_id: str, 
    cycle: str
    ) -> Tuple[np.ndarray, pydicom.dataset.FileDataset, pydicom.dataset.FileDataset, dict, dict]:
    """Get CT, SPECT and Masks for further processing."""
    
    folder = "/mnt/y/Sara/PR21_dosimetry"
    
    # Load CT
    CT, CT_voxel_sizes = load_CT(folder, patient_id, cycle)

    # Load SPECT1
    SPECT1_folder_path = f"{folder}/{patient_id}/cycle0{cycle}/SPECT1"
    SPECT1_hdr = pydicom.read_file(glob.glob(os.path.join(SPECT1_folder_path, '*.dcm'))[0])
    SPECT1_shape, SPECT1_voxel_sizes = voxel_sizes(SPECT1_hdr)
    
    # Load SPECT2
    if cycle == '1':
        SPECT2_folder_path = f"{folder}/{patient_id}/cycle0{cycle}/SPECT2"
        SPECT2_hdr = pydicom.read_file(glob.glob(os.path.join(SPECT2_folder_path, '*.dcm'))[0])
        SPECT2_shape, SPECT2_voxel_sizes = voxel_sizes(SPECT2_hdr)
    
    # Load RT1
    masks1_shape, masks1_resampled = load_and_resample_RT(folder, patient_id, cycle, 1)
    masks1_resampled = remainder(masks1_shape, masks1_resampled)
    
    # Load RT2
    if cycle == '1':
        masks2_shape, masks2_resampled = load_and_resample_RT(folder, patient_id, cycle, 2)
        masks2_resampled = remainder(masks2_shape, masks2_resampled)
    
    
    # Define your data for three observations: SPECT, CT, RT
    data = [
        {' ': 'SPECT1', 'xdim': SPECT1_shape[0], 'ydim': SPECT1_shape[1], 'zdim': SPECT1_shape[2], 'xsize': SPECT1_voxel_sizes[0], 'ysize': SPECT1_voxel_sizes[1], 'zsize': SPECT1_voxel_sizes[2]},
        {' ': 'SPECT2', 'xdim': SPECT2_shape[0], 'ydim': SPECT2_shape[1], 'zdim': SPECT2_shape[2], 'xsize': SPECT2_voxel_sizes[0], 'ysize': SPECT2_voxel_sizes[1], 'zsize': SPECT2_voxel_sizes[2]},
        {' ': 'CT', 'xdim': CT.shape[0], 'ydim': CT.shape[1], 'zdim': CT.shape[2], 'xsize': CT_voxel_sizes[0], 'ysize': CT_voxel_sizes[1], 'zsize': CT_voxel_sizes[2]},
        {' ': 'Mask1', 'xdim': masks1_shape[0], 'ydim': masks1_shape[1], 'zdim': masks1_shape[2], 'xsize': "", 'ysize': "", 'zsize': ""},
        {' ': 'Mask2', 'xdim': masks2_shape[0], 'ydim': masks2_shape[1], 'zdim': masks2_shape[2], 'xsize': "", 'ysize': "", 'zsize': ""}
    ]
    df = pd.DataFrame(data)
    print(df)

    if cycle == '1':
        return CT, SPECT1_hdr, SPECT2_hdr, masks1_resampled, masks2_resampled
    else:
        return  CT, SPECT1_hdr, masks1_resampled
    
    
    
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
    axs[0].imshow(image[:, :, 120])
    axs[0].set_title('Slice at index 50')
    axs[1].imshow(image[120, :, :].T)
    axs[1].set_title('Slice at index 100')
    axs[2].imshow(image[:,64, :])
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
