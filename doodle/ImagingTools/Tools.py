import numpy
from typing import Any, Tuple, Dict, List, Optional
import SimpleITK
from SimpleITK import Image
import pydicom
import glob
from rt_utils import RTStructBuilder
from skimage.transform import resize
from skimage import img_as_bool
from pathlib import Path
from doodle.dicomtools.dicomtools import sitk_load_dcm_series

MetaDataType = Dict[str, Any]  # This could be improved ...

# TODO: Move under dicomtools, and have two sets: one generic (the current dicomtools.py) and on specific for pyTheranostic functions (containing
# the code below)

def load_metadata(dir: str, modality: str) -> MetaDataType:
    """Loads relevant meta-data from a dicom dataset.

    Args:
        dir (str): _description_
        modality (str): _description_

    Raises:
        AssertionError: _description_
        ValueError: _description_
        AssertionError: _description_
        ValueError: _description_

    Returns:
        MetaDataType: _description_
    """

    dicom_slices  = [pydicom.dcmread(fname) for fname in glob.glob(dir + "/*.dcm", recursive=False)]

    if len(dicom_slices) == 0:
        raise AssertionError(f"No Dicom data was found under {dir}")
    
    radionuclide = "N/A"
    injected_activity = "N/A"

    if modality == "CT": 
        dicom_slices = [f for f in dicom_slices if hasattr(f, "SliceLocation")]
        dicom_slices = sorted(dicom_slices, key=lambda s: s.SliceLocation)

        if dicom_slices[0].Modality != "CT":
            raise ValueError(f"Wrong modality. User specified CT, howere dicom indicates {dicom_slices[0].Modality}.")
        
    else:
        # Should only be a single DICOM file for SPECT Reconstruction
        if len(dicom_slices) > 1:
            raise AssertionError(f"Found more than one potential SPECT dicom dataset")
        
        if dicom_slices[0].Modality != "NM":
            raise ValueError(f"Wrong modality. User specified NM, however dicom indicates {dicom_slices[0].Modality}.")

        radionuclide = modality.split("_")[0]
        
        # This only applies to Q-SPECT TODO: replace for something more generic.
        injected_activity = dicom_slices[0].RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose

    # Global attributes. Should be the same in all slices!
    slice_ = dicom_slices[0]

    meta = {"AcquisitionDate": slice_.AcquisitionDate,
            "AcquisitionTime": slice_.AcquisitionTime,
            "PatientID": slice_.PatientID,
            "Radionuclide": radionuclide,
            "Injected_Activity_MBq": injected_activity
            }

    return meta

def itk_image_from_array(array: numpy.ndarray, ref_image: Image) -> Image:
    """Create an ITK Image object with a new array and existing
        meta-data from another reference ITK image.

    Args:
        array (numpy.ndarray): _description_
        ref_image (Image): _description_

    Returns:
        Image: _description_
    """
    
    image = SimpleITK.GetImageFromArray(array)

    # Set Manually basic meta:
    tmp_spacing = list(ref_image.GetSpacing())
    tmp_origin = list(ref_image.GetOrigin())
    
    if len(tmp_spacing) - len(array.shape) == 1:  # Sometime we get NM data with 4 dimensions, the last one being just a dummy.
        tmp_spacing = tmp_spacing[:-1]
        tmp_origin = tmp_origin[:-1]
    
    image.SetSpacing(tmp_spacing)
    image.SetOrigin(tmp_origin)
    tmp_direction = list(ref_image.GetDirection())
    
    if len(tmp_direction) > 9:
        image.SetDirection(tmp_direction[0:3] + tmp_direction[4:7] + tmp_direction[8:11])
    else:
        image.SetDirection(tmp_direction)

    # Here we set the additional meta-data.
    for key in ref_image.GetMetaDataKeys():
        image.SetMetaData(key, ref_image.GetMetaData(key))

    return image

def apply_qspect_dcm_scaling(image: Image, dir: str, scale_factor: Optional[Tuple[float, float]] = None) -> Image:
    """Read dicom metadata to extract appropriate scaling for Image voxel values, then 
    apply to original image and generate a new SimpleITK image object.

    Args:
        image (Image): _description_
        dir (str): _description_
        scale_factor (Optional[Tuple[float, float]], optional): _description_. Defaults to None.

    Raises:
        AssertionError: _description_

    Returns:
        Image: _description_
    """
    if scale_factor is None:
        # We use pydicom to access the appropriate tag:
        # First, find the SPECT dicom file:
        path_dir = Path(dir)
        nm_files = [files for files in path_dir.glob("*.dcm")]
        if len(nm_files) != 1:
            raise AssertionError(f"Found more than 1 .dcm file inside {path_dir.name}, not sure which is is SPECT.")
        
        dcm_data = pydicom.dcmread(str(nm_files[0]))
        slope = dcm_data.RealWorldValueMappingSequence[0].RealWorldValueSlope
        intercept = dcm_data.RealWorldValueMappingSequence[0].RealWorldValueIntercept
    else:
        slope = scale_factor[0]
        intercept = scale_factor[1]

    # Re-scale voxel values
    image_array = numpy.squeeze(SimpleITK.GetArrayFromImage(image)) 
    image_array = slope * image_array.astype(numpy.float32) + intercept

    # Generate re-scaled SimpleITK.Image object by building a new Image from re_scaled array, and copying
    # metadata from original image.
    return itk_image_from_array(array=image_array, ref_image=image)

def load_from_dicom_dir(
    dir: str, modality: str
    ) -> Tuple[Image, MetaDataType]:
    """Load CT or SPECT data from DICOM files in the specified folder.
    Returns the Image object and some relevant metadata. 

    Args:
        dir (str): _description_
        modality (str): _description_

    Returns:
        Tuple[Image, MetaDataType]: _description_
    """
    # Read image content and spatial information using SimpleITK
    image = sitk_load_dcm_series(dcm_dir=Path(dir))

    # If Q-SPECT, need to re-scale Data as SimpleITK does not do it:
    if modality != "CT":
        image = apply_qspect_dcm_scaling(image=image, dir=dir)

    # Load Meta Data using pydicom.
    meta = load_metadata(dir=dir, modality=modality)

    return image, meta

def load_and_resample_RT(
    ref_dicom_dir: str,
    rt_struct_dir: str,
    target_shape: Tuple[int, int, int]
    ) -> Tuple[Dict[str, numpy.ndarray], Dict[str, numpy.ndarray]]:
    """Load and resample RT data from DICOM files in the specified folder.

    Args:
        ref_dicom_dir (str): _description_
        rt_struct_dir (str): _description_
        target_shape (Tuple[int, int, int]): _description_

    Raises:
        FileNotFoundError: _description_

    Returns:
        Tuple[Dict[str, numpy.ndarray], Dict[str, numpy.ndarray]]: _description_
    """
    
    # TODO: separate RTtoMask loading from re-sampling. 
    # For generalizability, Re-sampling should use spacing and origin from reference and target 
    # images.

    def clean_roi_name(roi_name: str) -> str:
        cleaned_roi_name = roi_name.replace(" ", "").replace('-', '_').replace('(', '_').replace(')', '')
        return cleaned_roi_name

    def resample_mask(mask: numpy.ndarray, shape: Tuple[int, int, int]) -> numpy.ndarray:
        return img_as_bool(resize(mask, shape))

    CT_folder = Path(ref_dicom_dir)
    RT_folder = Path(rt_struct_dir)

    if not CT_folder.exists():
        raise FileNotFoundError(f"Folder {CT_folder.name} does not exists.")

    # TODO: very specific to current local structure. We might want to make it more generic... (e.g., when
    # we build a 'database')
    RT = RTStructBuilder.create_from(
        dicom_series_path=glob.glob(str(CT_folder))[0],
        rt_struct_path=glob.glob(str(RT_folder) + "/*.dcm")[0]
    )
    roi_masks = {}
    roi_names = RT.get_roi_names()
    
    # Clean names, as they might come with unsupported characters from third party software.
    for roi_name in roi_names:
        cleaned_roi_name = clean_roi_name(roi_name)
        mask = RT.get_roi_mask_by_name(roi_name)
        roi_masks[cleaned_roi_name] = mask
        
    # Resampling:
    roi_masks_resampled = {organ: resample_mask(mask, target_shape) for organ, mask in roi_masks.items()}

    return roi_masks, roi_masks_resampled

def ensure_masks_disconnect(original_masks: Dict[str, numpy.ndarray]) -> Dict[str, numpy.ndarray]:
    """_summary_

    Args:
        original_masks (Dict[str, numpy.ndarray]): _description_

    Returns:
        Dict[str, numpy.ndarray]: _description_
    """
    
    if len(original_masks) == 0:
        return original_masks
    
    # Create multi-label array from all masks. Each mask is a different ID, overwriting the previous one if there are overlaps.
    original_masks_names = [region for region in original_masks.keys()]   
    all_original_mask = numpy.zeros(original_masks[original_masks_names[0]].shape, dtype=numpy.int16)
    
    id = 1
    final_regions: List[str] = []
    for region, mask in original_masks.items():
        all_original_mask[numpy.where(mask)] = id
        final_regions.append(region)
        id += 1
        
    # Split array into individual masks arrays.
    final_masks: Dict[str, numpy.ndarray] = {}
    for id_final in range(1, id):
        final_masks[final_regions[id_final - 1]] = numpy.where(all_original_mask == id_final, True, False)
        
    return final_masks
   
    
def extract_masks(time_id: int, mask_dataset: Dict[int, Dict[str, numpy.ndarray]], requested_rois: List[str]) -> Dict[str, numpy.ndarray]:
    """Extract masks from NM dataset, according to user-defined list. Enforce that masks are disconnected.
        Constrains: 
        - Tumors are always going to be removed from organs. 
        - For non-tumor regions with overlapping voxels, the newly added region will prevail.

    Returns:
        Dict[str, numpy.ndarray]: Dictionary of compliant masks.
    """
    # Available Mask Names:
    mask_names = [name for name in mask_dataset[time_id]]
            
    # Disconnect tumor masks (if there is any overlap among them)
    tumor_labels = [region for region in requested_rois if "Lesion" in region]
    tumors_masks = ensure_masks_disconnect(original_masks={tumor_label: mask_dataset[time_id][tumor_label] for tumor_label in tumor_labels})
    
    # Get mask of total tumor burden
    tumor_burden_mask = numpy.zeros_like(mask_dataset[time_id][requested_rois[0]])
    
    for _, tumor_mask in tumors_masks.items():
        tumor_burden_mask[numpy.where(tumor_mask)] = True
    
    # Remove tumor from normal tissue regions.
    non_tumor_masks_aggregate: Dict[str, numpy.ndarray] = {
        region: (
            numpy.clip((mask_dataset[time_id][region]).astype(numpy.int8) - tumor_burden_mask.astype(numpy.int8), 0, 1)
            ).astype(bool) for region in requested_rois if region not in tumor_labels
        }
    
    corrected_masks = ensure_masks_disconnect(original_masks=non_tumor_masks_aggregate)
    corrected_masks.update(tumors_masks)
    
    
    # Generate Whole Body Mask = RemainderOfBody + All Masks.
    whole_body = numpy.zeros_like(corrected_masks[mask_names[0]])
    
    for _, mask in corrected_masks.items():
        whole_body += mask
        
    whole_body = numpy.clip(whole_body, 0, 1)
    
    corrected_masks["WholeBody"] = whole_body
        
    return corrected_masks
                
