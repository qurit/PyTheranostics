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
from doodle.ImagingDS.LongStudy import LongitudinalStudy
from doodle.dicomtools.dicomtools import sitk_load_dcm_series

MetaDataType = Dict[str, Any]  # This could be improved ...

# TODO: Move under dicomtools, and have two sets: one generic (the current dicomtools.py) and on specific for pyTheranostic functions (containing
# the code below)

def load_metadata(dir: str, modality: str) -> MetaDataType:
    """Loads relevant meta-data from a dicom dataset. """

    dicom_slices  = [pydicom.dcmread(fname) for fname in glob.glob(dir + "/*.dcm", recursive=False)]

    if len(dicom_slices) == 0:
        raise AssertionError(f"No Dicom data was found under {dir}")
    
    radionuclide = "N/A"

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
            raise ValueError(f"Wrong modality. User specified NM, howere dicom indicates {dicom_slices[0].Modality}.")

        radionuclide = modality.split("_")[0]

    # Global attributes. Should be the same in all slices!
    slice_ = dicom_slices[0]

    meta = {"AcquisitionDate": slice_.AcquisitionDate,
            "AcquisitionTime": slice_.AcquisitionTime,
            "PatientID": slice_.PatientID,
            "Radionuclide": radionuclide}

    return meta

def apply_qspect_dcm_scaling(image: Image, dir: str, scale_factor: Optional[Tuple[float, float]] = None) -> Image:
    """Read dicom metadata to extract appropriate scaling for Image voxel values, then 
    apply to original image and generate a new SimpleITK image object.
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
    re_scaled_image = SimpleITK.GetImageFromArray(image_array)
    
    # Set Manually basic meta:
    re_scaled_image.SetSpacing(list(image.GetSpacing())[:-1])
    re_scaled_image.SetOrigin(list(image.GetOrigin())[:-1])
    tmp_direction = list(image.GetDirection())
    re_scaled_image.SetDirection(tmp_direction[0:3] + tmp_direction[4:7] + tmp_direction[8:11])

    # Here we set the additional meta-data.
    for key in image.GetMetaDataKeys():
        re_scaled_image.SetMetaData(key, image.GetMetaData(key))

    return re_scaled_image


def load_from_dicom_dir(
    dir: str, modality: str
    ) -> Tuple[Image, MetaDataType]:
    """Load CT or SPECT data from DICOM files in the specified folder.
    Returns the Image object and some relevant metadata. 
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
    dir: str,
    target_shape: Tuple[int, int, int]
    ) -> Tuple[Dict[str, numpy.ndarray], Dict[str, numpy.ndarray]]:
    """Load and resample RT data from DICOM files in the specified folder."""
    
    # TODO: separate RTtoMask loading from re-sampling. 
    # For generalizability, Re-sampling should use spacing and origin from reference and target 
    # images.

    def clean_roi_name(roi_name: str) -> str:
        cleaned_roi_name = roi_name.replace(" ", "").replace('-', '_').replace('(', '_').replace(')', '')
        return cleaned_roi_name

    def resample_mask(mask: numpy.ndarray, shape: Tuple[int, int, int]) -> numpy.ndarray:
        return img_as_bool(resize(mask, shape))

    CT_folder = Path(dir)

    if not CT_folder.exists():
        raise FileNotFoundError(f"Folder {CT_folder.name} does not exists.")

    # TODO: very specific to current local structure. We might want to make it more generic... (e.g., when
    # we build a 'database')
    RT = RTStructBuilder.create_from(
        dicom_series_path=glob.glob(str(CT_folder))[0],
        rt_struct_path=glob.glob(str(CT_folder/"rt-struct.dcm"))[0]
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

def create_logitudinal_from_dicom(dicom_dirs: List[str], modality: str = "CT") -> LongitudinalStudy:
    """Creates a LongitudinalStudy object from a list of dicom dirs. Currently it assumes the order of the list
    corresponds to the order of the time points... should fix this to make it robust and look at dicom header info for
    sorting time-points."""

    images: Dict[int, Image] = {}
    metadata: Dict[int, MetaDataType] = {}

    for time_id, dir in enumerate(dicom_dirs):
        image, meta = load_from_dicom_dir(dir=dir, modality=modality)
        images[time_id] = image
        metadata[time_id] = meta

    return LongitudinalStudy(images=images, meta=metadata)


