import numpy
from typing import Any, Tuple, Dict, List
import SimpleITK
from SimpleITK import Image
import pydicom
import glob
from rt_utils import RTStructBuilder
from skimage.transform import resize
from skimage import img_as_bool
from pathlib import Path
from doodle.ImagingDS.LongStudy import LongitudinalStudy

MetaDataType = Dict[str, Any]  # This could be improved ...

def load_metadata(dir: str, is_CT: bool = True) -> MetaDataType:
    """Loads relevant meta-data from a dicom dataset. """

    dicom_slices  = [pydicom.dcmread(fname) for fname in glob.glob(dir + "/*.dcm", recursive=False)]

    if len(dicom_slices) == 0:
        raise AssertionError(f"No Dicom data was found under {dir}")
    
    if is_CT:
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


    # Global attributes. Should be the same in all slices!
    slice_ = dicom_slices[0]

    meta = {"AcquisitionDate": slice_.AcquisitionDate,
            "AcquisitionTime": slice_.AcquisitionTime,
            "PatientID": slice_.PatientID}

    return meta

def load_from_dicom_dir(
    dir: str, is_CT: bool = True
    ) -> Tuple[SimpleITK.Image, MetaDataType]:
    """Load CT or SPECT data from DICOM files in the specified folder.
    Returns the Image object and some relevant metadata. 
    """
    # Read image content and spatial information using SimpleITK
    reader = SimpleITK.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dir)
    reader.SetFileNames(dicom_files)
    
    image = reader.Execute()

    # Load Meta Data using pydicom.
    meta = load_metadata(dir=dir, is_CT=is_CT)

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

def create_logitudinal_from_dicom(dicom_dirs: List[str], is_CT: bool = True) -> LongitudinalStudy:
    """Creates a LongitudinalStudy object from a list of dicom dirs. Currently it assumes the order of the list
    corresponds to the order of the time points... should fix this to make it robust and look at dicom header info for
    sorting time-points."""

    images: Dict[int, SimpleITK.Image] = {}
    metadata: Dict[int, MetaDataType] = {}

    for time_id, dir in enumerate(dicom_dirs):
        image, meta = load_from_dicom_dir(dir=dir, is_CT=is_CT)
        images[time_id] = image
        metadata[time_id] = meta

    return LongitudinalStudy(images=images, meta=metadata)


