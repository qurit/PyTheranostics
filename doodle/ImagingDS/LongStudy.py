from typing import Any, Dict, Optional, List
import SimpleITK
from SimpleITK import Image
from pathlib import Path
import numpy
from doodle.ImagingTools.Tools import itk_image_from_array, load_from_dicom_dir
from doodle.registration.PhantomToCT import PhantomToCTBoneReg
from doodle.ImagingTools.Tools import jaccard_index

MetaDataType = Dict[int, Dict[str, Any]]

class LongitudinalStudy:
    """Longitudinal Study Data Class to hold multiple medical imaging datasets, alongside with masks for organs 
    /regions of interest and meta-data."""

    def __init__(self, 
                 images: Dict[int, SimpleITK.Image],
                 meta: Dict[int, MetaDataType],
                 modality: str = "NM"
                 ) -> None:
        """
        images: Dictionary of (time-point ID, numpy array) representing CT or quantitative nuclear medicine images. 
        meta: Dictionary of (time-point ID, Dictionary of meta-data) representing meta-data for each time point."""
       
        # TODO Consistency checks: verify that all time points are present in images, masks and meta.
        # TODO Consistency checks: verify that there are no missing masks across time points. 

        if modality not in ["NM", "PT", "CT", "DOSE"]:
            raise ValueError(f"Modality {modality} is not supported.")
        
        self.modality = modality
        self.images = images
        self.masks:  Dict[int, Dict[str, numpy.ndarray]] = {}  # {time_id: {mask_name: array}}
        self.meta = meta
        
        # Mask mapping format:
        self._valid_masks = ["Kidney_Left", "Kidney_Right", "Liver", "Spleen", "Bladder",
                             "SubmandibularGland_Left", "SubmandibularGland_Right", "ParotidGland_Left", "ParotidGland_Right", 
                             "BoneMarrow",
                             "Skeleton",
                             "WholeBody",
                             "RemainderOfBody"
                             ]

        return None
    
    def array_at(self, time_id: int) -> numpy.ndarray:
        """Access Array Data"""
        return numpy.transpose(
            numpy.squeeze(SimpleITK.GetArrayFromImage(self.images[time_id])), axes=(1, 2, 0)
            )

    def array_of_activity_at(self, time_id: int, region: Optional[str] = None) -> numpy.ndarray:
        """Returns the array in units of activity in Bq, with the posibility of masking out for one specific region."""
        if self.modality not in ["NM", "PT"]:
            raise AssertionError(f"Activity can't be calculated from {self.modality} data.")
            
        array = self.array_at(time_id=time_id)
        mask = self.masks[time_id][region] if region is not None else numpy.ones(shape=array.shape, dtype=numpy.int8)

        return array * mask * self.voxel_volume(time_id=time_id)

    def add_masks_to_time_point(self, time_id: int, masks: Dict[str, numpy.ndarray], mask_mapping: Optional[Dict[str, str]] = None) -> None:
        """Add Masks to time point.

        Args:
            time_id (int): Index of time-point ID.
            masks (Dict[str, numpy.ndarray]): Dictionary containing masks for time point time_id, in the format {mask_name: mask_array}
            mask_mapping (Optional[Dict[str, str]], optional): Mapping between masks names in input masks dictionary, and standard mask names in pyTheranostics. Defaults to None. If None, takes each name as is.
            
        Raises:
            ValueError: If mapping between user input masks and pyTheranostics standard mask names is invalid.

        """
        
        # If mask mapping is not specified, utilize user defined names in masks Dictionary.
        if mask_mapping is None:
            mask_mapping = {mask_name: mask_name for mask_name in masks.keys()}        

        if time_id not in self.masks:
            self.masks[time_id] = {}
        
        for mask_source, mask_target in mask_mapping.items():
            
            if mask_source not in masks:
                raise ValueError(f"{mask_source} is not part of the available masks: {masks.keys()}")
            
            if mask_target not in self._valid_masks:
                raise ValueError(f"{mask_target} is not a valid mask name. Please use one of: {self._valid_masks}")
            
            if mask_target in self.masks[time_id]:
                print(f"Warning: {mask_target} found at Time = {time_id}. It will be over-written!")
                
            self.masks[time_id][mask_target] = masks[mask_source]
        
        return None
    
    def volume_of(self, region: str, time_id: int) -> float:
        """Returns the volume of a region of interest, in mL"""
        return numpy.sum(self.masks[time_id][region]) * self.voxel_volume(time_id=time_id)
    
    def activity_in(self, region: str, time_id: int) -> float:
        """Returns the activity within a region of interest. The units of the nuclear medicine data
        should be Bq/mL."""
        if self.meta[time_id]["Radionuclide"] == "N/A" or self.modality not in ["NM", "PT"]:
            raise AssertionError("Can't compute activity if the image data does not represent the distribution of a radionuclide")

        return numpy.sum(self.masks[time_id][region] * self.array_at(time_id=time_id) * self.voxel_volume(time_id=time_id))

    def voxel_volume(self, time_id: int) -> float:
        """Returns the volume of a voxel in mL"""
        spacing = self.images[time_id].GetSpacing()
        return spacing[0]/10 * spacing[1]/10 * spacing[2]/10
    
    def average_of(self, region: str, time_id: int) -> float:
        """_summary_

        Args:
            region (str): _description_
            time_id (int): _description_

        Returns:
            float: _description_
        """
        return numpy.average(self.array_at(time_id=time_id)[self.masks[time_id][region]])
        
    def add_bone_marrow_mask_from_phantom(self, phantom_skeleton_path: Path,
                                          phantom_bone_marrow_path: Path,
                                          num_iterations: int = 3) -> None:
        """Generate Bone Marrow mask on each time point by registering a generic skeleton
        derived from an XCAT phantom into the patient's Skeleton CT and subsequently applying
        this spatial transformation to register the phantom's bone marrow into the patient's 
        anatomy.

        Args:
            phantom_skeleton_path (Path): Path to phantom Skeleton .nii file.
            phantom_bone_marrow_path (Path): Path to phantom Bone Marrow .nii file.
        """
        print("Running Personalized Bone Marrow generation from XCAT Phantom. This feature is unstable. Please review the generated BoneMarrow masks.")
        
        if self.modality != "CT":
            raise AssertionError(f"Phantom skeleton can only be registered to CT data. This is modality = {self.modality}")
        
        if "Skeleton" not in self.masks[0]:
            raise AssertionError("Skeleton mask not found. Can't continue.")
        
        
        # Since algorithm is not very stable (sometimes registration fails), we perform multiple iterations (aka repetitions) and keep best
        # results according to jaccard index.
        best_index = {time_id: 0 for time_id in self.images.keys()}
        
        for i in range(num_iterations):
            print(f"Registration :: Iteration {i+1}")
            # Loop through each time point:
            for time_id, ct in self.images.items():
                # Register Skeleton
                print(f" >> Registering Phantom Skeleton to CT at time point {time_id} ...")
                RegManager = PhantomToCTBoneReg(CT=ct, phantom_skeleton_path=phantom_skeleton_path)
                _ = RegManager.register(fixed_image=RegManager.CT, moving_image=RegManager.Phantom)
                
                # Register Bone Marrow
                marrow_mask = numpy.transpose(
                    SimpleITK.GetArrayFromImage(
                    RegManager.register_mask(fixed_image=RegManager.CT, mask_path=phantom_bone_marrow_path)), 
                    axes=(1, 2, 0))
                
                # Threshold:
                marrow_mask = (marrow_mask >= 1)
                
                # Exclude voxels outside of the patient's skeleton.
                marrow_mask *= self.masks[time_id]["Skeleton"]
                
                # Compute Index:
                jaccard = jaccard_index(self.masks[time_id]["Skeleton"], marrow_mask)
                
                if jaccard > best_index[time_id]:
                    self.masks[time_id]["BoneMarrow"] = marrow_mask # Threshold.
                    best_index[time_id] = jaccard
                
                # Calculate Index
                print(f" >>> Jaccard Index between Skeleton and Segmented Bone Marrow: {jaccard: 1.2f}")
            
        # Final Results:
        print(f" >>> Final Jaccard Indices:")
        for time_id in self.masks.keys():
            print(f" >>> Time point {time_id}: {best_index[time_id]}")
            
        return None    

        
    def check_masks_consistency(self) -> None:
        """Check that we have the same masks in all time points"""
        masks_list = [sorted(list(masks.keys())) for _, masks in self.masks.items()]

        sample = masks_list[0]

        for masks in masks_list:
            if masks != sample:
                raise AssertionError(f"Incosistent Masks! -> {masks_list}")
            
        return None
    
    def save_image_to_nii_at(self, time_id: int, out_path: Path, name: str = "") -> None:
        """Save Image from a particular time-point as a nifty file.

        Args:
            time_id (int): The time ID representing the time point to be saved. 
            out_path (Path): The path to the folder where images will be written.
        """
        print(f"Writing Image ({name}) into nifty file.")
        SimpleITK.WriteImage(image=SimpleITK.Cast(self.images[time_id], SimpleITK.sitkInt32), fileName=out_path / f"Image_{time_id}{name}.nii.gz")
        return None
    
    def save_masks_to_nii_at(self, time_id: int, out_path: Path, regions: List[str]) -> None:
        """Save Masks from a particular time-point as a nifty file.

        Args:
            time_id (int): The time ID representing  the time point to be saved. 
            out_path (Path): The path to the folder where images will be written.
            regions (List[str]): A list of regions (masks) to be saved. If empty, save all masks.
        """
        mask_names = list(self.masks[time_id].keys())
        all_masks = numpy.zeros_like(self.masks[time_id][mask_names[0]]).astype(numpy.int16)  # Get the shape of the first mask available.
        
        if len(regions) > 0:
            mask_names = [region for region in regions if region in self.masks[time_id].keys()]
        
        for mask_id, region_name in enumerate(mask_names):
            all_masks += (mask_id + 1) * (self.masks[time_id][region_name]).astype(numpy.int16)

        mask_image = itk_image_from_array(array=numpy.transpose(all_masks, axes=(2, 0, 1)), ref_image=self.images[time_id])
        
        print(f"Writing Masks ({mask_names}) into nifty file.")
        
        SimpleITK.WriteImage(image=mask_image, fileName=out_path / f"Masks_{time_id}.nii.gz")
        
        return None
    
    
# TODO: Find the proper placement. Currently here to avoid circular imports. Consider making it part of the init class.
def create_logitudinal_from_dicom(dicom_dirs: List[str], modality: str = "CT") -> LongitudinalStudy:
    """Creates a LongitudinalStudy object from a list of dicom dirs. Currently it assumes the order of the list
    corresponds to the order of the time points.

    Args:
        dicom_dirs (List[str]): _description_
        modality (str, optional): _description_. Defaults to "CT".

    Returns:
        LongitudinalStudy: _description_
    """
    # TODO: should fix this to make it robust and look at dicom header info for sorting time-points.
    mod_supported = ["CT", "Lu177_SPECT"]
    if modality not in mod_supported:
        raise NotImplemented(f"{modality} not supported. Currently, the following  modalities are supported: {mod_supported}")

    images: Dict[int, Image] = {}
    metadata: Dict[int, MetaDataType] = {}

    for time_id, dir in enumerate(dicom_dirs):
        image, meta = load_from_dicom_dir(dir=dir, modality=modality)
        images[time_id] = image
        metadata[time_id] = meta

    return LongitudinalStudy(images=images, meta=metadata, modality=modality if modality == "CT" else "NM")