from typing import Any, Dict
import SimpleITK

import numpy

MetaDataType = Dict[int, Dict[str, Any]]

class LongitudinalStudy:
    """Longitudinal Study Data Class to hold multiple medical imaging datasets, alongside with masks for organs 
    /regions of interest and meta-data."""

    def __init__(self, 
                 images: Dict[int, SimpleITK.Image],
                 meta: Dict[int, MetaDataType]
                 ) -> None:
        """
        images: Dictionary of (time-point ID, numpy array) representing CT or quantitative nuclear medicine images. 
        meta: Dictionary of (time-point ID, Dictionary of meta-data) representing meta-data for each time point."""
       
        # TODO Consistency checks: verify that all time points are present in images, masks and meta.
        # TODO Consistency checks: verify that there are no missing masks across time points. 

        self.images = images
        self.masks:  Dict[int, Dict[str, numpy.ndarray]] = {}
        self.meta = meta

        return None
    
    def array_at(self, time_id: int) -> numpy.ndarray:
        """Access Array Data"""
        return numpy.transpose(
            numpy.squeeze(SimpleITK.GetArrayFromImage(self.images[time_id])), axes=(1, 2, 0)
            )

    def add_masks_to_time_point(self, time_id: int, masks: Dict[str, numpy.ndarray]) -> None:
        """Add a Dictionary of masks for a given time point."""
        if time_id in self.masks:
            print(f"Warning: Masks for Time ID = {time_id} already exist. Overwriting them...")

        self.masks[time_id] = masks
        #self.check_masks_consistency() # I commented it for now; since there were lesions segmented only in the last point for now

        return None
    

    def volume_of(self, region: str, time_id: int) -> float:
        """Returns the volume of a region of interest, in mL"""
        return numpy.sum(self.masks[time_id][region]) * self.voxel_volume(time_id=time_id)
    
    def activity_in(self, region: str, time_id: int) -> float:
        """Returns the activity within a region of interest. The units of the nuclear medicine data
        should be Bq/mL."""
        if self.meta[time_id]["Radionuclide"] == "N/A":
            raise AssertionError("Can't compute activity if the image data does not represent the distribution of a radionuclide")

        return numpy.sum(self.masks[time_id][region] * self.array_at(time_id=time_id) * self.voxel_volume(time_id=time_id))

    def voxel_volume(self, time_id: int) -> float:
        """Returns the volume of a voxel in mL"""
        spacing = self.images[time_id].GetSpacing()
        return spacing[0]/10 * spacing[1]/10 * spacing[2]/10
    
    def check_masks_consistency(self) -> None:
        """Check that we have the same masks in all time points"""
        masks_list = [sorted(list(masks.keys())) for _, masks in self.masks.items()]

        sample = masks_list[0]

        for masks in masks_list:
            if masks != sample:
                raise AssertionError(f"Incosistent Masks! -> {masks_list}")
            
        return None
    