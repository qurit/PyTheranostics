from typing import Any, Dict

import numpy


class LongitudinalStudy:
    """Longitudinal Study Data Class to hold multiple medical imaging datasets, alongside with masks for organs 
    /regions of interest and meta-data."""

    def __init__(self, images: Dict[int, numpy.ndarray],
                 masks: Dict[int, Dict[str, numpy.ndarray]],
                 meta: Dict[int, Dict[str, Any]]
                 ) -> None:
        """
        images: Dictionary of (time-point ID, numpy array) representing quantitative nuclear medicine images. 
        masks: Dictionary of (time-point ID, Dictionary of masks) representing the masks for each time-point.
        meta: Dictionary of (time-point ID, Dictionary of meta-data) representing meta-data for each time point."""
       
        # TODO Consistency checks: verify that all time points are present in images, masks and meta.
        # TODO Consistency checks: verify that there are no missing masks across time points. 

        self.images = images
        self.masks = masks
        self.meta = meta

        return None
    
    def volume_of(self, region: str, time_id: int) -> float:
        """Returns the volume of a region of interest, in mL"""
        return numpy.sum(self.masks[time_id][region]) * self.voxel_volume(time_id=time_id)
    
    def sum_of(self, region: str, time_id: int) -> float:
        """Returns the sum of voxel values within a region of interest. If imaging data is
         quantitative Nuclear Medicine, this method returns activity in region, in Bq."""
        return numpy.sum(self.masks[time_id][region] * self.images[time_id])

    def voxel_volume(self, time_id: int) -> float:
        """Returns the volume of a voxel in mL"""
        return self.meta[time_id]["x_dim"] * self.meta[time_id]["y_dim"] * self.meta[time_id]["z_dim"]
    