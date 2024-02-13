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
    
    