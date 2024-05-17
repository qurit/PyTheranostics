from typing import Optional
from scipy import signal
import numpy
import os
from doodle.MiscTools.Tools import hu_to_rho


class DoseVoxelKernel:
    def __init__(self, isotope: str, voxel_size_mm: float) -> None:
        """_summary_

        Args:
            isotope (str): _description_
            voxel_size_mm (float): _description_
        """

        self.kernel = numpy.fromfile(
            os.path.dirname(__file__) + f"/../data/voxel_kernels/{isotope}-{voxel_size_mm:1.2f}-mm-mGyperMBqs-SoftICRP.img",
            dtype=numpy.float32
            )
        
        self.kernel = self.kernel.reshape((51, 51, 51)).astype(numpy.float64)
            
    def tia_to_dose(self, tia_mbq_s: numpy.ndarray, ct: Optional[numpy.ndarray] = None) -> numpy.ndarray:

        dose_mGy = signal.fftconvolve(tia_mbq_s, self.kernel, mode='same', axes=None)

        if ct is not None:
            # TODO: Handle erroneous scale-up of dose outside of body.
            print("Warning -> Scaling dose by density will yield erroneous dose values in very low density voxels (e.g., air inside the body)."
                  " Please use at your own risk")
            dose_mGy = self.weight_dose_by_density(dose_map=dose_mGy, ct=ct)

        return dose_mGy
        
    def weight_dose_by_density(self, dose_map: numpy.ndarray, ct: numpy.ndarray) -> numpy.ndarray:
        """Scale dose per voxel by voxel density. This is only valid for voxels of density similar to that of soft tissue and will also improve results for voxels 
        with higher density of soft tissue in some instances. However, it will over-estimate doses in voxels with lower density than soft tissue.
        To prevent dose to shoot-up in areas of air where there is activity present (e.g., in the patient's gut), we do not apply scaling based on density in those voxels (i.e., we apply a factor of 1, which is equivalent to saying 
        the tissue is ~ soft tissue).

        Args:
            dose_map (numpy.ndarray): Dose-map obtained from convolution of TIA map and Dose Kernel.
            ct (numpy.ndarray): CT image, in HU.

        Returns:
            numpy.ndarray: Modified Dose-map with dose per voxel scaled-up by density.
        """
        
        
        return 1 / hu_to_rho(hu=numpy.clip(ct, 0, 99999)) * dose_map
    

