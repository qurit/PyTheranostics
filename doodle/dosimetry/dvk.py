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
            os.path(__file__) + f"/../data/voxel_kernels/{isotope}-{voxel_size_mm:1.2f}-mm-mGyperMBqs-SoftICRP.img"
            )
        
        self.kernel = self.kernel.reshape((51, 51, 51)).astype(numpy.float64)
            
    def tiac_to_dose(self, tiac_mbq_s: numpy.ndarray, ct: Optional[numpy.ndarray] = None) -> numpy.ndarray:

        dose_mGy = signal.fftconvolve(tiac_mbq_s, self.kernel, mode='same', axes=None)

        if ct is not None:
            dose_mGy = self.weight_dose_by_density(dose_map=dose_mGy, ct=ct)

        return dose_mGy
        
    def weight_dose_by_density(dose_map: numpy.ndarray, ct: numpy.ndarray) -> numpy.ndarray:

        return 1 / hu_to_rho(hu=ct) * dose_map
    

