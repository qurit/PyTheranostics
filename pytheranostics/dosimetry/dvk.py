import os
from pathlib import Path
from typing import Optional

import numpy
from scipy import signal

from pytheranostics.MiscTools.Tools import hu_to_rho, load_kernel_from_csv


class DoseVoxelKernel:
    def __init__(self, isotope: str, voxel_size_mm: float) -> None:
        """Initialize Dose Voxel-Kernel for convolution-based dosimetry.

        Args:
            isotope (str): Isotope name, e.g., "Lu177".
            voxel_size_mm (float): Voxel size in mm, e.g., 4.80
        Raises:
            FileNotFoundError: If Voxel-Kernel file is not found.
        """

        # Set file path for Voxel-Kernel.
        kernel_file = Path(
            os.path.dirname(__file__)
            + f"/../data/voxel_kernels/{isotope}-{voxel_size_mm:2.2f}-mm-mGyperMBqs-Soft.csv"
        )

        if not kernel_file.exists():
            raise FileNotFoundError(
                f" >> Voxel Kernel for SPECT voxel size ({voxel_size_mm:2.2f} mm) not found."
            )

        self.kernel = load_kernel_from_csv(path=kernel_file)

    def tia_to_dose(
        self, tia_mbq_s: numpy.ndarray, ct: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:

        dose_mGy = signal.fftconvolve(tia_mbq_s, self.kernel, mode="same", axes=None)

        if ct is not None:
            # TODO: Handle erroneous scale-up of dose outside of body.
            print(
                "Warning -> Scaling dose by density will yield erroneous dose values in very low density voxels (e.g., air inside the body)."
                " Please use at your own risk"
            )
            dose_mGy = self.weight_dose_by_density(dose_map=dose_mGy, ct=ct)

        return dose_mGy

    def weight_dose_by_density(
        self, dose_map: numpy.ndarray, ct: numpy.ndarray
    ) -> numpy.ndarray:
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
