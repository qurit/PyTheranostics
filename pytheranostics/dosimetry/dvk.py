"""Dose voxel kernel module for convolution-based dosimetry."""

from typing import Optional

import numpy
from scipy import signal

from pytheranostics.misc_tools.tools import hu_to_rho
from pytheranostics.shared.resources import resource_path


class DoseVoxelKernel:
    """Dose Voxel Kernel for convolution-based dosimetry calculations."""

    def __init__(self, isotope: str, voxel_size_mm: float) -> None:
        """Initialize the DoseVoxelKernel.

        Args
        ----
            isotope (str): The isotope name (e.g., 'Lu177').
            voxel_size_mm (float): Voxel size in millimeters.
        """
        kernel_filename = (
            f"voxel_kernels/{isotope}-{voxel_size_mm:1.2f}-mm-mGyperMBqs-SoftICRP.img"
        )
        try:
            with resource_path("pytheranostics.data", kernel_filename) as kernel_path:
                self.kernel = numpy.fromfile(kernel_path, dtype=numpy.float32)
        except FileNotFoundError:
            print(
                f" >> Voxel Kernel for SPECT voxel size ({voxel_size_mm:2.2f} mm) not found. Using default kernel for 4.8 mm voxels..."
            )

            fallback_filename = (
                f"voxel_kernels/{isotope}-4.80-mm-mGyperMBqs-SoftICRP.img"
            )
            with resource_path("pytheranostics.data", fallback_filename) as kernel_path:
                self.kernel = numpy.fromfile(kernel_path, dtype=numpy.float32)

        self.kernel = self.kernel.reshape((51, 51, 51)).astype(numpy.float64)

    def tia_to_dose(
        self, tia_mbq_s: numpy.ndarray, ct: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        """Convert Time-Integrated Activity to dose.

        Parameters
        ----------
        tia_mbq_s : numpy.ndarray
            Time-integrated activity in MBq*s.
        ct : numpy.ndarray, optional
            CT image in HU for density weighting.

        Returns
        -------
        numpy.ndarray
            Dose map in mGy.
        """
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
        """Scale dose per voxel by voxel density.

        This is only valid for voxels of density similar to that of soft tissue and will also improve results for voxels
        with higher density of soft tissue in some instances. However, it will over-estimate doses in voxels with lower density than soft tissue.
        To prevent dose to shoot-up in areas of air where there is activity present (e.g., in the patient's gut), we do not apply scaling based on density in those voxels (i.e., we apply a factor of 1, which is equivalent to saying
        the tissue is ~ soft tissue).

        Args:
            dose_map (numpy.ndarray): Dose-map obtained from convolution of TIA map and Dose Kernel.
            ct (numpy.ndarray): CT image, in HU.

        Returns
        -------
        numpy.ndarray
            Modified Dose-map with dose per voxel scaled-up by density.
        """
        return 1 / hu_to_rho(hu=numpy.clip(ct, 0, 99999)) * dose_map
