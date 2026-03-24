"""Module for voxel-level dosimetry calculations."""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy
import pandas
import SimpleITK
from pandas import DataFrame

from pytheranostics.dosimetry.base_dosimetry import BaseDosimetry
from pytheranostics.dosimetry.dvk import DoseVoxelKernel
from pytheranostics.fits.fits import get_exponential
from pytheranostics.imaging_ds.longitudinal_study import LongitudinalStudy
from pytheranostics.imaging_tools.tools import itk_image_from_array, resample_to_target
from pytheranostics.shared.resources import resource_path

logger = logging.getLogger(__name__)


class VoxelSDosimetry(BaseDosimetry):
    """Voxel S Dosimetry class.

    Computes parameters of fit for time activity curves at the region (organ/lesion) level,
    and apply them at the voxel level for voxels belonging to user-defined regions.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        nm_data: LongitudinalStudy,
        ct_data: LongitudinalStudy,
        clinical_data: Optional[DataFrame] = None,
    ) -> None:
        super().__init__(config, nm_data, ct_data, clinical_data)

        # Time-integrated activity and dose maps at the voxel level.
        self.tia_map: LongitudinalStudy = LongitudinalStudy(
            images={}, meta={}, modality="NM"
        )
        self.dose_map: LongitudinalStudy = LongitudinalStudy(
            images={}, meta={}, modality="DOSE"
        )

        self.toMBqs = 3600  # Convert MBqh toMBqs

    def compute_voxel_tia(self) -> None:
        """Compute the Time Integrated Activity (TIA) for each voxel in specified regions.

        This method uses the fit parameters for each region to compute the TIA for
        each voxel within those regions. It handles different regions appropriately,
        ensuring no double-counting or overlapping of regions.

        The result of this operation is tia_map that is a longitudinal study

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If overlapping structures are found when adding regions to calculate voxel-TIA.
        """
        ref_time_id = int(self.config["ReferenceTimePoint"])
        tia_map = numpy.zeros_like(
            self.nm_data.array_at(time_id=ref_time_id), dtype=numpy.float64
        )

        # Check we're not having overlapping regions:
        masks = numpy.zeros_like(tia_map, dtype=numpy.int8)

        for region, region_data in self.results.iterrows():
            
            # Type check:
            if type(region) != str:
                raise TypeError(f"Region names should be strings. Found {type(region)} instead.")

            if region == "WholeBody":
                continue  # We do not want to double count voxels!

            print(f"Computing Voxel-S dose for {region} ...")

            region_mask = self.nm_data.masks[ref_time_id][region]
            masks += region_mask
            if numpy.max(masks) > 1:
                raise AssertionError(
                    f"Overlapping structures found when {region} was added to calculate voxel-TIA"
                )

            act_map_at_ref = (
                self.nm_data.array_of_activity_at(time_id=ref_time_id, region=region)
                * self.toMBq
            )  # MBq
            region_tia = region_data["TIA_MBq_h"]

            region_fit_params = region_data["Fit_params"]  # fit params
            exp_order = self.config["VOIs"][region]["fit_order"]
            region_fit, _, _ = get_exponential(
                order=exp_order, param_init=None, decayconst=1.0
            )  # Decay-constant not used here.

            ref_time = region_data["Time_hr"][
                self.config["ReferenceTimePoint"]
            ]  # In hours, post injection.
            f_to = region_fit(ref_time, *tuple(region_fit_params))

            tia_map += (
                region_mask.astype(numpy.float64) * region_tia * act_map_at_ref / f_to
            )  # MBq_h

        # Create ITK Image Object and embed it into a LongitudinalStudy.  #TODO: modularize, repeated code downwards.
        tia_image = itk_image_from_array(
            array=numpy.transpose(tia_map, axes=(2, 0, 1)),
            ref_image=self.nm_data.images[ref_time_id],
        )
        self.tia_map = LongitudinalStudy(
            images={0: tia_image},
            meta={0: self.nm_data.meta[ref_time_id]},
            modality="NM",
        )
        self.tia_map.masks[0] = self.nm_data.masks[0].copy()  # Copy masks.

        return None

    def apply_voxel_s(self) -> None:
        """Apply convolution over TIA map."""
        ref_time_id = self.config["ReferenceTimePoint"]
        output_grid = str(self.config.get("VoxelSOutputGrid", "NM")).upper()
        logger.info("Applying voxel-S dosimetry using %s output grid.", output_grid)
        output_ref_image, output_mask = self._resolve_output_grid_reference(
            ref_time_id=ref_time_id, output_grid=output_grid
        )
        tia_ref_image = self.tia_map.images[0]
        tia_voxel_mm = self._scalar_voxel_size_mm(tia_ref_image, image_name="TIA map")
        logger.info("Selecting dose kernel from TIA map spacing %.3f mm.", tia_voxel_mm)

        if self.nm_data.meta[0].Radionuclide is None:
            raise ValueError(
                "Radionuclide information is required in nm_data meta to apply voxel S-value convolution."
            )

        dose_kernel = DoseVoxelKernel(
            isotope=self.nm_data.meta[0].Radionuclide, voxel_size_mm=tia_voxel_mm
        )
        logger.info(
            "Using %s kernel with voxel size %.3f mm and matrix size %d.",
            dose_kernel.isotope,
            dose_kernel.voxel_size_mm,
            dose_kernel.matrix_size,
        )
        kernel_ref_image = self._build_reference_image_with_spacing(
            ref_image=tia_ref_image,
            spacing_mm=(dose_kernel.voxel_size_mm, dose_kernel.voxel_size_mm, dose_kernel.voxel_size_mm),
        )
        tia_kernel_grid = self._resample_tia_to_target_grid(target_img=kernel_ref_image)

        logger.info("Resampling CT to the kernel grid for density scaling.")
        resampled_ct = resample_to_target(
            source_img=self.ct_data.images[ref_time_id],
            target_img=kernel_ref_image,
        )

        logger.info("Computing dose on the kernel grid.")
        kernel_dose_array = dose_kernel.tia_to_dose(
            tia_mbq_s=tia_kernel_grid * self.toMBqs,
            ct=(
                numpy.transpose(
                    SimpleITK.GetArrayFromImage(resampled_ct), axes=(1, 2, 0)
                )
                if self.config["ScaleDoseByDensity"]
                else None
            ),
        )
        kernel_dose_img = itk_image_from_array(
            array=numpy.transpose(kernel_dose_array, axes=(2, 0, 1)),
            ref_image=kernel_ref_image,
        )
        logger.info("Resampling dose map from kernel grid to %s output grid.", output_grid)
        dose_map_array = numpy.transpose(
            SimpleITK.GetArrayFromImage(
                resample_to_target(
                    source_img=kernel_dose_img,
                    target_img=output_ref_image,
                    default_value=0.0,
                )
            ),
            axes=(1, 2, 0),
        )

        # Create ITK Image Object and embed it into a LongitudinalStudy
        # Clear dose outside patient body on the chosen output grid when a mask exists.
        if output_mask is not None:
            logger.info("Applying WholeBody mask on the %s output grid.", output_grid)
            dose_map_array *= output_mask

        logger.info("Storing voxel-S dose map on the %s output grid.", output_grid)
        self.dose_map = LongitudinalStudy(
            images={
                0: itk_image_from_array(
                    array=numpy.transpose(dose_map_array, axes=(2, 0, 1)),
                    ref_image=output_ref_image,
                )
            },
            meta={0: self.nm_data.meta[ref_time_id]},
        )

        if output_grid == "NM" and ref_time_id in self.nm_data.masks:
            self.dose_map.masks[0] = self.nm_data.masks[ref_time_id].copy()
        elif output_grid == "CT" and ref_time_id in self.ct_data.masks:
            self.dose_map.masks[0] = self.ct_data.masks[ref_time_id].copy()

        return None

    @staticmethod
    def _build_reference_image_with_spacing(
        ref_image: SimpleITK.Image, spacing_mm: Tuple[float, float, float]
    ) -> SimpleITK.Image:
        """Create a blank reference image using a new spacing on the same physical frame."""
        original_spacing = ref_image.GetSpacing()[:3]
        original_size = ref_image.GetSize()[:3]
        new_size = [
            max(
                1,
                int(
                    round(
                        original_size[idx] * original_spacing[idx] / spacing_mm[idx]
                    )
                ),
            )
            for idx in range(3)
        ]
        target_img = SimpleITK.Image(new_size, ref_image.GetPixelID())
        target_img.SetSpacing(spacing_mm)
        target_img.SetOrigin(ref_image.GetOrigin())
        target_img.SetDirection(ref_image.GetDirection())
        return target_img

    @staticmethod
    def _scalar_voxel_size_mm(image: SimpleITK.Image, image_name: str) -> float:
        """Return a scalar voxel size for isotropic-kernel selection."""
        spacing = tuple(float(value) for value in image.GetSpacing()[:3])
        mean_spacing = float(numpy.mean(spacing))
        if not numpy.allclose(spacing, mean_spacing, atol=0.1):
            logger.warning(
                "%s spacing %s mm is anisotropic. Kernel selection will use the "
                "mean voxel size %.3f mm and resample to the chosen isotropic grid.",
                image_name,
                spacing,
                mean_spacing,
            )
        return mean_spacing

    def _resample_tia_to_target_grid(
        self, target_img: SimpleITK.Image
    ) -> numpy.ndarray:
        """Resample TIA totals per voxel through a temporary density representation."""
        logger.info(
            "Resampling TIA map to kernel grid using voxel-volume normalization to preserve total activity."
        )
        tia_array = self.tia_map.array_at(0).astype(numpy.float64)
        source_voxel_volume_ml = self._voxel_volume_ml(self.tia_map.images[0])
        target_voxel_volume_ml = self._voxel_volume_ml(target_img)

        tia_density = tia_array / source_voxel_volume_ml
        tia_density_img = itk_image_from_array(
            array=numpy.transpose(tia_density, axes=(2, 0, 1)),
            ref_image=self.tia_map.images[0],
        )
        resampled_density_img = resample_to_target(
            source_img=tia_density_img,
            target_img=target_img,
            default_value=0.0,
        )
        resampled_density = numpy.transpose(
            SimpleITK.GetArrayFromImage(resampled_density_img),
            axes=(1, 2, 0),
        ).astype(numpy.float64)
        resampled_tia = resampled_density * target_voxel_volume_ml

        source_total = float(numpy.sum(tia_array))
        resampled_total = float(numpy.sum(resampled_tia))
        logger.info(
            "TIA totals before renormalization: source=%.6g MBq h, resampled=%.6g MBq h.",
            source_total,
            resampled_total,
        )
        if source_total > 0.0 and resampled_total > 0.0:
            resampled_tia *= source_total / resampled_total
            logger.info("Renormalized resampled TIA map to preserve total activity.")

        return resampled_tia

    @staticmethod
    def _voxel_volume_ml(image: SimpleITK.Image) -> float:
        """Return the voxel volume in mL from image spacing in mm."""
        spacing = image.GetSpacing()[:3]
        return float(spacing[0] * spacing[1] * spacing[2] / 1000.0)

    def _resolve_output_grid_reference(
        self, ref_time_id: int, output_grid: str
    ) -> Tuple[SimpleITK.Image, Optional[numpy.ndarray]]:
        """Return the reference image and optional body mask for the requested output grid."""
        if output_grid == "NM":
            return (
                self.nm_data.images[ref_time_id],
                self.nm_data.masks[ref_time_id].get("WholeBody"),
            )
        if output_grid == "CT":
            ct_mask = None
            if ref_time_id in self.ct_data.masks:
                ct_mask = self.ct_data.masks[ref_time_id].get("WholeBody")
            return self.ct_data.images[ref_time_id], ct_mask

        raise ValueError(
            f"VoxelSOutputGrid '{output_grid}' is not supported. Use 'NM' or 'CT'."
        )

    def run_MC(self) -> None:  # TODO: finish the code!!!!!
        """Run MC."""
        raise NotImplementedError("MC is not implemmented yet.")
        n_cpu = self.config["#CPU"]
        n_primaries = self.config["#primaries"]
        output_dir = self.config["results_path"]

        # =============================================================================
        # Split Simulations
        # =============================================================================
        n_primaries_per_mac = int(n_primaries / n_cpu)

        with resource_path(
            "pytheranostics.data", "monte_carlo/main_template.mac"
        ) as template_path:
            with template_path.open("r", encoding="utf-8") as mac_file:
                filedata = mac_file.read()

        for i in range(0, n_cpu):
            new_mac = filedata

            new_mac = new_mac.replace("distrib-SPLIT.mhd", f"distrib_SPLIT_{i+1}.mhd")
            new_mac = new_mac.replace("stat-SPLIT.txt", f"stat__SPLIT_{i+1}.txt")
            new_mac = new_mac.replace("XXX", str(n_primaries_per_mac))

            with open(f"{output_dir}/main_normalized_{i+1}.mac", "w") as output_mac:
                output_mac.write(new_mac)

        # =============================================================================
        # Create Folders with Data
        # =============================================================================
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "output"), exist_ok=True)

        with resource_path("pytheranostics.data", "monte_carlo/data") as folder_path:
            for entry in folder_path.iterdir():
                if entry.is_file():
                    shutil.copy(
                        entry,
                        os.path.join(output_dir, "data", entry.name),
                    )

        # TODO: Below is still work in progress

        # total_acc_A = np.sum(np.sum(np.sum(self.tia_map[0])))
        # self.source_normalized = self.TIAp / self.total_acc_A

        ref_time_id = self.config["ReferenceTimePoint"]
        self.tia_map.save_image_to_mhd_at(
            time_id=0,
            out_path=os.path.join(output_dir, "data"),
            name="Source_normalized",
        )
        self.ct_data.save_image_to_mhd_at(
            time_id=ref_time_id, out_path=os.path.join(output_dir, "data"), name="CT"
        )

        with open(
            os.path.join(os.path.join(output_dir, "output"), "TotalAccA.txt"), "w"
        ) as fileID:
            fileID.write("%.2f" % self.total_acc_A)

        # =============================================================================
        # Run Monte Carlo
        # =============================================================================

        return None

    def compute_dose(self) -> None:
        """Compute dose by performing the following steps.

        Compute TIA at the region level.
        Get parameters of fit from region and compute TIA at the voxel level.
        Convolve TIA map with Dose kernel and (optional) scale with CT density.
        """
        self.compute_tia()
        self.compute_voxel_tia()

        ref_time_id = self.config["ReferenceTimePoint"]
        self.nm_data.save_image_to_nii_at(
            time_id=ref_time_id, out_path=self.db_dir, name="_nm_ref"
        )
        self.ct_data.save_image_to_nii_at(
            time_id=ref_time_id, out_path=self.db_dir, name="_ct_ref"
        )
        self.tia_map.save_image_to_nii_at(
            time_id=0, out_path=self.db_dir, name="_tia_map"
        )

        if self.config["Method"] == "Voxel-S-value":
            self.apply_voxel_s()
        elif self.config["Method"] == "Monte-Carlo":
            self.run_MC()
        else:
            raise ValueError(
                f"Dosimetry Method {self.config['Method']} not implemented."
            )

        # Generate DataFrame.
        dose_Gy = []
        dose_Gy_GBq = []
        for region in self.results.index:

            tmp_Gy = self.dose_map.average_of(region=region, time_id=0) / 1000

            dose_Gy.append(tmp_Gy)
            dose_Gy_GBq.append(tmp_Gy / (float(self.config["InjectedActivity"]) / 1000))

        self.df_ad = pandas.DataFrame(
            {"AD[Gy]": dose_Gy, "AD[Gy/GBq]": dose_Gy_GBq},
            index=[region for region in self.results.index],
        )

        # Save dose-map to .nii -> use integer version
        self.dose_map.save_image_to_nii_at(
            time_id=0, out_path=self.db_dir, name="DoseMap.nii.gz"
        )

        return None
