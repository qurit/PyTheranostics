import os
import shutil
from typing import Any, Dict, Optional

import numpy
import pandas
import SimpleITK
from pandas import DataFrame

from pytheranostics.dosimetry.BaseDosimetry import BaseDosimetry
from pytheranostics.dosimetry.dvk import DoseVoxelKernel
from pytheranostics.fits.fits import get_exponential
from pytheranostics.ImagingDS.longitudinal_study import LongitudinalStudy
from pytheranostics.ImagingTools.Tools import itk_image_from_array, resample_to_target


class VoxelSDosimetry(BaseDosimetry):
    """Voxel S Dosimetry class: Computes parameters of fit for time activity curves at the region (organ/lesion) level, and
    apply them at the voxel level for voxels belonging to user-defined regions."""

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
        """
        Computes the Time Integrated Activity (TIA) for each voxel in specified regions.

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
            exp_order = self.config["rois"][region]["fit_order"]
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
        nm_voxel_mm = self.nm_data.images[ref_time_id].GetSpacing()[0]

        dose_kernel = DoseVoxelKernel(
            isotope=self.nm_data.meta[0].Radionuclide, voxel_size_mm=nm_voxel_mm
        )

        # Resample CT to NM (Default using linear interpolator)
        resampled_ct = resample_to_target(
            source_img=self.ct_data.images[ref_time_id],
            target_img=self.nm_data.images[ref_time_id],
        )

        dose_map_array = dose_kernel.tia_to_dose(
            tia_mbq_s=self.tia_map.array_at(0) * self.toMBqs,
            ct=(
                numpy.transpose(
                    SimpleITK.GetArrayFromImage(resampled_ct), axes=(1, 2, 0)
                )
                if self.config["ScaleDoseByDensity"]
                else None
            ),
        )

        # Create ITK Image Object and embed it into a LongitudinalStudy
        # Clear dose outside patient body:
        dose_map_array *= self.nm_data.masks[ref_time_id]["WholeBody"]

        self.dose_map = LongitudinalStudy(
            images={
                0: itk_image_from_array(
                    array=numpy.transpose(dose_map_array, axes=(2, 0, 1)),
                    ref_image=self.nm_data.images[ref_time_id],
                )
            },
            meta={0: self.nm_data.meta[ref_time_id]},
        )

        self.dose_map.masks[0] = self.nm_data.masks[0].copy()

        return None

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

        file_path = os.path.join(
            os.path.dirname(__file__), "../data/monte_carlo/main_template.mac"
        )

        mac_file = numpy.fromfile(file_path, dtype=numpy.float32)

        with open(file_path, "r") as mac_file:
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

        folder_path = os.path.join(
            os.path.dirname(__file__), "../data/monte_carlo/data"
        )
        # Copy files from the source directory to the destination directory
        for file_name in os.listdir(folder_path):
            full_file_name = os.path.join(folder_path, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, os.path.join(output_dir, "data"))

        # List the files in the destination directory to confirm the copy operation
        os.listdir(os.path.join(output_dir, "data"))

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
        """Steps:
        Compute TIA at the region level
        Get parameters of fit from region and compute TIA at the voxel level
        Convolve TIA map with Dose kernel and (optional) scale with CT density.
        """

        self.compute_tia()
        self.compute_voxel_tia()
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
