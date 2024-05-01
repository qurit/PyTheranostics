from typing import Any, Dict, Optional
from pandas import DataFrame
from doodle.ImagingDS.LongStudy import LongitudinalStudy
from doodle.dosimetry.BaseDosimetry import BaseDosimetry
from doodle.ImagingTools.Tools import itk_image_from_array
from doodle.fits.fits import get_exponential
import numpy
from doodle.dosimetry.dvk import DoseVoxelKernel


class VoxelSDosimetry(BaseDosimetry):
    """Voxel S Dosimetry class: Computes parameters of fit for time activity curves at the region (organ/lesion) level, and
    apply them at the voxel level for voxels belonging to user-defined regions."""
    def __init__(self, 
                 config: Dict[str, Any], 
                 nm_data: LongitudinalStudy, 
                 ct_data: LongitudinalStudy, 
                 clinical_data: Optional[DataFrame] = None) -> None:
        super().__init__(config, nm_data, ct_data, clinical_data)

        # Time-integrated activity and dose maps at the voxel level.
        self.tiac_map: LongitudinalStudy = LongitudinalStudy(images={}, meta={})
        self.dose_map: LongitudinalStudy = LongitudinalStudy(images={}, meta={})
        
        self.toMBqs = 3600  # Convert MBqh toMBqs

    def compute_voxel_tiac(self) -> None:
        """Takes parameters of the fit for each region, and compute TIAC for each voxel
        belonging to each specified region"""

        ref_time_id = self.config["ReferenceTimePoint"]
        tiac_map = numpy.zeros_like(self.nm_data.array_at(time_id=ref_time_id))
        
        # Check we're not having overlapping regions:
        masks = numpy.zeros_like(tiac_map, dtype=numpy.int8)
        
        for region, region_data in self.results.iterrows():
            if "ROB" in region:
                rob_region_data = region_data
                rob_region_mask = self.nm_data.masks[ref_time_id][region]
                rob_act_map_at_ref = self.nm_data.array_of_activity_at(time_id=ref_time_id, region=region) * self.toMBq
                rob_region_tiac = rob_region_data["TIAC_MBq_h"][0]
                rob_region_fit_params = rob_region_data["Fit_params"]
                rob_exp_order = self.config["rois"][region]["fit_order"]
                rob_region_fit, _, _ = get_exponential(order=rob_exp_order, param_init=None, decayconst=1.0)
                rob_ref_time = rob_region_data["Time_hr"][self.config["ReferenceTimePoint"]]
                rob_f_to = rob_region_fit(rob_ref_time, *tuple(rob_region_fit_params[:-1]))
    
                tiac_map += rob_region_mask * rob_region_tiac * rob_act_map_at_ref / rob_f_to  # MBq_h


        for region, region_data in self.results.iterrows():
            
            if region == "BoneMarrow" or region == "ROB":
                continue
                # Temp placeholder.
                
            region_mask = self.nm_data.masks[ref_time_id][region]
            masks += region_mask
            #if numpy.max(masks) > 1:
            #    raise AssertionError(f"Overlapping structures found when {region} was added to calculate voxel-TIAC")

            act_map_at_ref = self.nm_data.array_of_activity_at(time_id=ref_time_id, region=region) * self.toMBq  # MBq
            region_tiac = region_data["TIAC_MBq_h"][0]

            region_fit_params = region_data["Fit_params"]  # fit params and R-square.
            exp_order = self.config["rois"][region]["fit_order"]
            region_fit, _, _ = get_exponential(order=exp_order, param_init=None, decayconst=1.0)  # Decay-constant not used here.
            
            ref_time = region_data["Time_hr"][self.config["ReferenceTimePoint"]]  # In hours, post injection.
            f_to = region_fit(ref_time, *tuple(region_fit_params[:-1]))

            tiac_map += region_mask * region_tiac * act_map_at_ref / f_to  # MBq_h

        # Create ITK Image Object and embed it into a LongStudy.  #TODO: modularize, repeated code downwards.
        tiac_image = itk_image_from_array(array=numpy.transpose(tiac_map, axes=(2, 0, 1)), ref_image=self.nm_data.images[ref_time_id])
        self.tiac_map = LongitudinalStudy(images={0: tiac_image}, meta={0: self.nm_data.meta[ref_time_id]})
        self.tiac_map.add_masks_to_time_point(time_id=0, masks=self.nm_data.masks[0].copy())        

        return None

    def apply_voxel_s(self) -> None:
        """Apply convolution over TIAC map.
        """
        ref_time_id = self.config["ReferenceTimePoint"]
        nm_voxel_mm = self.nm_data.images[ref_time_id].GetSpacing()[0]
        
        dose_kernel = DoseVoxelKernel(isotope=self.nm_data.meta[0]["Radionuclide"], voxel_size_mm=nm_voxel_mm)
        
        dose_map_array = dose_kernel.tiac_to_dose(
            tiac_mbq_s=self.tiac_map.array_at(0) * self.toMBqs, 
            ct=self.ct_data.array_at(time_id=ref_time_id) if self.config["ScaleDoseByDensity"] else None
            )
        
        # Create ITK Image Object and embed it into a LongStudy
        # Clear dose outside patient body:
        #dose_map_array *= self.nm_data.masks[ref_time_id]["WholeBody"]
        dose_map_array *= self.nm_data.masks[ref_time_id]["WBCT"]
        
        self.dose_map = LongitudinalStudy(
            images={
                0: itk_image_from_array(array=numpy.transpose(dose_map_array, axes=(2, 0, 1)), ref_image=self.nm_data.images[ref_time_id])},
            meta={0: self.nm_data.meta[ref_time_id]}
            )
        self.dose_map.add_masks_to_time_point(time_id=0, masks=self.nm_data.masks[0].copy())        
        
        return None
            
    def compute_dose(self) -> None:
        """Steps:
        Compute TIAC at the organ level
        Get parameters of fit and compute TIAC at the voxel level
        Convolve TIAC map with Dose kernel and (optional) scale with CT density.
        """

        self.compute_tiac()
        self.compute_voxel_tiac()
        self.apply_voxel_s()
        # Save dose-map to .nii -> use integer version
        #self.dose_map.save_image_to_nii_at(time_id=0, out_path=self.db_dir, name="DoseMap.nii.gz")
        return None
