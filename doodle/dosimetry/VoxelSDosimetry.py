from typing import Any, Dict, Optional
from pandas import DataFrame
from doodle.ImagingDS.LongStudy import LongitudinalStudy
from doodle.dosimetry.BaseDosimetry import BaseDosimetry
from doodle.ImagingTools.Tools import itk_image_from_array
from doodle.fits.fits import get_exponential
import numpy
import shutil
import os
from doodle.dosimetry.DoseVoxelKernel import DoseVoxelKernel


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
        self.tia_map: LongitudinalStudy = LongitudinalStudy(images={}, meta={}, modality="NM")
        self.dose_map: LongitudinalStudy = LongitudinalStudy(images={}, meta={}, modality="DOSE")
        
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

        ref_time_id = self.config["ReferenceTimePoint"]
        tia_map = numpy.zeros_like(self.nm_data.array_at(time_id=ref_time_id))
        
        # Check we're not having overlapping regions:
        masks = numpy.zeros_like(tia_map, dtype=numpy.int8)

        for region, region_data in self.results.iterrows():
            
            
            if region == "WholeBody":
                continue  # We do not want to double count voxels!
                
            region_mask = self.nm_data.masks[ref_time_id][region]
            masks += region_mask
            if numpy.max(masks) > 1:
                raise AssertionError(f"Overlapping structures found when {region} was added to calculate voxel-TIA")

            act_map_at_ref = self.nm_data.array_of_activity_at(time_id=ref_time_id, region=region) * self.toMBq  # MBq
            region_tia = region_data["TIA_MBq_h"][0]

            region_fit_params = region_data["Fit_params"]  # fit params and R-square.
            exp_order = self.config["rois"][region]["fit_order"]
            region_fit, _, _ = get_exponential(order=exp_order, param_init=None, decayconst=1.0)  # Decay-constant not used here.
            
            ref_time = region_data["Time_hr"][self.config["ReferenceTimePoint"]]  # In hours, post injection.
            f_to = region_fit(ref_time, *tuple(region_fit_params[:-1]))

            tia_map += region_mask * region_tia * act_map_at_ref / f_to  # MBq_h

        # Create ITK Image Object and embed it into a LongStudy.  #TODO: modularize, repeated code downwards.
        tia_image = itk_image_from_array(array=numpy.transpose(tia_map, axes=(2, 0, 1)), ref_image=self.nm_data.images[ref_time_id])
        self.tia_map = LongitudinalStudy(images={0: tia_image}, meta={0: self.nm_data.meta[ref_time_id]}, modality="NM")
        self.tia_map.add_masks_to_time_point(time_id=0, masks=self.nm_data.masks[0].copy())        

        return None

    def apply_voxel_s(self) -> None:
        """Apply convolution over TIA map.
        """
        ref_time_id = self.config["ReferenceTimePoint"]
        nm_voxel_mm = self.nm_data.images[ref_time_id].GetSpacing()[0]
        
        dose_kernel = DoseVoxelKernel(isotope=self.nm_data.meta[0]["Radionuclide"], voxel_size_mm=nm_voxel_mm)
        
        dose_map_array = dose_kernel.tia_to_dose(
            tia_mbq_s=self.tia_map.array_at(0) * self.toMBqs, 
            ct=self.ct_data.array_at(time_id=ref_time_id) if self.config["ScaleDoseByDensity"] else None
            )
        
        # Create ITK Image Object and embed it into a LongStudy
        # Clear dose outside patient body:
        dose_map_array *= self.nm_data.masks[ref_time_id]["WholeBody"]
        
        self.dose_map = LongitudinalStudy(
            images={
                0: itk_image_from_array(array=numpy.transpose(dose_map_array, axes=(2, 0, 1)), ref_image=self.nm_data.images[ref_time_id])},
            meta={0: self.nm_data.meta[ref_time_id]}
            )
        self.dose_map.add_masks_to_time_point(time_id=0, masks=self.nm_data.masks[0].copy())        
        
        return None
    
    def run_MC(self) -> None: #### TODO: finish the code!!!!!
        """Run MC.
        """
        
        n_cpu = self.config["#CPU"] 
        n_primaries = self.config["#primaries"]
        output_dir = self.config["results_path"]
        
        ##### SPLIT SIMULATIONS #####
        n_primaries_per_mac = int(n_primaries / n_cpu)
        
        file_path = os.path.join(os.path.dirname(__file__), "../data/monte_carlo/main_template.mac")

        mac_file = numpy.fromfile(file_path,
            dtype=numpy.float32
            )
          
        with open(file_path, 'r') as mac_file:
            filedata = mac_file.read()
        
        for i in range(0, n_cpu):
            new_mac = filedata
            
            new_mac = new_mac.replace('distrib-SPLIT.mhd',f'distrib_SPLIT_{i+1}.mhd')
            new_mac = new_mac.replace('stat-SPLIT.txt',f'stat__SPLIT_{i+1}.txt')    
            new_mac = new_mac.replace('XXX',str(n_primaries_per_mac))
            
            with open(f'{output_dir}/main_normalized_{i+1}.mac','w') as output_mac:
                output_mac.write(new_mac)
                
                
        ##### CREATE FOLDERS WITH DATA #####
        os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "output"), exist_ok=True)

        folder_path = os.path.join(os.path.dirname(__file__), "../data/monte_carlo/data")
        # Copy files from the source directory to the destination directory
        for file_name in os.listdir(folder_path):
            full_file_name = os.path.join(folder_path, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, os.path.join(output_dir, "data"))

        # List the files in the destination directory to confirm the copy operation
        os.listdir(os.path.join(output_dir, "data"))
        
        
        ##### below is still work in progress

        #total_acc_A = np.sum(np.sum(np.sum(self.tia_map[0])))
        #self.source_normalized = self.TIAp / self.total_acc_A
        
        ref_time_id = self.config["ReferenceTimePoint"]
        self.tia_map.save_image_to_mhd_at(time_id=0, out_path=os.path.join(output_dir, "data"), name="Source_normalized")
        self.ct_data.save_image_to_mhd_at(time_id=ref_time_id, out_path=os.path.join(output_dir, "data"), name="CT")

        with open(os.path.join(os.path.join(output_dir, "output"), 'TotalAccA.txt'), 'w') as fileID:
            fileID.write('%.2f' % self.total_acc_A)
        
        ##### RUN MC #####
        
    
        
        return None
            
    def compute_dose(self) -> None:
        """Steps:
        Compute TIA at the region level
        Get parameters of fit from region and compute TIA at the voxel level
        Convolve TIA map with Dose kernel and (optional) scale with CT density.
        """

        self.compute_tia()
        self.compute_voxel_tia()
        if self.config["Method"] == "S-Value":
            self.apply_voxel_s()
        elif self.config["Method"] == "Monte-Carlo":
            self.run_MC()
        # Save dose-map to .nii -> use integer version
        self.dose_map.save_image_to_nii_at(time_id=0, out_path=self.db_dir, name="DoseMap.nii.gz")
        
        return None
