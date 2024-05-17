import json
import math
import abc
from pathlib import Path
from typing import Any, Dict, List, Optional
from os import path
import abc

import numpy
import pandas
from doodle.fits.fits import fit_tac, fit_tac_with_fixed_biokinetics
from doodle.fits.functions import monoexp_fun, biexp_fun, triexp_fun, biexp_fun_uptake
from doodle.plots.plots import plot_tac, plot_tac_fixed_biokinetics
from doodle.ImagingDS.LongStudy import LongitudinalStudy
from scipy.integrate import quad
from doodle.MiscTools.Tools import calculate_time_difference
from doodle.dosimetry.BoneMarrow import bm_scaling_factor
from doodle.ImagingTools.Tools import extract_masks

class BaseDosimetry(metaclass=abc.ABCMeta):

    """Base Dosimetry Class. Takes Nuclear Medicine Data and CT Data to perform Organ-level 
    patient-specific dosimetry by computing organ time-integrated activity curves and
    leveraging organ level S-values"""

    def __init__(self,
                 config: Dict[str, Any],
                 nm_data: LongitudinalStudy, 
                 ct_data: LongitudinalStudy,
                 clinical_data: Optional[pandas.DataFrame] = None
                 ) -> None:
        """Inputs:

            patient_id: patient ID, string.
            cycle: the cycle number (1, 2, ...), int.
            config: Configuration parameters for dosimetry calculations, a Dict.
            database_dir: a folder to store patient-dosimetry results, string.
            nm_data: longitudinal, quantitative, nuclear-medicine imaging data, type LongitudinalStudy.
                     Note: voxel values should be in units of Bq/mL.
            ct_data: longitudinal CT imaging data, type LongitudinalStudy,
                     Note: voxel values should be in HU units.
            clinical_data: clinical data such as blood sampling, an optional pandas DataFrame. 
                     Note: blood counting should be in units of Bq/mL.
        """
        
        # Configuration
        self.config = config
        self.toMBq = 1e-6  # Factor to scale activity from Bq to MBq
        
        if "Apply_biokinetics_from_previous_cycle" not in config:
            self.config["Apply_biokinetics_from_previous_cycle"] = "No"  # By default.
        else:
            if self.config["Apply_biokinetics_from_previous_cycle"] not in ["Yes", "No"]:
                raise ValueError(f"Invalid value for {self.config['Apply_biokinetics_from_previous_cycle']}")
        
        # Store data
        self.patient_id = config["PatientID"] 
        self.cycle = config["Cycle"]
        self.db_dir = Path(config["DatabaseDir"])
        self.check_mandatory_fields()
        self.check_patient_in_db() # TODO: Traceability/database?

        self.nm_data = nm_data
        self.ct_data = ct_data
        self.clinical_data = clinical_data
       
        
        if self.clinical_data is not None and self.clinical_data["PatientID"].unique()[0] != self.patient_id:
            raise AssertionError(f"Clinical Data does not correspond to patient specified by user.")

        # Verify radionuclide information is present in nm_data.
        self.radionuclide = self.check_nm_data()

        # Extract ROIs from user-specified list, and ensure there are no overlaps.
        self.extract_masks_and_correct_overlaps()
        
        # DataFrame storing results
        self.results = self.initialize()

        # Sanity Checks:
        self.sanity_checks(metric="Volume_CT_mL")
        self.sanity_checks(metric="Activity_MBq")
        
        # Dose Maps: use LongitudinalStudy Data Structure to store dose maps and leverage built-in operations.
        self.dose_map: LongitudinalStudy = LongitudinalStudy(images={}, meta={}, modality="DOSE")  # Initialize to empty study.
    
    def extract_masks_and_correct_overlaps(self) -> None:
        """_summary_
        """
        # Inform the user if some masks are unused and therefore excluded.
        for roi_name in self.nm_data.masks[0]:
            if roi_name not in self.config["rois"] and roi_name != "BoneMarrow":
                print(f"Although mask for {roi_name} is present, we are ignoring it because this region was not included in the"
                      " configuration input file.\n")
                continue
            
        self.nm_data.masks = {time_id: extract_masks(
            time_id=time_id, mask_dataset=self.nm_data.masks, requested_rois=list(self.config["rois"].keys())
            ) for time_id in self.nm_data.masks.keys()}
        
        self.ct_data.masks = {time_id: extract_masks(
            time_id=time_id, mask_dataset=self.ct_data.masks, requested_rois=list(self.config["rois"].keys())
            ) for time_id in self.ct_data.masks.keys()}
        
        # Check availability of requested rois in existing masks
        for roi_name in self.config["rois"]:
            if roi_name not in self.nm_data.masks[0] and roi_name != "BoneMarrow":
                raise AssertionError(f"The following mask was NOT found: {roi_name}\n")
            
        # Verify that masks in NM and CT data are consistent (i.e., there is a mask for each region in both domains):
        self.check_nm_ct_masks()
        
        return None

    def check_nm_ct_masks(self) -> None:
        """ Checks that, for each time point, each region contains masks in both NM and CT datasets.
        """
        for time_id, nm_masks in self.nm_data.masks.items():
            nm_masks_list = list(nm_masks.keys())
            ct_masks_list = list(self.ct_data.masks[time_id].keys())
            
            if sorted(nm_masks_list) != sorted(ct_masks_list):
                raise AssertionError(f"Found inconsistent masks at Time ID: {time_id}: \n"
                                     f"NM: {sorted(nm_masks_list)} \n"
                                     f"CT: {sorted(ct_masks_list)}")
                
        return None
    
    def check_mandatory_fields(self) -> None:
        if "InjectionDate" not in self.config or "InjectionTime" not in self.config:
            raise ValueError(f"Incomplete Configuration file.")
        
        if "ReferenceTimePoint" not in self.config:
            print("No Reference Time point was given. Assigning time ID = 0")
            self.config["ReferenceTimePoint"] = 0
        
        if "WholeBody" not in self.config["rois"]:
            raise ValueError("Missing 'WholeBody' region parameters.")
        
        if "RemainderOfBody" not in self.config["rois"]:
            raise ValueError("Missing 'RemainderOfBody' region parameters.")
        
        return None
    
    def initialize(self) -> pandas.DataFrame:
        """Populates initial result dataframe containing organs of interest, volumes, acquisition times, etc."""
        
        tmp_results: Dict[str, List[float]] = {
            roi_name: [] for roi_name in self.nm_data.masks[0].keys() if roi_name != "BoneMarrow" and roi_name in self.config["rois"]
            }  # BoneMarrow is a special case.
        
        cols: List[str] = ["Time_hr", "Volume_CT_mL", "Activity_MBq"]
        time_ids = [time_id for time_id in self.nm_data.masks.keys()]

        # Normalize Acquisition Times, relative to time of injection
        for time_id in self.nm_data.meta.keys():
            self.normalize_time_to_injection(time_id=time_id)

        for roi_name in tmp_results.keys():
            
            # Time (relative to time of injection, in hours)
            tmp_results[roi_name].append(
                [self.nm_data.meta[time_id]["Time"]
                  for time_id in time_ids]
                )

            # Volume (from CT, in mL)
            tmp_results[roi_name].append(
                [self.ct_data.volume_of(region=roi_name, time_id=time_id)
                  for time_id in time_ids]
                )
            
            # Activity, in MBq                
            tmp_results[roi_name].append(
                [self.nm_data.activity_in(region=roi_name, time_id=time_id) * self.toMBq
                   for time_id in time_ids]
                   )

        return pandas.DataFrame.from_dict(
            self.initialize_bone_marrow(tmp_results), 
            orient="index", 
            columns=cols
            )

    def initialize_bone_marrow(self, temp_results: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Initialize activity and times for Bone-Marrow blood-based measurements"""
        
        if "BoneMarrow" in self.config["rois"] and self.clinical_data is not None:
            
            # Computing blood-based method -> Scale activity concentration in blood 
            # to activity in Bone-Marrow, using ICRP phantom mass and haematocrit.
            scaling_factor = bm_scaling_factor(
                gender=self.config["Gender"], 
                hematocrit=self.clinical_data["Haematocrit"].unique()[0]
                )

            temp_results["BoneMarrow"] = [
                 self.clinical_data["Time_hr"].to_list(),
                 self.clinical_data["Volume_mL"].to_list(),
                 [act * scaling_factor * self.toMBq for act in self.clinical_data["Activity_Bq"].to_list()]
            ]
        
        return temp_results

    def check_nm_data(self) -> Dict[str, Any]:
        """Verify that radionuclide info is present in NM data, and that radionuclide data (e.g., half-life) 
        is available in internal database."""

        # Load Radionuclide data
        rad_data_path = path.dirname(__file__) + "/../data/isotopes.json"
        with open(rad_data_path, "r") as rad_data:
            radionuclide_data = json.load(rad_data)

        if "Radionuclide" not in self.nm_data.meta[0]:
            raise ValueError("Nuclear Medicine Data missing radionuclide")
        
        if self.nm_data.meta[0]["Radionuclide"] not in radionuclide_data:
            raise ValueError(f"Data for {self.nm_data.meta['Radionuclide']} is not available.")
        
        return radionuclide_data[self.nm_data.meta[0]["Radionuclide"]]
    
    def check_patient_in_db(self) -> None:
        """Check if prior dosimetry exists for this patient"""
        # TODO: handle logging: error/warnings/prints.
        print("Database search function not implemented. Dosimetry for this patient might "
                             "already exists...")

        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        return None
    
    def sanity_checks(self, metric: str) -> None:
        """Checks that metric in wholebody is equal to sum of metric in individual regions.
        Note: currently excluding BoneMarrow.

        Args:
            metric (str): _description_
        """
        
        if "BoneMarrow" in self.results.index:
            tmp_results = self.results.drop("BoneMarrow", axis=0)
        else:
            tmp_results = self.results.copy()
        
        # TODO: add assertions, run it silently.
        print(" -------------------------------   ")
        print(f"Running Sanity Checks on: {metric}")
        metric_data = tmp_results[metric].to_list()
        times = tmp_results["Time_hr"].to_list()
        
        for time_id in range(len(metric_data[-1])):
            whole_metric = metric_data[-1][time_id]
            sum_metric = sum([vol[time_id] for vol in metric_data[:-1]])
            print(f"At T = {times[0][time_id]:2.2f} hours:")
            print(f" >>> WholeBody {metric}  = {whole_metric: 2.2f}")
            print(f" >>> Regions {metric} = {sum_metric: 2.2f}")
            print(f" >>> % Difference      = {(whole_metric - sum_metric) / whole_metric * 100:2.2f}")
            print(" ")
            
        return None
            
    def compute_tia(self) -> None:
        """Computes Time-Integrated Activity over each source-organ.
        Steps: 

            2.c For Bone-Marrow, if image-based, utilize atlas method. TODO
            
            """
        decay_constant = math.log(2) / (self.radionuclide["half_life"])  # 1/h
        if self.radionuclide["half_life_units"] != "hours":
            raise AssertionError("Radionuclide Half-Life in Database should be in hours.")

        tmp_tia_data = {"Fit_params": [], "TIA_MBq_h": [], "TIA_h": [], "Lambda_eff": []}
        
        if self.config["Apply_biokinetics_from_previous_cycle"] == 'No':
            for region, region_data in self.results.iterrows():
                fit_params, residuals = fit_tac(
                    time=numpy.array(region_data["Time_hr"]),
                    activity=numpy.array(region_data["Activity_MBq"]),
                    decayconst=decay_constant,
                    exp_order=self.config["rois"][region]["fit_order"],
                    param_init=self.config["rois"][region]["param_init"]
                )

                # TODO: Bring plots outside of the main workflow.
                plot_tac(
                    time = numpy.array(region_data["Time_hr"]),
                    activity = numpy.array(region_data["Activity_MBq"]),
                    exp_order = self.config["rois"][region]["fit_order"],
                    parameters = fit_params,
                    residuals = residuals,
                    organ = region, 
                    xlabel = 't (hours)', 
                    ylabel = 'A (MBq)')

                # Fitting Parameters ## TODO: Implement functions from Glatting paper so that unknown parameter is only biological half-life
                tmp_tia_data["Fit_params"].append(fit_params)

                # Calculate Integral
                tmp_tia_data["TIA_MBq_h"].append(
                    self.numerical_integrate(fit_params[:-1])
                )

                # Lambda effective 
                tmp_tia_data["Lambda_eff"].append(fit_params[1])

                # Residence Time
                tmp_tia_data["TIA_h"].append(tmp_tia_data["TIA_MBq_h"][-1][0] / (float(self.config['InjectedActivity'])))
                
        else: # Must be Yes by construction.
            for region, region_data in self.results.iterrows():
                
                fit_params, residuals = fit_tac_with_fixed_biokinetics(
                    time=numpy.array(region_data["Time_hr"]),
                    activity=numpy.array(region_data["Activity_MBq"]),
                    decayconst=decay_constant,
                    exp_order=self.config["rois"][region]["fit_order"],
                    param_init=self.config["rois"][region]["param_init"],
                    fixed_biokinetics = self.config["rois"][region]["fixed_parameters"]
                )

                plot_tac_fixed_biokinetics(
                    time = numpy.array(region_data["Time_hr"]),
                    activity = numpy.array(region_data["Activity_MBq"]),
                    exp_order = self.config["rois"][region]["fit_order"],
                    parameters = fit_params,
                    fixed_biokinetics = self.config["rois"][region]["fixed_parameters"],
                    residuals = residuals,
                    organ = region, 
                    xlabel = 't (hours)', 
                    ylabel = 'A (MBq)')

                # Fitting Parameters ## TODO: Implement functions from Glatting paper so that unknown parameter is only biological half-life
                tmp_tia_data["Fit_params"].append(fit_params)

                # Calculate Integral
                if self.config["rois"][region]["fit_order"] == 1:
                    tmp_tia_data["TIA_MBq_h"].append(
                    quad(monoexp_fun, 0, numpy.inf, args=tuple(numpy.concatenate((fit_params[:-1], (self.config["rois"][region]["fixed_parameters"])))))
                )
                elif self.config["rois"][region]["fit_order"] == 2:
                    tmp_tia_data["TIA_MBq_h"].append(
                    quad(biexp_fun, 0, numpy.inf, args=tuple(numpy.concatenate((fit_params[:-1], (self.config["rois"][region]["fixed_parameters"])))))
                )
                elif self.config["rois"][region]["fit_order"] == -2:
                    tmp_tia_data["TIA_MBq_h"].append(
                    quad(biexp_fun_uptake, 0, numpy.inf, args=tuple(numpy.concatenate((fit_params[:-1], (self.config["rois"][region]["fixed_parameters"])))))
                )
                elif self.config["rois"][region]["fit_order"] == 3:
                    fit_param_0_value = fit_params[0]
                    fit_param_1_value = fit_params[1]
                    fixed_param_0_value = self.config["rois"][region]["fixed_parameters"][0]
                    fixed_param_1_value = self.config["rois"][region]["fixed_parameters"][1]
                    fixed_param_2_value = self.config["rois"][region]["fixed_parameters"][2]

                    # Concatenating arrays
                    concatenated_array = numpy.concatenate((numpy.array([fit_param_0_value]), 
                                                          
                                                         numpy.array([fixed_param_0_value]), 
                                                         numpy.array([fit_param_1_value]),
                                                         numpy.array([fixed_param_1_value]), 
                                                         numpy.array([fixed_param_2_value])))

                    tmp_tia_data["TIA_MBq_h"].append(
                    quad(triexp_fun, 0, numpy.inf, args=tuple(concatenated_array))
                )
                
                tmp_tia_data["Lambda_eff"].append(self.config["rois"][region]["fixed_parameters"])
                tmp_tia_data["TIA_h"].append(tmp_tia_data["TIA_MBq_h"][-1][0] / (float(self.config['InjectedActivity'])))

        for key, values in tmp_tia_data.items():
            self.results.loc[:, key] = values

        return None
    
    def numerical_integrate(self, exp_params: List[float]) -> float:
        """Perform numerical integration of exponential function"""
        if len(exp_params) < 3:
            exp_func = monoexp_fun 
        elif len(exp_params) < 4:
            exp_func = biexp_fun_uptake      
        elif len(exp_params) < 5:
            exp_func = biexp_fun
        elif len(exp_params) < 7:
            exp_func = triexp_fun
        else:
            raise ValueError("Too many parameters to define a function. Only supported mono, bi or tri-exponentials")

        return quad(exp_func, 0, numpy.inf, args=tuple(exp_params))
        
    def normalize_time_to_injection(self, time_id: int) -> None:
        """Express acquisition time corresponding to time_id in terms of injection time"""

        acq_time = f"{self.nm_data.meta[time_id]['AcquisitionDate']} {self.nm_data.meta[time_id]['AcquisitionTime']}"
        inj_time = f"{self.config['InjectionDate']} {self.config['InjectionTime']}"

        self.nm_data.meta[time_id]["Time"] = calculate_time_difference(date_str1=acq_time, date_str2=inj_time)

        return None
    
    @abc.abstractmethod
    def compute_dose(self) -> None:
        """The abstract method to compute Dose to Organs and voxels. Must be implemented in all daughter dosimetry
        classes inheriting from BaseDosimetry. Should run `compute_tia()` first."""
        self.compute_tia()
        return None
        
    def calculate_bed(self,
                      kinetic: str
                      ) -> None:
        """
         monoexp equation based on the paper Bodei et al. "Long-term evaluation of renal toxicity after peptide receptor radionuclide therapy with 90Y-DOTATOC 
         and 177Lu-DOTATATE: the role of associated risk factors"
         biexp   
        """
        this_dir=Path(__file__).resolve().parent.parent
        RADIOBIOLOGY_DATA_FILE = Path(this_dir,"data","radiobiology.json")
        with open(RADIOBIOLOGY_DATA_FILE) as f:
            self.radiobiology_dic = json.load(f)
        bed_df = self.df_ad[self.df_ad.index.isin(list(self.radiobiology_dic.keys()))] # only organs that we know the radiobiology parameters
        organs = numpy.array(bed_df.index.unique())
        bed = {}
        

        for organ in organs:
            t_repair = self.radiobiology_dic[organ]['t_repair']
            alpha_beta = self.radiobiology_dic[organ]['alpha_beta']
            AD = float(bed_df.loc[bed_df.index == organ]['AD[Gy/GBq]'].values[0])  * float(self.config['InjectedActivity']) / 1000 # Gy
            if kinetic == 'monoexp':
                t_eff = numpy.log(2) / self.results.loc[organ]['Fit_params'][1]
                bed[organ] = AD + 1/alpha_beta * t_repair/(t_repair + t_eff) * AD**2
            elif kinetic == 'biexp':
                mean_lambda_washout = (self.results.loc['Kidney_Left']['Fit_params'][1] + self.results.loc['Kidney_Right']['Fit_params'][1]) / 2
                mean_lambda_uptake = (self.results.loc['Kidney_Left']['Fit_params'][2] + self.results.loc['Kidney_Right']['Fit_params'][2]) / 2
                t_washout = numpy.log(2) /  mean_lambda_washout
                t_uptake = numpy.log(2) /  mean_lambda_uptake
                bed[organ] = AD * (1 + (AD / (t_washout - t_uptake)) * (1 / alpha_beta) * (( (2 * t_repair**4 * (t_washout - t_uptake)) / ((t_repair**2 - t_washout**2) * (t_repair**2 - t_uptake**2)) ) + 
                              ((2 * t_washout * t_uptake * t_repair) / (t_washout**2 - t_uptake**2) * (((t_washout)/(t_repair - t_washout)) + ((t_uptake) / (t_repair - t_uptake)))) - 
                              (((t_repair) / (t_washout - t_uptake)) * (((t_washout**2)/(t_repair - t_washout)) + ((t_uptake**2)/(t_repair - t_uptake))))))
            
            print(f'{organ}', bed[organ])
            
        self.df_ad['BED[Gy]'] = self.df_ad.index.map(bed)

    def save_images_and_masks_at(self, time_id: int) -> None:
        """Save CT, NM and masks for a specific time point.

        Args:
            time_id (int): The time point ID.
        """
        
        self.ct_data.save_image_to_nii_at(time_id=time_id, out_path=self.db_dir, name="CT")
        self.nm_data.save_image_to_nii_at(time_id=time_id, out_path=self.db_dir, name="SPECT")
        self.nm_data.save_masks_to_nii_at(time_id=time_id, out_path=self.db_dir, regions=self.config["rois"])
        
        return None