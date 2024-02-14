import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy
import pandas

from doodle.fits.fits import fit_tac, get_exponential
from doodle.fits.functions import monoexp_fun, biexp_fun, triexp_fun
from doodle.ImagingDS.LongStudy import LongitudinalStudy
from scipy.integrate import quad


class BaseDosimetry:
    """Base Dosimetry Class. Takes Nuclear Medicine Data and CT Data to perform Organ-level 
    patient-specific dosimetry by computing organ time-integrated activity curves and
    leveraging organ level S-values"""

    def __init__(self, patient_id: str, cycle: int,
                 config: Dict[str, Any],
                 database_dir: str,
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
        
        # Store data
        self.patient_id = patient_id 
        self.cycle = cycle
        self.db_dir = Path(database_dir)
        self.check_patient_in_db() # TODO: Traceability/database?

        self.nm_data = nm_data
        self.ct_data = ct_data
        self.clinical_data = clinical_data

        # Veryfy radionuclide information is present in nm_data.
        self.radionuclide = self.check_nm_data()

        # DataFrame storing results
        self.results = self.initialize_results()
        
    def initialize_results(self) -> pandas.DataFrame:
        """Populates initial result dataframe containing organs of interest, volumes, acquisition times, etc."""
        
        tmp_results: Dict[str, List[float]] = {
            roi_name: [] for roi_name in self.config["rois"] if roi_name != "BoneMarrow"
            }  # BoneMarrow is a special case.
        
        cols: List[str] = ["Time_hr", "Volume_CT_mL", "Activity_Bq"]
        time_ids = [time_id for time_id in self.nm_data.masks.keys]

        for roi_name, _ in self.nm_data.masks[0].items():
            if roi_name not in self.config["rois"]:
                print(f"Skipping {roi_name}, as it was not included in the list of ROIs for dose calculations.")
                continue

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
            
            # Activity, in Bq
            tmp_results[roi_name].append(
                [self.nm_data.sum_of(region=roi_name, time_id=time_id) 
                   for time_id in time_ids]
                   )

        return pandas.DataFrame.from_dict(tmp_results, orient="index", columns=cols)

    def check_nm_data(self) -> Dict[str, Any]:
        """Verify that radionuclide info is present in NM data, and that radionuclide data (e.g., half-life) 
        is available in internal database."""

        # Load Radionuclide data
        with open("../data/isotopes.json", "r") as rad_data:
            radionuclide_data = json.load(rad_data)

        if "radionuclide" not in self.nm_data.meta:
            raise ValueError("Nuclear Medicine Data missing radionuclide")
        
        if self.nm_data.meta["radionuclide"] not in radionuclide_data:
            raise ValueError(f"Data for {self.nm_data.meta['radionuclide']} is not available.")
        
        return radionuclide_data
    
    def check_patient_in_db() -> None:
        """Check if prior dosimetry exists for this patient"""
        raise RuntimeWarning("Database search function not implemented. Dosimetry for this patient might "
                             "already exists...")
        return None
    
    def compute_tiac(self) -> None:
        """Computes Time-Integrated Activity over each source-organ.
        Steps: 

            2.b For Bone-Marrow, if blood-counting, utilize external function.
            2.c For Bone-Marrow, if image-based, utilize atlas method.
            
            3. Integrate each TAC and determine TIAC/MBq (or residence time) for each organ of interest
               and update DataFrame.
            """
        decay_constant = math.log(2) / (self.radionuclide["half_life"] * 3600)  # 1/s

        tmp_tiac_data = {"Fit_params": [], "TIAC_MBq_s": [], "TIAC_h": [], "Lambda_eff": []}

        for region, region_data in self.results.iterrows():

            # TODO: QA of fit. (e.g., plots)
            fit_params, _ = fit_tac(
                time=numpy.array(region_data["Time_hr"]),
                activity=numpy.aarray(region_data["Activity_Bq"]),
                decayconst=decay_constant,
                exp_order=self.config[region]["exp_order"],
                through_origin=self.config[region]["through_origin"]
            )

            # Fitting Parameters
            tmp_tiac_data["Fit_params"].append(fit_params)
            tmp_tiac_data["TIAC_Bq_s"].append(
                self.numerical_integrate(fit_params[:-1])
            )

            # Residence Time
            tmp_tiac_data["TIAC_h"].append(tmp_tiac_data["TIAC_Bq_s"][-1] / (self.nm_data.meta["Injected_Activity"] * 3600))

        for key, values in tmp_tiac_data.items():
            self.results.loc[:, key] = values

        return None
    
    def analytical_integrate(self, exp_params: List[float]) -> float:
        """Perform numerical integration of exponential function"""
        if len(exp_params < 4):
            exp_func = monoexp_fun
        elif len(exp_params < 6):
            exp_func = biexp_fun
        elif len(exp_params) == 6:
            exp_func = triexp_fun
        else:
            raise ValueError("Too many parameters to define a function. Only supported mono, bi or tri-exponentials")
        
        return quad(exp_func, 0, numpy.inf, args=tuple(exp_params))
        