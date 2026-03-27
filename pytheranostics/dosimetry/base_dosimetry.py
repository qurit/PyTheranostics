"""Base dosimetry module for radiation dose calculations."""

import abc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lmfit
import numpy
import pandas

from pytheranostics.dosimetry.bone_marrow import bm_scaling_factor
from pytheranostics.fits.fits import exponential_fit_lmfit
from pytheranostics.imaging_ds.longitudinal_study import LongitudinalStudy
from pytheranostics.imaging_tools.tools import extract_masks
from pytheranostics.misc_tools.tools import calculate_time_difference
from pytheranostics.plots.plots import plot_tac_residuals
from pytheranostics.shared.resources import resource_path


class BaseDosimetry(metaclass=abc.ABCMeta):
    """Base class for performing organ-level patient-specific dosimetry.

    This class provides the foundation for computing organ time-integrated activity curves
    and leveraging organ-level S-values for dosimetry calculations. It handles both Nuclear
    Medicine and CT data to perform comprehensive dosimetry analysis.

    Parameters
    ----------
    nm_data : LongitudinalStudy
            Nuclear Medicine data containing time series of images and masks.
    ct_data : LongitudinalStudy
            CT data containing anatomical information and masks.
    config : dict
            Configuration dictionary containing dosimetry parameters and settings.

    Attributes
    ----------
    nm_data : LongitudinalStudy
            Nuclear Medicine data instance.
    ct_data : LongitudinalStudy
            CT data instance.
    config : dict
            Configuration parameters.
    results : pandas.DataFrame
            DataFrame containing dosimetry results.
    db_dir : Path
            Directory for storing dosimetry results.

    Notes
    -----
    This is an abstract base class that should be subclassed to implement specific
    dosimetry calculation methods.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        nm_data: LongitudinalStudy,
        ct_data: LongitudinalStudy,
        clinical_data: Optional[pandas.DataFrame] = None,
    ) -> None:
        """Initialize the base dosimetry class.

        Parameters
        ----------
        patient_id : str
                Patient ID.
        cycle : int
                The cycle number (1, 2, ...).
        config : Dict
                Configuration parameters for dosimetry calculations.
        database_dir : str
                A folder to store patient-dosimetry results.
        nm_data : LongitudinalStudy
                Longitudinal, quantitative, nuclear-medicine imaging data.
                Note: voxel values should be in units of Bq/mL.
        ct_data : LongitudinalStudy
                Longitudinal CT imaging data.
                Note: voxel values should be in HU units.
        clinical_data : pandas.DataFrame, optional
                Clinical data such as blood sampling.
                Note: blood counting should be in units of Bq/mL.
        """
        # Configuration
        self.config = config
        self.toMBq = 1e-6  # Factor to scale activity from Bq to MBq

        # Store data
        self.patient_id = (
            config["PatientID"] if "PatientID" in config else "UnknownPatient"
        )
        self.cycle = config["Cycle"] if "Cycle" in config else 1
        self.db_dir = (
            Path(config["DatabaseDir"]) if "DatabaseDir" in config else Path("./")
        )

        self.check_mandatory_fields()
        self.check_patient_in_db()  # TODO: Traceability/database?

        self.nm_data = nm_data
        self.nm_data.check_masks_consistency()

        self.ct_data = ct_data
        self.ct_data.check_masks_consistency()

        self.clinical_data = clinical_data

        with resource_path(
            "pytheranostics.data", "s-values/spheres.json"
        ) as spheres_path:
            with spheres_path.open("r", encoding="utf-8") as file:
                self.mass_and_s_values = json.load(file)

        if (
            self.clinical_data is not None
            and self.clinical_data["PatientID"].unique()[0] != self.patient_id
        ):
            raise AssertionError(
                "Clinical Data does not correspond to patient specified by user."
            )

        # Verify radionuclide information is present in nm_data.
        self.radionuclide = self.check_nm_data()

        # Extract ROIs from user-specified list, and ensure there are no overlaps.
        self.extract_masks_and_correct_overlaps()

        # DataFrame storing results
        self.results = self.initialize()
        self.results_dosimetry_lesions = pandas.DataFrame()
        self.results_dosimetry_salivaryglands = pandas.DataFrame()
        self.results_dosimetry_organs = pandas.DataFrame()

        # Sanity Checks:
        self.sanity_checks(metric="Volume_CT_mL")
        self.sanity_checks(metric="Activity_MBq")

        # Handle default values, if missing in config:
        self.default_config()

        # Dose Maps: use LongitudinalStudy Data Structure to store dose maps and leverage built-in operations.
        self.dose_map: LongitudinalStudy = LongitudinalStudy(
            images={}, meta={}, modality="DOSE"
        )  # Initialize to empty study.

    def default_config(self) -> None:
        """Set to None/False the mandatory keys in the config dictionary if not defined.

        We could achieve the same behaviour with dict.get(key, None) but this way we
        inform the user.
        """
        defaults = {
            "fixed_parameters": None,
            "param_init": None,
            "with_uptake": False,
            "fit_order": 1,
            "bounds": None,
            "washout_ratio": None,
        }

        for key, value in defaults.items():
            for region, _ in self.results.iterrows():
                if key not in self.config["VOIs"][region]:
                    self.config["VOIs"][region][key] = value
                    print(
                        f"For {region}, the parameter '{key}' was not defined by the user, set to {value}."
                    )

    def extract_masks_and_correct_overlaps(self) -> None:
        """Extract masks and correct overlaps between regions."""
        # Inform the user if some masks are unused and therefore excluded.
        for roi_name in self.nm_data.masks[0]:
            if roi_name not in self.config["VOIs"] and roi_name != "BoneMarrow":
                print(
                    f"Although mask for {roi_name} is present, we are ignoring it because this region was not included in the"
                    " configuration input file.\n"
                )
                continue

        self.nm_data.masks = {
            time_id: extract_masks(
                time_id=time_id,
                mask_dataset=self.nm_data.masks,
                requested_rois=list(self.config["VOIs"].keys()),
            )
            for time_id in self.nm_data.masks.keys()
        }

        self.ct_data.masks = {
            time_id: extract_masks(
                time_id=time_id,
                mask_dataset=self.ct_data.masks,
                requested_rois=list(self.config["VOIs"].keys()),
            )
            for time_id in self.ct_data.masks.keys()
        }

        # Check availability of requested rois in existing masks
        for roi_name in self.config["VOIs"]:
            if roi_name not in self.nm_data.masks[0] and roi_name != "BoneMarrow":
                raise AssertionError(f"The following mask was NOT found: {roi_name}\n")

        # Verify that masks in NM and CT data are consistent (i.e., there is a mask for each region in both domains):
        self.check_nm_ct_masks()

        return None

    def check_nm_ct_masks(self) -> None:
        """Check that, for each time point, each region contains masks in both NM and CT datasets."""
        for time_id, nm_masks in self.nm_data.masks.items():
            nm_masks_list = list(nm_masks.keys())
            ct_masks_list = list(self.ct_data.masks[time_id].keys())

            if sorted(nm_masks_list) != sorted(ct_masks_list):
                raise AssertionError(
                    f"Found inconsistent masks at Time ID: {time_id}: \n"
                    f"NM: {sorted(nm_masks_list)} \n"
                    f"CT: {sorted(ct_masks_list)}"
                )

        return None

    def check_mandatory_fields(self) -> None:
        """Check for required fields in the configuration.

        Raises
        ------
        ValueError
                If required fields are missing from configuration.
        """
        if "InjectionDate" not in self.config or "InjectionTime" not in self.config:
            raise ValueError("Incomplete Configuration file.")

        if "ReferenceTimePoint" not in self.config:
            print("No Reference Time point was given. Assigning time ID = 0")
            self.config["ReferenceTimePoint"] = 0

        # If WholeBody and RemainderOfBody were not defined by the user, add them by default to the VOIs to ensure consistency with dosimetry calculations.
        for missing in ["WholeBody", "RemainderOfBody"]:
            if missing not in self.config["VOIs"]:
                print(
                    f"Adding {missing} to the list of VOIs with default parameters. This region is required for dosimetry calculations."
                )
                self.config["VOIs"][missing] = {
                    "fit_order": None,
                    "with_uptake": None,
                    "fixed_parameters": None,
                    "bounds": None,
                    "param_init": None,
                }

        if "Organ" in self.config["Level"]:
            if "WholeBody" not in self.config["VOIs"]:
                if "No" in self.config["OrganLevel"]["AdditionalOptions"]["WholeBody"]:
                    pass
                else:
                    raise ValueError("Missing 'WholeBody' region parameters.")

            if "RemainderOfBody" not in self.config["VOIs"]:
                if (
                    "No"
                    in self.config["OrganLevel"]["AdditionalOptions"]["RemainderOfBody"]
                ):
                    pass
                else:
                    raise ValueError("Missing 'RemainderOfBody' region parameters.")

        return None

    def initialize(self) -> pandas.DataFrame:
        """Populate initial result dataframe containing organs of interest, volumes, acquisition times, etc."""
        tmp_results: Dict[str, List[float]] = {
            roi_name: []
            for roi_name in self.nm_data.masks[0].keys()
            if roi_name in self.config["VOIs"]
            or roi_name in ["WholeBody", "RemainderOfBody"]
        }

        cols: List[str] = ["Time_hr", "Volume_CT_mL", "Activity_MBq", "Density_HU"]
        time_ids = [time_id for time_id in self.nm_data.masks.keys()]

        # Normalize Acquisition Times, relative to time of injection
        for time_id in self.nm_data.meta.keys():
            self.normalize_time_to_injection(time_id=time_id)

        for roi_name in tmp_results.keys():

            # Time (relative to time of injection, in hours)
            tmp_results[roi_name].append(
                [self.nm_data.meta[time_id].HoursAfterInjection for time_id in time_ids]
            )

            # Volume (from CT, in mL)
            tmp_results[roi_name].append(
                [
                    self.ct_data.volume_of(region=roi_name, time_id=time_id)
                    for time_id in time_ids
                ]
            )

            # Activity, in MBq
            tmp_results[roi_name].append(
                [
                    self.nm_data.activity_in(region=roi_name, time_id=time_id)
                    * self.toMBq
                    for time_id in time_ids
                ]
            )
            # Density (from CT, in HU)
            tmp_results[roi_name].append(
                [
                    self.ct_data.density_of(region=roi_name, time_id=time_id)
                    for time_id in time_ids
                ]
            )

        return pandas.DataFrame.from_dict(
            self.initialize_bone_marrow(tmp_results), orient="index", columns=cols
        )

    def initialize_bone_marrow(
        self, temp_results: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        """Initialize activity and times for Bone-Marrow blood-based measurements."""
        if (
            "BoneMarrow" in self.config["VOIs"]
            and self.clinical_data is not None
            and "BoneMarrow" not in self.nm_data.masks[0]
        ):

            # Computing blood-based method -> Scale activity concentration in blood
            # to activity in Bone-Marrow, using ICRP phantom mass and haematocrit.
            scaling_factor = bm_scaling_factor(
                gender=self.config["Gender"],
                hematocrit=self.clinical_data["Haematocrit"].unique()[0],
            )

            temp_results["BoneMarrow"] = [
                self.clinical_data["Time_hr"].to_list(),
                self.clinical_data["Volume_mL"].to_list(),
                [
                    act * scaling_factor * self.toMBq
                    for act in self.clinical_data["Activity_Bq"].to_list()
                ],
            ]

        return temp_results

    def check_nm_data(self) -> Dict[str, Any]:
        """Verify that radionuclide info is present in NM data.

        Also verify that radionuclide data (e.g., half-life) is available in internal database.
        """
        # Load Radionuclide data
        with resource_path("pytheranostics.data", "isotopes.json") as rad_data_path:
            with rad_data_path.open("r", encoding="utf-8") as rad_data:
                radionuclide_data = json.load(rad_data)

        if self.nm_data.meta[0].Radionuclide is None:
            raise ValueError("Nuclear Medicine Data missing radionuclide")

        if self.nm_data.meta[0].Radionuclide not in radionuclide_data:
            raise ValueError(
                f"Data for {self.nm_data.meta[0].Radionuclide} is not available."
            )

        return radionuclide_data[self.nm_data.meta[0].Radionuclide]

    def check_patient_in_db(self) -> None:
        """Check if prior dosimetry exists for this patient."""
        # TODO: handle logging: error/warnings/prints.
        print(
            "Database search function not implemented. Dosimetry for this patient might "
            "already exists..."
        )

        self.db_dir.mkdir(parents=True, exist_ok=True)

        return None

    def sanity_checks(self, metric: str) -> None:
        """Check that metric in wholebody is equal to sum of metric in individual regions.

        Note: currently excluding BoneMarrow.

        Args
        ----
                metric (str): The metric to check.
        """
        if (
            "BoneMarrow" in self.results.index
            and "BoneMarrow" not in self.nm_data.masks[0].keys()
        ):
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
            print(
                f" >>> % Difference      = {(whole_metric - sum_metric) / whole_metric * 100:2.2f}"
            )
            print(" ")

        return None

    def compute_tia(self) -> None:
        """Compute Time-Integrated Activity over each source-organ."""
        if self.radionuclide["half_life_units"] != "hours":
            raise AssertionError(
                "Radionuclide Half-Life in Database should be in hours."
            )

        tmp_tia_data = {
            "Fit_params": [],
            "R_squared_AIC": [],
            "TIA_MBq_h": [],
            "TIA_h": [],
            "Lambda_eff": [],
        }

        for region, region_data in self.results.iterrows():

            if not isinstance(region, str):
                raise TypeError(
                    f"Region names should be strings. Found {type(region)} instead."
                )

            fit_results = self.smart_fit_selection(
                region_data=region_data, region=region
            )

            plot_tac_residuals(
                result=fit_results,
                region=region,
                cycle=self.cycle,
                output_dir=self.db_dir,
            )

            # Parameters for sum of exponential functions:
            fit_params = [
                fit_results.params[param].value for param in fit_results.params.keys()
            ]  # A1, B1, A2, B2, ...
            print(fit_results.fit_report())

            # CHECK BOUNDS PHYSICAL DECAY
            # Fitting Parameters ## TODO: Implement functions from Glatting paper so that unknown parameter is only biological half-life
            tmp_tia_data["Fit_params"].append(fit_params)

            # R_Squared and Akaike Information Criterion
            try:
                tmp_tia_data["R_squared_AIC"].append(
                    [fit_results.rsquared, fit_results.aic]
                )
            except AttributeError:
                tmp_tia_data["R_squared_AIC"].append([numpy.nan, numpy.nan])

            # Calculate Integral:
            tmp_tia_data["TIA_MBq_h"].append(
                self.analytical_integrate(result=fit_results)
            )

            # Lambda effective Olny informative for mono-exponential.
            exp_params = [1, 3, 5]
            tmp_tia_data["Lambda_eff"].append(
                [
                    fit_params[exp_params[i]]
                    for i in range(self.config["VOIs"][region]["fit_order"])
                ]
            )

            # Residence Time
            tmp_tia_data["TIA_h"].append(
                tmp_tia_data["TIA_MBq_h"][-1] / (float(self.config["InjectedActivity"]))
            )

        for key, values in tmp_tia_data.items():
            self.results.loc[:, key] = values

        return None

    def smart_fit_selection(
        self, region_data: pandas.Series, region: str
    ) -> lmfit.model.ModelResult:
        """Select the best fit based on Akaike Information Criterion."""
        # If fit_order is defined by user:
        if self.config["VOIs"][region]["fit_order"] is not None:
            print(region)
            fit_results, _ = exponential_fit_lmfit(
                x_data=numpy.array(region_data["Time_hr"]),
                y_data=numpy.array(region_data["Activity_MBq"]),
                fixed_params=self.config["VOIs"][region]["fixed_parameters"],
                num_exponentials=self.config["VOIs"][region]["fit_order"],
                bounds=self.config["VOIs"][region]["bounds"],
                params_init=self.config["VOIs"][region]["param_init"],
                with_uptake=self.config["VOIs"][region]["with_uptake"],
                washout_ratio=self.config["VOIs"][region]["washout_ratio"],
            )

            return fit_results

        print(
            f"WARNING: 'fit_order' for {region} was not specified, finding the best fit from Akaike Information Criteria..."
        )

        # Determine maximum fit order based on avialable data.
        n_samples = numpy.array(region_data["Time_hr"]).shape[0]
        activity_init = region_data["Activity_MBq"][0]

        max_order = min(n_samples // 2, 3)  # Don't use more than tri-exponential.

        all_fits: List[lmfit.model.ModelResult] = []
        fit_config: List[Tuple[bool, int]] = []

        for order in range(1, max_order + 1):
            for with_uptake in [True, False]:

                if order == 1 and with_uptake:
                    continue

                fit_results, _ = exponential_fit_lmfit(
                    x_data=numpy.array(region_data["Time_hr"]),
                    y_data=numpy.array(region_data["Activity_MBq"]),
                    fixed_params=None,
                    num_exponentials=order,
                    bounds=self.config["VOIs"][region]["bounds"],
                    params_init={"A1": activity_init},
                    with_uptake=with_uptake,
                )

                all_fits.append(fit_results)
                fit_config.append((with_uptake, order))

        # Apply Criterion
        aic_results = [(idx, fit.aic) for idx, fit in enumerate(all_fits)]
        aic_results = sorted(aic_results, key=lambda x: x[1])  # Sort

        # If only one model fit, that is the winner.
        if len(aic_results) == 1:
            self.config["VOIs"][region]["with_uptake"] = fit_config[0][0]
            self.config["VOIs"][region]["fit_order"] = fit_config[0][1]
            return all_fits[0]

        # If there are two more models, we check the top two models and compare their AIC. If the difference
        # in AIC is less than 2, we pick the model with the lowest number of parameters.

        best_model_idx = aic_results[0][0]

        if (
            aic_results[1][1] - aic_results[0][1] <= 2
            and all_fits[aic_results[0][0]].nvarys > all_fits[aic_results[1][0]].nvarys
        ):
            best_model_idx = aic_results[1][0]

        self.config["VOIs"][region]["with_uptake"] = fit_config[best_model_idx][0]
        self.config["VOIs"][region]["fit_order"] = fit_config[best_model_idx][1]

        return all_fits[best_model_idx]

    def analytical_integrate(self, result: lmfit.model.ModelResult) -> float:
        """Compute the analytical integral of a fitted exponential function.

        This method calculates the analytical integral from 0 to infinity of a
        fitted exponential function. It handles mono-, bi-, and tri-exponential
        functions by summing the integrals of individual exponential terms.

        Parameters
        ----------
        result : lmfit.model.ModelResult
                The result object from fitting an exponential function using lmfit.
                Should contain parameters for the exponential terms (A1, A2, B1, B2, etc.).

        Returns
        -------
        float
                The computed integral value.

        Notes
        -----
        - For each exponential term, the integral is computed as A1/A2 where:
          - A1 is the amplitude parameter
          - A2 is the decay constant
        - Terms with non-positive decay constants are ignored
        - The function handles missing parameters gracefully
        """
        # Extract the parameter values from the result
        params = result.params.valuesdict()

        # Initialize integral
        integral = 0.0

        # Loop over the possible exponential terms
        num_exponentials = len(params) // 2  # Each exponential has two parameters
        terms = ["A", "B", "C"][:num_exponentials]

        for term in terms:
            A1_name = f"{term}1"
            A2_name = f"{term}2"
            if A1_name in params and A2_name in params:
                A1 = params[A1_name]
                A2 = params[A2_name]
                if A2 > 0:
                    integral += A1 / A2
                else:
                    # Handle the case where A2 is zero or negative
                    print(
                        f"Warning: Decay constant {A2_name} is non-positive ({A2}). Term is ignored in integral calculation."
                    )
            else:
                # Parameters for this term are not present in the fit
                continue

        return integral

    def normalize_time_to_injection(self, time_id: int) -> None:
        """Express acquisition time corresponding to time_id in terms of injection time."""
        acq_time = f"{self.nm_data.meta[time_id].AcquisitionDate} {self.nm_data.meta[time_id].AcquisitionTime}"
        inj_time = f"{self.config['InjectionDate']} {self.config['InjectionTime']}"

        self.nm_data.meta[time_id].HoursAfterInjection = calculate_time_difference(
            date_str1=acq_time, date_str2=inj_time
        )

        return None

    @abc.abstractmethod
    def compute_dose(self) -> None:
        """Compute Dose to Organs and voxels.

        This abstract method must be implemented in all daughter dosimetry classes inheriting
        from BaseDosimetry. Should run `compute_tia()` first.
        """
        self.compute_tia()
        return None

    def calculate_bed(self, kinetic: str) -> None:
        """Calculate Biologically Effective Dose (BED).

        Monoexp equation based on the paper Bodei et al. "Long-term evaluation of renal toxicity
        after peptide receptor radionuclide therapy with 90Y-DOTATOC and 177Lu-DOTATATE: the role
        of associated risk factors".
        """
        this_dir = Path(__file__).resolve().parent.parent
        RADIOBIOLOGY_DATA_FILE = Path(this_dir, "data", "radiobiology.json")
        with open(RADIOBIOLOGY_DATA_FILE) as f:
            self.radiobiology_dic = json.load(f)
        bed_df = self.results_dosimetry_organs[
            self.results_dosimetry_organs.index.isin(list(self.radiobiology_dic.keys()))
        ]  # only organs that we know the radiobiology parameters
        organs = numpy.array(bed_df.index.unique())
        bed = {}

        for organ in organs:
            t_repair = self.radiobiology_dic[organ]["t_repair"]
            alpha_beta = self.radiobiology_dic[organ]["alpha_beta"]
            AD = (
                float(bed_df.loc[bed_df.index == organ]["AD_total[Gy/GBq]"].values[0])
                * float(self.config["InjectedActivity"])
                / 1000
            )  # Gy

            if kinetic == "monoexp":
                # gather existing kidneys dynamically
                kidney_labels = [
                    s for s in self.results.index if s.startswith("Kidney_")
                ]

                # extract alpha parameters for those that exist
                alphas = [self.results.loc[k]["Fit_params"][1] for k in kidney_labels]

                # compute effective half-time using the mean alpha
                alpha_mean = numpy.mean(alphas)
                t_eff = numpy.log(2) / alpha_mean

                bed[organ] = AD + 1 / alpha_beta * t_repair / (t_repair + t_eff) * AD**2

            elif kinetic == "biexp":
                mean_lambda_washout = (
                    self.results.loc["Kidney_Left"]["Fit_params"][1]
                    + self.results.loc["Kidney_Right"]["Fit_params"][1]
                ) / 2
                mean_lambda_uptake = (
                    self.results.loc["Kidney_Left"]["Fit_params"][2]
                    + self.results.loc["Kidney_Right"]["Fit_params"][2]
                ) / 2
                t_washout = numpy.log(2) / mean_lambda_washout
                t_uptake = numpy.log(2) / mean_lambda_uptake
                bed[organ] = AD * (
                    1
                    + (AD / (t_washout - t_uptake))
                    * (1 / alpha_beta)
                    * (
                        (
                            (2 * t_repair**4 * (t_washout - t_uptake))
                            / (
                                (t_repair**2 - t_washout**2)
                                * (t_repair**2 - t_uptake**2)
                            )
                        )
                        + (
                            (2 * t_washout * t_uptake * t_repair)
                            / (t_washout**2 - t_uptake**2)
                            * (
                                ((t_washout) / (t_repair - t_washout))
                                + ((t_uptake) / (t_repair - t_uptake))
                            )
                        )
                        - (
                            ((t_repair) / (t_washout - t_uptake))
                            * (
                                ((t_washout**2) / (t_repair - t_washout))
                                + ((t_uptake**2) / (t_repair - t_uptake))
                            )
                        )
                    )
                )
            print(f"{organ}", bed[organ])

        self.results_dosimetry_organs["BED[Gy]"] = (
            self.results_dosimetry_organs.index.map(bed)
        )

    def save_images_and_masks_at(self, time_id: int) -> None:
        """Save CT, NM and masks for a specific time point.

        Args
        ----
                time_id (int): The time point ID.
        """
        self.ct_data.save_image_to_nii_at(
            time_id=time_id, out_path=self.db_dir, name="CT"
        )
        self.nm_data.save_image_to_nii_at(
            time_id=time_id, out_path=self.db_dir, name="SPECT"
        )
        self.nm_data.save_masks_to_nii_at(
            time_id=time_id, out_path=self.db_dir, regions=self.config["VOIs"]
        )

        return None

    def write_json_data(
        self,
        file_path,
        InstitutionName: str,
        ClinicalTrial: str,
        Radionuclide: str,
        create_new: bool = True,
    ) -> None:
        """Write dosimetry results to a JSON file.

        If create_new is True, a new JSON file is created. If False, existing data is updated,
        usually used to add results from subsequent cycles.

        Args
        ----
                file_path (str): Path to the JSON file.
                create_new (bool): Whether to create a new file or update existing data.
        """
        # Open empty json to load its structure:
        if create_new:
            with resource_path("pytheranostics.data", "output.json") as template_json:
                with template_json.open("r", encoding="utf-8") as file:
                    data = json.load(file)
        else:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

        data["PatientID"] = self.config["PatientID"]
        data["InstitutionName"] = InstitutionName
        data["ClinicalTrial"] = ClinicalTrial
        data["Radionuclide"] = Radionuclide
        data["Gender"] = self.config["Gender"]
        data["No_of_completed_cycles"] = self.config["Cycle"]

        cycle_key = f"Cycle_{self.config['Cycle']:02d}"

        if cycle_key not in data:
            data[cycle_key] = [{}]
        else:
            print(
                f"WARNING: Might be Overwiting existing data. {cycle_key} in Patient {data['PatientID']} already exists."
            )

        cycle = data[cycle_key][0]
        cycle["CycleNumber"] = self.config["Cycle"]
        cycle["Operator"] = self.config["Operator"]
        cycle["DatabaseDir"] = self.config["DatabaseDir"]
        cycle["InjectionDate"] = self.config["InjectionDate"]
        cycle["InjectionTime"] = self.config["InjectionTime"]
        cycle["InjectedActivity"] = self.config["InjectedActivity"]
        cycle["Weight_g"] = self.config["PatientWeight_g"]
        cycle["Height_cm"] = self.config["PatientHeight_cm"]
        cycle["Level"] = self.config["Level"]
        if cycle["Level"] == "Organ":
            cycle["Method"] = self.config["OrganLevel"]
        elif cycle["Level"] == "Voxel":
            cycle["Method"] = self.config["VoxelLevel"]
            cycle["ScaleDoseByDensity"] = self.config.get(
                "ScaleDoseByDensity", cycle.get("ScaleDoseByDensity", "NA")
            )
        cycle["ReferenceTimePoint"] = self.config["ReferenceTimePoint"]
        cycle["TimePoints_h"] = self.results["Time_hr"][0]

        for organ in self.config["VOIs"].keys():
            if organ not in cycle["VOIs"]:
                cycle["VOIs"][organ] = {
                    "volumes_mL": {},
                    "activity_MBq": {},
                    "timepoints_h": {},
                    "doserate_MBq_per_h": {},
                    "density_HU": {},
                    "density_gml": {},
                    "mass_g": {},
                    "composition": {},
                    "fitting_eq": {},
                    "no_of_fit_params": {},
                    "fit_params": {},
                    "fit_params_uncertainty": {},
                    "R_2": {},
                    "AIC": {},
                    "TIA_MBqh": {},
                    "TIA_MBqh_uncertainty": {},
                    "TIA_h": {},
                    "TIA_h_uncertainty": {},
                    "total_s_value": {},
                    "total_s_value_uncertainty": {},
                    "mean_AD_Gy": {},
                    "mean_AD_Gy_uncertainty": {},
                    "min_AD_Gy": {},
                    "max_AD_Gy": {},
                    "peak_AD_Gy": {},
                    "repair_halflife": {},
                    "alpha_beta": {},
                    "BED_Gy": {},
                    "BED_Gy_uncertainty": {},
                }

            cycle["VOIs"][organ]["volumes_mL"]["different_tps"] = self.results.loc[
                organ, "Volume_CT_mL"
            ]
            cycle["VOIs"][organ]["volumes_mL"]["uncertainty"] = "NA"
            cycle["VOIs"][organ]["volumes_mL"]["mean"] = numpy.mean(
                self.results.loc[organ, "Volume_CT_mL"]
            )
            cycle["VOIs"][organ]["volumes_mL"]["mean_uncertainty"] = "NA"
            cycle["VOIs"][organ]["activity_MBq"]["values"] = [
                float(x) for x in self.results.loc[organ, "Activity_MBq"]
            ]
            cycle["VOIs"][organ]["activity_MBq"]["uncertainty"] = "NA"
            cycle["VOIs"][organ]["timepoints_h"]["values"] = self.results.loc[
                organ, "Time_hr"
            ]
            cycle["VOIs"][organ]["doserate_MBq_per_h"]["values"] = "NA"
            cycle["VOIs"][organ]["doserate_MBq_per_h"]["uncertainty"] = "NA"
            try:
                cycle["VOIs"][organ]["density_HU"]["different_tps"] = self.results.loc[
                    organ, "Density_HU"
                ]
            except (KeyError, AttributeError):  # TODO: Handle errors explicitly
                pass
            cycle["VOIs"][organ]["density_HU"]["uncertainty"] = "NA"
            try:
                cycle["VOIs"][organ]["density_HU"]["mean"] = numpy.mean(
                    self.results.loc[organ, "Density_HU"]
                )
            except (
                KeyError,
                AttributeError,
                TypeError,
            ):  # TODO: Handle errors explicitly
                pass
            cycle["VOIs"][organ]["density_HU"]["mean_uncertainty"] = "NA"
            cycle["VOIs"][organ]["density_gml"]["different_tps"] = "NA"
            cycle["VOIs"][organ]["density_gml"]["uncertainty"] = "NA"
            cycle["VOIs"][organ]["density_gml"]["mean"] = "NA"
            cycle["VOIs"][organ]["density_gml"]["mean_uncertainty"] = "NA"
            cycle["VOIs"][organ]["mass_g"]["different_tps"] = "NA"
            cycle["VOIs"][organ]["mass_g"]["uncertainty"] = "NA"
            cycle["VOIs"][organ]["mass_g"]["mean"] = "NA"
            cycle["VOIs"][organ]["mass_g"]["mean_uncertainty"] = "NA"
            cycle["VOIs"][organ]["fitting_eq"] = self.config["VOIs"][organ]["fit_order"]
            cycle["VOIs"][organ]["no_of_fit_params"] = "NA"
            cycle["VOIs"][organ]["fit_params"] = list(
                self.results.loc[organ, "Fit_params"]
            )
            cycle["VOIs"][organ]["washout_ratio"] = self.config["VOIs"][organ][
                "washout_ratio"
            ]
            cycle["VOIs"][organ]["fit_params_uncertainty"] = "NA"
            cycle["VOIs"][organ]["R_2"] = (
                "NA"
                if pandas.isna(self.results.loc[organ, "R_squared_AIC"][0])
                else self.results.loc[organ, "R_squared_AIC"][0]
            )
            cycle["VOIs"][organ]["AIC"] = (
                "NA"
                if pandas.isna(self.results.loc[organ, "R_squared_AIC"][1])
                else self.results.loc[organ, "R_squared_AIC"][1]
            )
            cycle["VOIs"][organ]["TIA_MBqh"] = self.results.loc[organ, "TIA_MBq_h"]
            cycle["VOIs"][organ]["TIA_MBqh_uncertainty"] = "NA"
            cycle["VOIs"][organ]["TIA_h"] = self.results.loc[organ, "TIA_h"]
            cycle["VOIs"][organ]["TIA_h_uncertainty"] = "NA"
            cycle["VOIs"][organ]["mean_AD_Gy"] = "NA"
            cycle["VOIs"][organ]["mean_AD_Gy_uncertainty"] = "NA"
            cycle["VOIs"][organ]["min_AD_Gy"] = "NA"
            cycle["VOIs"][organ]["max_AD_Gy"] = "NA"
            cycle["VOIs"][organ]["peak_AD_Gy"] = "NA"
            cycle["VOIs"][organ]["repair_halflife"] = "NA"
            cycle["VOIs"][organ]["alpha_beta"] = "NA"
            cycle["VOIs"][organ]["composition"] = "NA"
            cycle["VOIs"][organ]["total_s_value"] = "NA"
            cycle["VOIs"][organ]["total_s_value_uncertainty"] = "NA"

            if "Lesion" in organ or "TTB" in organ:
                cycle["VOIs"][organ]["density_gml"]["different_tps"] = "NA"
                cycle["VOIs"][organ]["density_gml"]["uncertainty"] = "NA"
                cycle["VOIs"][organ]["density_gml"]["mean"] = (
                    self.results_dosimetry_lesions.loc[organ, "Density_g_per_mL"]
                )
                cycle["VOIs"][organ]["density_gml"]["mean_uncertainty"] = "NA"
                cycle["VOIs"][organ]["mass_g"]["different_tps"] = "NA"
                cycle["VOIs"][organ]["mass_g"]["uncertainty"] = "NA"
                cycle["VOIs"][organ]["mass_g"]["mean"] = (
                    self.results_dosimetry_lesions.loc[organ, "Mass_g"]
                )
                cycle["VOIs"][organ]["mass_g"]["mean_uncertainty"] = "NA"
                cycle["VOIs"][organ]["composition"] = (
                    self.results_dosimetry_lesions.loc[organ, "Composition"]
                )
                cycle["VOIs"][organ]["total_s_value"] = (
                    self.results_dosimetry_lesions.loc[organ, "Total_S_Value"]
                )
                cycle["VOIs"][organ]["total_s_value_uncertainty"] = "NA"
                cycle["VOIs"][organ]["mean_AD_Gy"] = self.results_dosimetry_lesions.loc[
                    organ, "AD_Gy"
                ]
                cycle["VOIs"][organ]["mean_AD_Gy_uncertainty"] = "NA"

            if "BoneMarrow" in organ:
                cycle["VOIs"][organ]["volumes_mL"]["different_tps"] = 1170
                cycle["VOIs"][organ]["volumes_mL"]["uncertainty"] = "NA"
                cycle["VOIs"][organ]["volumes_mL"]["mean"] = 1170

            if "Gland" in organ:
                cycle["VOIs"][organ]["density_gml"]["different_tps"] = "NA"
                cycle["VOIs"][organ]["density_gml"]["uncertainty"] = "NA"
                cycle["VOIs"][organ]["density_gml"]["mean"] = (
                    self.results_dosimetry_salivaryglands.loc[organ, "Density_g_per_mL"]
                )
                cycle["VOIs"][organ]["density_gml"]["mean_uncertainty"] = "NA"
                cycle["VOIs"][organ]["mass_g"]["different_tps"] = "NA"
                cycle["VOIs"][organ]["mass_g"]["uncertainty"] = "NA"
                cycle["VOIs"][organ]["mass_g"]["mean"] = (
                    self.results_dosimetry_salivaryglands.loc[organ, "Mass_g"]
                )
                cycle["VOIs"][organ]["mass_g"]["mean_uncertainty"] = "NA"
                cycle["VOIs"][organ]["composition"] = (
                    self.results_dosimetry_salivaryglands.loc[organ, "Composition"]
                )
                cycle["VOIs"][organ]["total_s_value"] = (
                    self.results_dosimetry_salivaryglands.loc[organ, "Total_S_Value"]
                )
                cycle["VOIs"][organ]["total_s_value_uncertainty"] = "NA"
                cycle["VOIs"][organ]["mean_AD_Gy"] = (
                    self.results_dosimetry_salivaryglands.loc[organ, "AD_Gy"]
                )
                cycle["VOIs"][organ]["mean_AD_Gy_uncertainty"] = "NA"

        if self.config["Level"] == "Organ":
            for organ in self.results_dosimetry_organs.index:
                if organ in self.results_dosimetry_organs.index:
                    cycle["Organ-level_AD"][organ] = {
                        "AD[Gy/GBq]": {},
                        "AD[Gy/GBq]_uncertianty": {},
                        "AD[Gy]": {},
                        "AD[Gy]_uncertianty": {},
                        "BED[Gy]": {},
                        "BED[Gy]_uncertianty": {},
                    }
                cycle["Organ-level_AD"][organ]["AD[Gy/GBq]"] = (
                    self.results_dosimetry_organs.loc[organ, "AD_total[Gy/GBq]"]
                )
                cycle["Organ-level_AD"][organ]["AD[Gy/GBq]_uncertainty"] = "NA"
                cycle["Organ-level_AD"][organ]["AD[Gy]"] = (
                    self.results_dosimetry_organs.loc[organ, "AD_total[Gy]"]
                )
                cycle["Organ-level_AD"][organ]["AD[Gy]_uncertainty"] = "NA"

                if "BED[Gy]" in self.results_dosimetry_organs.columns:
                    cycle["Organ-level_AD"][organ]["BED[Gy]"] = (
                        self.results_dosimetry_organs.loc[organ, "BED[Gy]"]
                        if pandas.notna(
                            self.results_dosimetry_organs.loc[organ, "BED[Gy]"]
                        )
                        else "NA"
                    )
                else:
                    cycle["Organ-level_AD"][organ]["BED[Gy]"] = "NA"

                cycle["Organ-level_AD"][organ]["BED[Gy]_uncertianty"] = "NA"

        if "Yes" in self.config["OrganLevel"]["AdditionalOptions"].get(
            "LesionDosimetry"
        ):
            cycle["Organ-level_AD"]["TTB"] = {
                "mass_g": {},
                "volumes_mL": {},
                "TIA_h": {},
                "AD[Gy]": {},
                "AD[Gy]_uncertianty": {},
                "AD[Gy/GBq]": {},
                "AD[Gy/GBq]_uncertianty": {},
            }
            cycle["Organ-level_AD"]["TTB"]["mass_g"] = (
                self.results_dosimetry_lesions.loc["TTB", "Mass_g"]
            )
            cycle["Organ-level_AD"]["TTB"]["volumes_mL"] = (
                self.results_dosimetry_lesions.loc["TTB", "Volume_CT_mL"]
            )
            cycle["Organ-level_AD"]["TTB"]["TIA_h"] = (
                self.results_dosimetry_lesions.loc["TTB", "TIA_h"]
            )
            cycle["Organ-level_AD"]["TTB"]["AD[Gy]"] = (
                self.results_dosimetry_lesions.loc["TTB", "AD_Gy"]
            )
            cycle["Organ-level_AD"]["TTB"]["AD[Gy]_uncertainty"] = "NA"
            cycle["Organ-level_AD"]["TTB"]["AD[Gy/GBq]"] = (
                self.results_dosimetry_lesions.loc["TTB", "AD_Gy"]
                / (float(self.config["InjectedActivity"]) / 1000)
            )
            cycle["Organ-level_AD"]["TTB"]["AD[Gy/GBq]_uncertianty"] = "NA"

        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
