"""Miscellaneous utility tools for image processing and analysis."""

from datetime import datetime
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy
from scipy.ndimage import median_filter


def hu_to_rho(hu: numpy.ndarray) -> numpy.ndarray:
    """Convert a CT array in HU into a density map in g/cc.

    Conversion based on Schneider et al. 2000 (using GATE's material db example).

    Args:
        hu (numpy.ndarray): CT array in Hounsfield Units.

    Returns
    -------
    numpy.ndarray
        Density map in g/cc.
    """
    # Define the bin edges for HU values
    bins = numpy.array(
        [
            -1050,
            -950,
            -852.884,
            -755.769,
            -658.653,
            -561.538,
            -464.422,
            -367.306,
            -270.191,
            -173.075,
            -120,
            -82,
            -52,
            -22,
            8,
            19,
            80,
            120,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1640,
            1807.5,
            1975.01,
            2142.51,
            2300,
            2467.5,
            2635.01,
            2802.51,
            2970.02,
            3000,
        ]
    )

    # Define the corresponding density values for each bin
    values = numpy.array(
        [
            0.00121,
            0.102695,
            0.202695,
            0.302695,
            0.402695,
            0.502695,
            0.602695,
            0.702695,
            0.802695,
            0.880021,
            0.926911,
            0.957382,
            0.984277,
            1.01117,
            1.02955,
            1.0616,
            1.1199,
            1.11115,
            1.16447,
            1.22371,
            1.28295,
            1.34219,
            1.40142,
            1.46066,
            1.5199,
            1.57914,
            1.63838,
            1.69762,
            1.75686,
            1.8161,
            1.87534,
            1.94643,
            2.03808,
            2.13808,
            2.23808,
            2.33509,
            2.4321,
            2.5321,
            2.6321,
            2.7321,
            2.79105,
            2.9,
        ]
    )

    # Clip the HU array values to be within the range of defined bins
    hu_clipped = numpy.clip(hu, bins[0], bins[-1])

    # Apply Median filter to remove a bit of remaining noise
    hu_clipped = median_filter(hu_clipped, size=2)

    # Find the corresponding bin for each HU value
    bin_indices = numpy.digitize(hu_clipped, bins, right=True)

    # Map each bin index to the corresponding density value
    rho = values[bin_indices - 1]

    return rho


def calculate_time_difference(
    date_str1: str, date_str2: str, date_format: str = "%Y%m%d %H%M%S"
) -> float:
    """Calculate the time difference in hours between two dates.

    This function computes the time difference between two dates provided as strings.
    The dates should be in the format specified by date_format.

    Parameters
    ----------
    date_str1 : str
        First date string.
    date_str2 : str
        Second date string.
    date_format : str, optional
        Format string for parsing the dates, by default "%Y%m%d %H%M%S".

    Returns
    -------
    float
        Time difference in hours.

    Notes
    -----
    - The function removes any fractional seconds from the input strings.
    - The time difference is calculated as (date_str1 - date_str2).
    - The result is returned in hours as a float value.
    """
    # Remove fractional seconds if present
    # Clean up:
    date_str1 = date_str1.split(".")[0]
    date_str2 = date_str2.split(".")[0]

    # Convert string dates to datetime objects
    datetime1 = datetime.strptime(date_str1, date_format)
    datetime2 = datetime.strptime(date_str2, date_format)

    # Calculate the difference in hours
    time_diff = datetime1 - datetime2
    hours_diff = time_diff.total_seconds() / 3600

    return hours_diff


# New functions to extract parameters from JSON.
def extract_exponential_params_from_json(
    json_data: dict, cycle: str, region: str
) -> Tuple[Dict[str, float], bool, Dict[str, float]]:
    """Extract parameters of fit for a defined region and cycle from a JSON dictionary of a patient.

    Parameters
    ----------
    json_data : dict
        The patient's JSON dictionary.
    cycle : str
        The cycle ID.
    region : str
        The region of interest.

    Returns
    -------
    Tuple[Dict[str, float], bool, Dict[str, float]]
        A Tuple consisting of the exponential parameters from previous fit,
        a boolean representing whether or not initial uptake was accounted for in previous fit,
        and all the parameters of the fit.
    """
    with_uptake = False
    # Determine order:
    parameters = json_data[cycle][0]["VOIs"][region]["fit_params"]
    washout_ratio = json_data[cycle][0]["VOIs"][region]["washout_ratio"]

    # Handle Legacy:
    if len(parameters) in [3, 5]:
        return extract_exponential_params_from_json_legacy(json_data, cycle, region)

    fit_order = len(parameters) // 2

    exponential_idxs = [1, 3, 5]
    param_name_base = ["A", "B", "C"]

    fixed_parameters: Dict[str, float] = {}
    all_parameters: Dict[str, float] = {}
    for order in range(fit_order):
        fixed_parameters[f"{param_name_base[order]}2"] = parameters[
            exponential_idxs[order]
        ]

        all_parameters[f"{param_name_base[order]}1"] = parameters[
            exponential_idxs[order] - 1
        ]
        all_parameters[f"{param_name_base[order]}2"] = parameters[
            exponential_idxs[order]
        ]
    if washout_ratio is not None:
        # Fix A1 ONLY
        if "B1" in all_parameters:
            fixed_parameters["B1"] = all_parameters["B1"]
    if fit_order == 2 and parameters[0] == -parameters[2]:
        with_uptake = True

    if fit_order == 3 and parameters[4] == -(parameters[0] + parameters[2]):
        with_uptake = True
    return fixed_parameters, with_uptake, all_parameters, washout_ratio


def extract_exponential_params_from_json_legacy(
    json_data: dict, cycle: str, region: str
) -> Tuple[Dict[str, float], bool, Dict[str, float]]:
    """Extract parameters of fit for a defined region and cycle from a patient JSON dictionary.

    Legacy function to support the previous version of patient JSON where not all parameters
    of fit were stored.

    Parameters
    ----------
    json_data : dict
        The patient's JSON dictionary.
    cycle : str
        The cycle ID
    region : str
        The region of interest

    Returns
    -------
    Tuple[Dict[str, float], bool, Dict[str, float]]
        A Tuple consisting of the exponential parameters from previous fit,
        a boolean representing whether or not initial uptake was accounted for in previous fit,
        and all the parameters of the fit.

    Raises
    ------
    AssertionError
        When the parameter configuration is incompatible with new format.
    """
    # Read Parameters:
    parameters = json_data[cycle][0]["VOIs"][region]["fit_params"]

    if len(parameters) not in [3, 5]:
        raise AssertionError(
            "Legacy parameter extraction from JSON not compatible with this JSON file."
        )

    if len(parameters) == 3:
        # Order = 2, with uptake:
        return (
            {"A2": parameters[1], "B2": parameters[2]},
            True,
            {
                "A1": parameters[0],
                "A2": parameters[1],
                "B1": -parameters[0],
                "B2": parameters[2],
            },
        )

    else:
        # Order = 3, with uptake:
        return (
            {"A2": parameters[1], "B2": parameters[3], "C2": parameters[4]},
            True,
            {
                "A1": parameters[0],
                "A2": parameters[1],
                "B1": parameters[2],
                "B2": parameters[3],
                "C1": -(parameters[0] + parameters[2]),
                "C2": parameters[4],
            },
        )


def initialize_biokinetics_from_prior_cycle(
    config: dict, prior_treatment_data: dict, cycle: str
) -> dict:
    """Initialize biokinetics parameters from a previous treatment cycle.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    prior_treatment_data : dict
        Prior treatment data dictionary.
    cycle : str
        Cycle identifier.

    Returns
    -------
    dict
        Updated configuration dictionary.
    """
    for roi, roi_info in config["VOIs"].items():

        if (
            "biokinectics_from_previous_cycle" in roi_info
            and roi_info["biokinectics_from_previous_cycle"]
        ):

            # Get previous cycle parameters:
            fixed_param, with_uptake, all_params, washout_ratio = (
                extract_exponential_params_from_json(
                    json_data=prior_treatment_data, cycle=cycle, region=roi
                )
            )

            config["VOIs"][roi] = {
                "fixed_parameters": fixed_param,
                "fit_order": len(all_params) // 2,
                "param_init": all_params,
                "with_uptake": with_uptake,
                "washout_ratio": washout_ratio,
            }

            print(
                f"{roi} will utilize the following parameters from the previous cycle {cycle}:"
            )
            print(fixed_param)
            print("")

    return config
