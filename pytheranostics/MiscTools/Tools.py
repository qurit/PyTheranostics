from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy
import pandas
from numpy.typing import NDArray
from scipy.ndimage import median_filter

MEV_PER_G_TO_GY = 1.602176634e-10  # Gy per (MeV/g)


def hu_to_rho(hu: NDArray) -> NDArray:
    """Convert a CT array, in HU into a density map in g/cc
    Conversion based on Schneider et al. 2000 (using GATE's material db example)

    Args:
        hu (NDArray): _description_

    Returns:
        NDArray: _description_
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
        The cycle ID
    region : str
        The region of interest

    Returns
    -------
    Tuple[Dict[str, float], bool, Dict[str, float]]
        A Tuple consisting of the exponential parameters from previous fit,
        a boolean representing whether or not initial uptake was accounted for in previous fit,
        and all the parameters of the fit.
    """
    with_uptake = False

    # Determine order:
    parameters = json_data[cycle][0]["rois"][region]["fit_params"]

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

    if fit_order == 2 and parameters[0] == -parameters[2]:
        with_uptake = True

    if fit_order == 3 and parameters[4] == -(parameters[0] + parameters[2]):
        with_uptake = True

    return fixed_parameters, with_uptake, all_parameters


def extract_exponential_params_from_json_legacy(
    json_data: dict, cycle: str, region: str
) -> Tuple[Dict[str, float], bool, Dict[str, float]]:
    """Legacy function to extract parameters of fit for a defined region and cycle from a JSON dictionary of a patient.
    It supports the previous version of patient JSON where not all parameters of fit where stored.

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
    parameters = json_data[cycle][0]["rois"][region]["fit_params"]

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

    for roi, roi_info in config["rois"].items():

        if (
            "biokinectics_from_previous_cycle" in roi_info
            and roi_info["biokinectics_from_previous_cycle"]
        ):

            # Get previous cycle parameters:
            fixed_param, with_uptake, all_params = extract_exponential_params_from_json(
                json_data=prior_treatment_data, cycle=cycle, region=roi
            )

            config["rois"][roi] = {
                "fixed_parameters": fixed_param,
                "fit_order": len(all_params) // 2,
                "param_init": all_params,
                "with_uptake": with_uptake,
            }

            print(
                f"{roi} will utilize the following parameters from the previous cycle {cycle}:"
            )
            print(fixed_param)
            print("")

    return config


# Functions to generate voxel-s kernels
# ----------------------------
# CSV I/O
# ----------------------------
def load_kernel_from_csv(path: Path) -> NDArray:
    """Read Voxel Kernels from
    Graves, S., Tiwari, A., Merrick, M., Hyer, D., Flynn, R.,
    Kruzer, A., Nelson, A., Dewaraja, Y., Mirando, D.,
    & Sunderland, J. (2023). Accurate resampling of radial dose point
    kernels to a Cartesian matrix for voxelwise dose calculation (1.1)
    [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7596345

    Parameters
    ----------
    path : Path
        Path to .csv file containing voxel-kernel values for positive Octant.

    Returns
    -------
    NDArray
        array containing kernel values.

    Raises
    ------
    ValueError
        _description_
    """
    df = pandas.read_csv(
        path,
        header=None,
        skip_blank_lines=False,  # <- important
    )

    # rows that are completely blank will be all NaN
    blank_mask = df.isna().all(axis=1)

    # make a group id that increments every time we see a blank row
    # e.g. rows -> 0..172 (block 0), blank, 174..346 (block 1), ...
    block_ids = blank_mask.cumsum()

    # drop the blank rows themselves
    df_data = df[~blank_mask].reset_index(drop=True)
    block_ids = block_ids[~blank_mask].to_numpy()

    # infer N
    # number of rows in the first block
    first_block_rows = (block_ids == block_ids[0]).sum()
    N = first_block_rows

    # now we have (num_blocks * N) rows, each with N columns
    # we can groupby block_id and build the 3D array
    blocks = []
    for b in numpy.unique(block_ids):
        block_df = df_data[block_ids == b]
        arr = block_df.to_numpy(dtype=float)
        if arr.shape != (N, N):
            raise ValueError(f"Block {b} has shape {arr.shape}, expected {(N, N)}")
        blocks.append(arr)

    return expand_octant_to_full(numpy.stack(blocks, axis=0))


def expand_octant_to_full(octant: NDArray) -> NDArray:
    """Given a 3D array of shape (H, H, H) that represents the center voxel
    at [0,0,0] and the +x, +y, +z directions (i.e. the positive octant),
    reconstruct the full symmetric kernel of shape (2H-1, 2H-1, 2H-1).

    Parameters
    ----------
    octant : numpy.ndarray
        Positive octant of the kernel, shape (H, H, H)

    Returns
    -------
    numpy.ndarray
        The full symmetric kernel, shape (2H-1, 2H-1, 2H-1)

    Raises
    ------
    ValueError
        If the input octant is not 3D or not cubic.
    """
    if octant.ndim != 3:
        raise ValueError("octant must be 3D")

    H = octant.shape[0]
    if not (octant.shape[1] == H and octant.shape[2] == H):
        raise ValueError("octant must be cubic (H×H×H)")

    # mirror in x (axis=0): [-x | 0..+x]
    # octant[1:][::-1, :, :] gives slices 1..H-1 reversed → x=-1, -2, ...
    full_x = numpy.concatenate([octant[1:][::-1, :, :], octant], axis=0)  # (2H-1, H, H)

    # mirror in y (axis=1): [-y | 0..+y]
    full_xy = numpy.concatenate(
        [full_x[:, 1:][:, ::-1, :], full_x], axis=1
    )  # (2H-1, 2H-1, H)

    # mirror in z (axis=2): [-z | 0..+z]
    full_xyz = numpy.concatenate(
        [full_xy[:, :, 1:][:, :, ::-1], full_xy], axis=2
    )  # (2H-1, 2H-1, 2H-1)

    return full_xyz
