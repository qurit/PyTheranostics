import math
from datetime import datetime
from typing import Dict, List, Tuple

import numpy
import pandas
from scipy.ndimage import median_filter

MEV_PER_G_TO_GY = 1.602176634e-10  # Gy per (MeV/g)


def hu_to_rho(hu: numpy.ndarray) -> numpy.ndarray:
    """Convert a CT array, in HU into a density map in g/cc
    Conversion based on Schneider et al. 2000 (using GATE's material db example)

    Args:
        hu (numpy.ndarray): _description_

    Returns:
        numpy.ndarray: _description_
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
def read_dpk_csv(path: str) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Read a DPK CSV (from Graves et al. 2019 https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.13789)
    and return (r_mm, K_Gy_per_decay) as 1D arrays.

    Parameters
    ----------
    path : str
        Path to CSV file.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        Radius in mm, Dose Kernel in Gy/decay.

    Raises
    ------
    ValueError
        Expected CSV format not found.
    ValueError
        Expected columns not found.
    """
    # Read entire file, skip first line (metadata)
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 3:
        raise ValueError(f"File {path} doesn't look like expected CSV (too few lines).")

    # Pandas read_csv from the second line (header is on line index 1)
    from io import StringIO

    buf = StringIO("".join(lines[1:]))
    df = pandas.read_csv(buf)

    # Required columns:
    key_r = "Outer Radius of Bin (cm)"
    key_d = "Dose per decay (MeV/g)"

    df_columns = list(df.columns)
    if key_r not in df_columns or key_d not in df_columns:
        raise ValueError(
            f"{path}: Required columns not found. Got columns: {df_columns}"
        )

    # Convert radii to mm
    r_cm = df[key_r].to_numpy(dtype=float)
    r_mm = r_cm * 10.0

    # Convert MeV/g to Gy
    d_mev_per_g = df[key_d].to_numpy(dtype=float)
    K_Gy = d_mev_per_g * MEV_PER_G_TO_GY

    # Ensure monotonic radii and non-negative K
    order = numpy.argsort(r_mm)
    r_mm = r_mm[order]
    K_Gy = numpy.maximum(K_Gy[order], 0.0)

    return r_mm, K_Gy


def merge_multiple_csvs(
    csvs: List[str], df_mm: float = 0.1, rmax_mm: float = 200.0
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Read multiple CSVs and sum their K(r). Return common fine-grid r_mm and summed K(r).
        - Interpolates each K onto a common 0.1 mm grid from r=0 to rmax_mm.
        - Zero beyond the last provided radius in each file.

    Parameters
    ----------
    csvs : List[str]
        List of pahts to CSV files.
    df_mm : float, optional
        grid step in mm, by default 0.1
    rmax_mm : float, optional
        maximum radius in mm, by default 200.0

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        Radius in mm, Summed Dose Kernel in Gy/decay.
    """
    # Read first to determine radius grid
    r_mm, K_Gy = read_dpk_csv(csvs[0])

    if len(csvs) == 1:
        return r_mm, K_Gy

    for p in csvs[1:]:
        r_mm_, K_Gy_ = read_dpk_csv(p)

        if not (r_mm_ == r_mm).all():
            raise ValueError("All CSVs must have the same radius grid.")
        K_Gy += K_Gy_

    return r_mm, K_Gy


# ----------------------------
# Kernel construction
# ----------------------------
def build_3d_field_from_radial(
    K_r_mm: numpy.ndarray,
    r_mm: numpy.ndarray,
    df_mm: float = 0.5,
    Rmax_mm: float = 200.0,
) -> numpy.ndarray:
    """Build an isotropic 3-D field on a fine grid (spacing df_mm) by sampling K(r).
    The field spans [-Rmax_mm, +Rmax_mm] along each axis.

    Parameters
    ----------
    K_r_mm : numpy.ndarray
        Dose Kernel in Gy/decay.
    r_mm : numpy.ndarray
        Radius in mm.
    df_mm : float
        Grid step in mm.
    Rmax_mm : float
        Maximum radius in mm.

    Returns
    -------
    numpy.ndarray
        3-D Dose Kernel field in Gy/decay.
    """
    # Determine N (odd) so that extent covers Rmax_mm
    N = int(numpy.floor(2 * Rmax_mm / df_mm)) + 1
    if N % 2 == 0:
        N += 1
    ax = (numpy.arange(N, dtype=float) - N // 2) * df_mm
    X, Y, Z = numpy.meshgrid(ax, ax, ax, indexing="ij")
    R = numpy.sqrt(X * X + Y * Y + Z * Z)

    # Interpolate K(r) for all R (vectorized)
    K_field = numpy.interp(
        R.ravel(), r_mm, K_r_mm, left=K_r_mm[0] if r_mm[0] == 0 else 0.0, right=0.0
    )
    K_field = K_field.reshape(R.shape)
    return K_field


def box_filter_1d(arr: numpy.ndarray, M: int, axis: int) -> numpy.ndarray:
    """Separable 1-D box filter (uniform average) along a given axis using cumulative sums.
     Handles zero-padding at the boundaries.

    Parameters
    ----------
    arr : numpy.ndarray
        3-D dose kernel field.
    M : int
        Window length
    axis : int
        Axis along which to apply the filter (0, 1, or 2)

    Returns
    -------
    numpy.ndarray
        Filtered array.
    """
    if M <= 1:
        return arr.copy()
    # Move axis to front
    arr_swapped = numpy.moveaxis(arr, axis, 0)

    # Zero-pad by floor(M/2) on both sides
    pad = M // 2
    pad_before = pad
    pad_after = M - 1 - pad
    padded = numpy.pad(
        arr_swapped,
        ((pad_before, pad_after),) + tuple((0, 0) for _ in range(arr_swapped.ndim - 1)),
        mode="constant",
        constant_values=0.0,
    )
    # Cumulative sum along leading axis
    csum = numpy.cumsum(padded, axis=0, dtype=float)
    # Windowed sum: s[i] = csum[i+M] - csum[i]
    s = csum[M:] - csum[:-M]
    out = s / float(M)
    # Move axis back
    out = numpy.moveaxis(out, 0, axis)
    return out


def double_box_average_3d(
    K_field: numpy.ndarray, L_mm: float, df_mm: float
) -> Tuple[numpy.ndarray, int]:
    """Apply two box averages of width L_mm (source & target) to the 3-D field.
    Implemented as separable 1-D filters along x,y,z.

    Parameters
    ----------
    K_field : numpy.ndarray
        3-D dose kernel field.
    L_mm : float
        Box Width in mm. This is the voxel size of the coarse lattice. (i.e., SPECT voxel Size)
    df_mm : float
        Grid step in mm.

    Returns
    -------
    Tuple[numpy.ndarray, int]
        Averaged 3-D field, M (box length in voxels).
    """
    M = max(1, int(round(L_mm / df_mm)))
    out = K_field
    # First box (e.g., target average)
    for ax in (0, 1, 2):
        out = box_filter_1d(out, M, axis=ax)
    # Second box (e.g., source average)
    for ax in (0, 1, 2):
        out = box_filter_1d(out, M, axis=ax)
    return out, M


def sample_on_coarse_lattice(
    K_avg: numpy.ndarray, L_mm: float, df_mm: float, Rmax_mm: float
) -> Tuple[numpy.ndarray, int]:
    """Sample the averaged fine field at coarse lattice points (iL, jL, kL).

    Parameters
    ----------
    K_avg : numpy.ndarray
        Averaged 3-D dose kernel field.
    L_mm : float
        Box width in mm. This is the voxel size of the coarse lattice. (i.e., SPECT voxel Size)
    df_mm : float
        Grid step in mm.
    Rmax_mm : float
        Maximum radius in mm.

    Returns
    -------
    Tuple[numpy.ndarray, int]
        h: 3-D kernel (odd-sized cube)
        Nc: radius in coarse voxels (so size = 2*Nc+1)
    """
    # Determine stride in voxels
    stride = int(round(L_mm / df_mm))
    N = K_avg.shape[0]
    center = N // 2

    # Determine Nc so that Nc*L_mm <= Rmax_mm (exclusive on the next)
    Nc = int(numpy.floor(Rmax_mm / L_mm + 1e-9))
    # Indices along one axis
    idx = center + numpy.arange(-Nc, Nc + 1) * stride
    # Guard within bounds
    idx = idx[(idx >= 0) & (idx < N)]
    # Build 3D sub-sampling
    h = K_avg[numpy.ix_(idx, idx, idx)].copy()
    # Ensure it's odd
    assert h.shape[0] == h.shape[1] == h.shape[2], "Kernel shape must be cubic"
    return h, (len(idx) - 1) // 2


# ----------------------------
# Sanity checks
# ----------------------------
def spherical_integral_K(r_mm: numpy.ndarray, K_r: numpy.ndarray) -> float:
    """
    Approximate ∫ K(r) dV over 0..∞ using discrete shells on the provided r grid (mm).
    Returns value in Gy * mm^3 (convert to Gy*m^3 by *1e-9).
    """
    # Trapezoidal in r with shell volume 4π r^2 dr
    r = r_mm
    K = K_r
    dr = numpy.diff(r)
    r_mid = 0.5 * (r[:-1] + r[1:])
    shell = 4.0 * math.pi * (r_mid**2) * ((K[:-1] + K[1:]) * 0.5) * dr
    return float(numpy.sum(shell))  # Gy * mm^3


def kernel_volume_sum(h: numpy.ndarray, L_mm: float) -> float:
    """
    Sum(h) * voxel_volume (Gy * mm^3), comparable to spherical_integral_K.
    """
    V_vox_mm3 = L_mm**3
    return float(numpy.sum(h) * V_vox_mm3)


def radius_for_fraction(
    r_mm: numpy.ndarray, K_r: numpy.ndarray, frac: float = 0.995
) -> float:
    """
    Return the first *tabulated* radius r[j] (mm) at which the cumulative deposited dose
    (volume integral) exceeds `frac` of the total (default 99.5%).

    This uses trapezoidal integration over W(r)=4π r^2 K(r) *per bin* and returns the
    outer edge of the first bin whose cumulative integral crosses the threshold.
    No sub-bin interpolation is performed.
    """
    if not (0.0 < frac < 1.0):
        raise ValueError("frac must be in (0,1).")

    r = numpy.asarray(r_mm, dtype=float)
    K = numpy.asarray(K_r, dtype=float)

    if r.ndim != 1 or K.ndim != 1 or r.size != K.size or r.size < 2:
        raise ValueError("r_mm and K_r must be 1D arrays of the same length >= 2.")
    if numpy.any(numpy.diff(r) <= 0):
        raise ValueError("r_mm must be strictly increasing.")

    W = 4.0 * numpy.pi * (r**2) * K
    dr = numpy.diff(r)
    bin_int = 0.5 * (W[:-1] + W[1:]) * dr
    total = float(numpy.sum(bin_int))
    if total <= 0.0:
        return float(r[0])

    target = frac * total
    cum = 0.0
    for i, Ti in enumerate(bin_int):
        cum += Ti
        if cum >= target:
            return float(r[i + 1])  # outer radius of the crossing bin

    return float(r[-1])
