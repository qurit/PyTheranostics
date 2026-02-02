"""Plotting utilities for PyTheranostics workflows."""

import json
from pathlib import Path
from typing import Optional

import lmfit
import matplotlib.pyplot as plt
import numpy

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files


def _find_config_file(filename="voi_mappings_config.json", max_depth=2):
    """Search upward from current directory for config file.

    Searches current directory and one parent level only to avoid
    finding unrelated configs from other projects.

    Parameters
    ----------
    filename : str
        Name of config file to find
    max_depth : int
        Maximum number of parent directories to search (default: 2)

    Returns
    -------
    Path or None
        Path to config file if found, None otherwise
    """
    current = Path.cwd()
    for _ in range(max_depth):
        config_path = current / filename
        if config_path.exists():
            return config_path
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent
    return None


def ewin_montage(img: numpy.ndarray, ewin: dict) -> None:
    """Create a montage of energy window images.

    This function creates a 2x6 subplot montage showing energy window images
    from two detectors. The top row shows images from Detector 1, and the
    bottom row shows corresponding images from Detector 2.

    Parameters
    ----------
    img : numpy.ndarray
        Image data array containing energy window images.
        Shape should be (2*N, height, width) where N is the number of energy windows.
    ewin : dict
        Dictionary containing energy window information.
        Keys should be window identifiers, and values should be dictionaries
        containing at least a 'center' key with the energy value in keV.

    Notes
    -----
    - The function creates a figure with size (22, 6).
    - Each energy window is displayed in a separate subplot.
    - Colorbars are added to each subplot.
    - The layout is automatically adjusted using tight_layout().
    """
    plt.figure(figsize=(22, 6))
    for ind, i in enumerate(range(0, int(img.shape[0]), 2)):
        keys = list(ewin.keys())

        # Top row Detector 1
        plt.subplot(2, 6, ind + 1)
        plt.imshow(img[i, :, :])
        plt.title(f'Detector1 {ewin[keys[ind]]["center"]} keV')
        plt.colorbar()

        # Bottom row Detector 2
        plt.subplot(2, 6, ind + 7)
        plt.imshow(img[i + 1, :, :])
        plt.title(f'Detector2 {ewin[keys[ind]]["center"]} keV')
        plt.colorbar()

    plt.tight_layout()


def plot_tac_residuals(
    result: lmfit.model.ModelResult,
    region: str,
    cycle: int,
    x_label: str = "Time [hr]",
    y_label: str = "Activity [MBq]",
    output_dir: Optional[Path] = None,
) -> None:
    """Plot time-activity curve and residuals."""
    # Create a figure with 3 subplots
    # Create a figure with 3 subplots
    _, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    # Extract fitted parameters and format them
    params = result.params.valuesdict()
    formatted_params = {key: f"{value:.3f}" for key, value in params.items()}

    # Construct the mathematical expression for the title
    num_exponentials = len(params) // 2  # Each exponential has two parameters
    terms = []
    for _, term in enumerate(["A", "B", "C"][:num_exponentials]):
        A1 = formatted_params.get(f"{term}1", "0")
        A2 = formatted_params.get(f"{term}2", "0")
        terms.append(f"${A1}e^{{-{A2} t}}$")
    function_expression = " + ".join(terms)
    title_text = "$A(t) = $" + f"{function_expression}"

    # Retrieve x_data and y_data from the fit result
    x_data = result.userkws["x"]
    y_data = result.data
    if result.weights is not None:
        weights = 1 / result.weights
    else:
        weights = None
    # Generate x-values for plotting the fitted model starting from x=0
    x_fit = numpy.linspace(0, x_data.max() * 3, 500)
    y_fit = result.eval(x=x_fit)

    # First subplot: Linear scale plot
    ax1 = axs[0]
    # Plot data points
    ax1.errorbar(x_data, y_data, yerr=weights, fmt="o", markersize=5)
    # Plot fitted model
    ax1.plot(x_fit, y_fit, color="red")
    ax1.set_xlim(left=0)  # Start x-axis from zero
    ax1.set_xlim(right=x_data.max() * 2)  # Start y-axis from zero
    ax1.set_ylim(bottom=0)  # Start y-axis from zero
    ax1.set_title(region)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    # Add R-squared and AIC as text
    try:
        ax1.text(0.7, 0.9, f"$R^2={result.rsquared:.3f}$", transform=ax1.transAxes)
        ax1.text(0.7, 0.85, f"AIC={result.aic:.3f}", transform=ax1.transAxes)
    except AttributeError:
        pass
    # Remove legend if present
    legend = ax1.get_legend()
    if legend:
        legend.remove()

    # Second subplot: Semilog plot
    ax2 = axs[1]
    # Plot data points
    ax2.plot(x_data, y_data, "o", markersize=5)
    # Plot fitted model
    ax2.plot(x_fit, y_fit, color="red")
    ax2.set_xlim(left=0)  # Start x-axis from zero
    ax2.set_xlim(right=x_data.max() * 2)  # Start y-axis from zero
    ax2.set_yscale("log")
    ax2.set_title(title_text)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    # Remove legend if present
    legend = ax2.get_legend()
    if legend:
        legend.remove()

    # Third subplot: Residuals plot
    ax3 = axs[2]
    result.plot_residuals(ax=ax3, data_kws={"markersize": 5})
    ax3.set_title("Residuals")
    ax3.set_xlabel(x_label)
    ax3.set_ylabel("Residuals")

    if output_dir is not None:
        plt.savefig(
            output_dir / f"{region}_fit_Cycle_0{cycle}.png",
            format="png",
            bbox_inches="tight",
            dpi=300,
        )

    plt.show()

    return None


def plot_MIP_with_mask_outlines(
    SPECT,
    masks=None,
    vmax=300000,
    label=None,
    save_path=None,
    dpi=300,
    ax=None,
    figsize=None,
    spacing=None,
    mask_colors=None,
    config_path=None,
    colorbar=False,
    units=None,
):
    """Plot Maximum Intensity Projection (MIP) of SPECT data with masks outlines.

    Parameters
    ----------
    SPECT : numpy.ndarray
        3D SPECT data array.
    masks : dict, optional
        Dictionary of masks with organ names as keys and 3D arrays as values.
        By default None.
    vmax : int, optional
        Maximum value for display intensity. By default 300000.
    label : bool, optional
        Whether to add text labels at mask centers. By default None.
    save_path : str or Path, optional
        Path to save the figure. If provided, the parent directory will be created
        if it doesn't exist. By default None (no saving).
    dpi : int, optional
        Resolution for saved figure in dots per inch. By default 300.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to plot on. If None, creates a new figure and axes.
        By default None.
    figsize : tuple, optional
        Figure size (width, height) in inches when creating a new figure.
        If None, automatically calculates based on physical dimensions. By default None.
    spacing : tuple, optional
        Pixel spacing (x, y, z) in mm from DICOM. If provided, used to create
        physically accurate aspect ratio. By default None.
    mask_colors : dict, optional
        Dictionary mapping organ names (or keywords) to colors. If a mask name matches
        a key (case-insensitive substring match), that color is used. If None, uses
        default color mapping. Unmapped organs use matplotlib's color cycle.
        Example: {"kidney": "lime", "liver": "yellow", "lesion": "red"}
        By default None.
    config_path : str or Path, optional
        Path to voi_mappings_config.json file. If None, searches upward from current
        directory (like git) to find the config in your project root. Falls back to
        package template if not found.
        Example: Path("./my_project/voi_mappings_config.json")
        By default None.
    colorbar : bool, optional
        Whether to add a colorbar showing the intensity scale. By default False.
    units : str, optional
        Units from DICOM header (e.g., 'BQML', 'CNTS'). Can be extracted from
        DICOM tag (0054,1001). If provided, automatically determines scaling and
        colorbar label. Common values: 'BQML' (Bq/ml), 'CNTS' (counts).
        By default None (assumes Bq and scales to MBq).

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.
    """
    plt.sca(ax) if ax is not None else None
    spect_mip = SPECT.max(axis=0)

    # Calculate aspect ratio for proper physical scaling
    if spacing is not None:
        # spacing is (x, y, z) in mm
        # For proper aspect ratio: aspect = dy/dx
        data_aspect = spacing[1] / spacing[0]  # y-spacing / x-spacing
    else:
        data_aspect = 1.0

    # Automatically determine bounds based on data content
    # Use a threshold to find where there's actual signal
    threshold = vmax * 0.01  # 1% of max display value
    signal_mask = spect_mip.T > threshold

    if signal_mask.any():
        # Find bounding box of signal
        rows, cols = numpy.where(signal_mask)
        ylim_min, ylim_max = rows.min(), rows.max()
        xlim_min, xlim_max = cols.min(), cols.max()

        # Add small margin (5% on each side)
        margin_x = int((xlim_max - xlim_min) * 0.05)
        margin_y = int((ylim_max - ylim_min) * 0.05)

        xlim_min = max(0, xlim_min - margin_x)
        xlim_max = min(spect_mip.shape[1] - 1, xlim_max + margin_x)
        ylim_min = max(0, ylim_min - margin_y)
        ylim_max = min(spect_mip.shape[0] - 1, ylim_max + margin_y)
    else:
        # Fallback to full image if no signal detected
        xlim_min, xlim_max = 0, spect_mip.shape[1] - 1
        ylim_min, ylim_max = 0, spect_mip.shape[0] - 1

    # Create figure and axes if not provided
    if ax is None:
        if figsize is None:
            if spacing is not None:
                # Physical dimensions of ROI in mm
                roi_width_mm = (xlim_max - xlim_min) * spacing[0]
                roi_height_mm = (ylim_max - ylim_min) * spacing[1]

                # Create compact figure matching ROI aspect ratio
                base_width = 3  # inches - smaller base
                figsize = (base_width, base_width * roi_height_mm / roi_width_mm)
            else:
                # Fallback to pixel-based calculation
                xlim_range = xlim_max - xlim_min
                ylim_range = ylim_max - ylim_min
                aspect_ratio = ylim_range / xlim_range
                base_width = 3
                figsize = (base_width, base_width * aspect_ratio)

        fig, ax = plt.subplots(figsize=figsize)
        plt.sca(ax)

    # Determine scaling and label based on units
    if units is not None:
        units_upper = units.upper()
        if units_upper in ["BQML", "BQ/ML"]:
            scale_factor = 1e6
            colorbar_label = "MBq/ml"
        elif units_upper in ["BQ", "BECQUEREL"]:
            scale_factor = 1e6
            colorbar_label = "MBq"
        elif units_upper in ["CNTS", "COUNTS"]:
            scale_factor = 1.0
            colorbar_label = "Counts"
        else:
            # Unknown units, no scaling
            scale_factor = 1.0
            colorbar_label = units
    else:
        # Default: assume Bq and convert to MBq
        scale_factor = 1e6
        colorbar_label = "MBq"

    # Apply scaling
    spect_mip_scaled = spect_mip / scale_factor
    vmax_scaled = vmax / scale_factor

    im = plt.imshow(
        spect_mip_scaled.T,
        cmap="Greys",
        interpolation="Gaussian",
        vmax=vmax_scaled,
        vmin=0,
        aspect=data_aspect,
    )

    # Add colorbar if requested
    if colorbar:
        plt.colorbar(im, ax=ax, label=colorbar_label)

    # Hardcoded fallback defaults (only used if config file is missing)
    fallback_defaults = {
        "kidney": "lime",
        "parotid": "red",
        "submandibular": "red",
        "lesion": "magenta",
        "liver": "yellow",
        "spleen": "cyan",
        "tumor": "magenta",
    }

    # Determine color mapping: parameter > config file > fallback defaults
    if mask_colors is not None:
        # Explicit override via parameter
        color_map = mask_colors
    else:
        # Try to load from configuration file
        try:
            if config_path is not None:
                # User-provided config path
                config_file = Path(config_path)
            else:
                # Search for config in project directory (upward search)
                config_file = _find_config_file("voi_mappings_config.json")

                if config_file is None:
                    # Not found in project, use package template
                    config_file = files(
                        "pytheranostics.data.configuration_templates"
                    ).joinpath("voi_mappings_config.json")

            with config_file.open("r") as f:
                config = json.load(f)
                if "plot_colors" in config:
                    # Use plot_colors from config (even if empty - will trigger color cycle)
                    color_map = {
                        k: v
                        for k, v in config["plot_colors"].items()
                        if not k.startswith("_")
                    }
                else:
                    # plot_colors key missing, use fallback
                    color_map = fallback_defaults
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            # Config file missing or invalid, use fallback defaults
            color_map = fallback_defaults

    # Color cycle for unmapped organs
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_index = 0

    if masks is not None:
        for organ, mask in masks.items():
            organ_lower = organ.lower()

            # Skip organs with 'peak' in the name (background/noise)
            if "peak" in organ_lower:
                continue

            # Find matching color from color_map (keyword match)
            color = None
            for keyword, keyword_color in color_map.items():
                if keyword.lower() in organ_lower:
                    color = keyword_color
                    break

            # If no match found, use color cycle
            if color is None:
                color = color_cycle[color_index % len(color_cycle)]
                color_index += 1

            mip_mask = mask.max(axis=0)
            if mip_mask.shape != spect_mip.shape:
                mip_mask = mip_mask.T

            plt.contour(
                numpy.transpose(mip_mask, (1, 0)),
                levels=[0.5],
                colors=[color],
                linewidths=1.5,
                alpha=0.5,
            )

            # --- Add label at mask center ---
            if label is True:
                ys, xs = numpy.where(mip_mask > 0)

                if len(xs) > 0:
                    # Corrected for transpose in contour
                    x_center = ys.mean() - 0.2 * ys.mean()  # was xs
                    y_center = xs.mean()  # was ys

                    plt.text(
                        x_center,
                        y_center,
                        organ,
                        color=color,
                        fontsize=8,
                        ha="center",
                        va="center",
                        alpha=0.7,
                    )

    plt.xlim(xlim_min, xlim_max)
    plt.ylim(ylim_min, ylim_max)
    plt.axis("off")
    plt.xticks([])
    plt.yticks([])

    # Save figure if path is provided
    if save_path is not None:
        save_path = Path(save_path)
        # Create parent directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)

    return ax
