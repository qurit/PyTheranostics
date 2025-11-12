"""Plotting utilities for PyTheranostics workflows."""

from pathlib import Path
from typing import Optional

import lmfit
import matplotlib.pyplot as plt
import numpy


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
    x_fit = numpy.linspace(0, x_data[-1] * 3, 500)
    y_fit = result.eval(x=x_fit)

    # First subplot: Linear scale plot
    ax1 = axs[0]
    # Plot data points
    ax1.errorbar(x_data, y_data, yerr=weights, fmt="o", markersize=5)
    # Plot fitted model
    ax1.plot(x_fit, y_fit, color="red")
    ax1.set_xlim(left=0)  # Start x-axis from zero
    ax1.set_xlim(right=x_data[-1] * 2)  # Start y-axis from zero
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
    ax2.set_xlim(right=x_data[-1] * 2)  # Start y-axis from zero
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
