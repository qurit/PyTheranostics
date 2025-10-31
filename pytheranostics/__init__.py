"""PyTheranostics - A Python library for nuclear medicine processing and dosimetry."""

# Lazy access to subpackages (to support attribute access like `pytheranostics.imaging_ds`)
# without importing them eagerly or triggering unused-import lint issues.
import importlib

# Import submodules for easier access
from pytheranostics.calibrations.gamma_camera import GammaCamera  # Calibration
from pytheranostics.dicomtools.dicomtools import DicomModify  # DICOM handling
from pytheranostics.fits.fits import biexp_fun, monoexp_fun, triexp_fun  # Analysis
from pytheranostics.plots.plots import ewin_montage, plot_tac_residuals  # Visualization
from pytheranostics.qc.dosecal_qc import DosecalQC  # Core
from pytheranostics.qc.planar_qc import PlanarQC  # Core
from pytheranostics.qc.spect_qc import SPECTQC  # Core
from pytheranostics.segmentation.tools import rtst_to_mask  # Image processing
from pytheranostics.shared.corrections import tew_scatt
from pytheranostics.shared.evaluation_metrics import perc_diff
from pytheranostics.shared.radioactive_decay import decay_act, get_activity_at_injection

_SUBPACKAGES = {
    "imaging_ds": "pytheranostics.imaging_ds",
    "imaging_tools": "pytheranostics.imaging_tools",
    "dosimetry": "pytheranostics.dosimetry",
    "misc_tools": "pytheranostics.misc_tools",
    "registration": "pytheranostics.registration",
    "segmentation": "pytheranostics.segmentation",
    "shared": "pytheranostics.shared",
    "plots": "pytheranostics.plots",
    "qc": "pytheranostics.qc",
    "dicomtools": "pytheranostics.dicomtools",
    "fits": "pytheranostics.fits",
    "calibrations": "pytheranostics.calibrations",
}


def __getattr__(name):
    """Dynamically expose subpackages and legacy aliases on first access.

    This allows patterns like `import pytheranostics as tx; tx.imaging_ds` to work
    without importing all subpackages at import time. Also provides backwards-
    compatible aliases used in older notebooks (e.g., `tx.MiscTools`).
    """
    # Direct subpackages
    if name in _SUBPACKAGES:
        return importlib.import_module(_SUBPACKAGES[name])

    # Legacy aliases (backwards compatibility)
    if name == "MiscTools":  # alias to misc_tools
        return importlib.import_module("pytheranostics.misc_tools")
    if name == "ImagingTools":  # alias to imaging_tools
        return importlib.import_module("pytheranostics.imaging_tools")

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Define what should be imported with "from pytheranostics import *"
__all__ = [
    "PlanarQC",
    "DosecalQC",
    "SPECTQC",
    "decay_act",
    "get_activity_at_injection",
    "perc_diff",
    "tew_scatt",
    "GammaCamera",
    "ewin_montage",
    "plot_tac_residuals",
    "rtst_to_mask",
    "monoexp_fun",
    "biexp_fun",
    "triexp_fun",
    "DicomModify",
    # Expose subpackage names at the package level for discoverability
    "imaging_ds",
    "imaging_tools",
    "dosimetry",
    "misc_tools",
    "registration",
    "segmentation",
    "shared",
    "plots",
    "qc",
    "dicomtools",
    "fits",
    "calibrations",
    # Legacy aliases
    "MiscTools",
    "ImagingTools",
]
