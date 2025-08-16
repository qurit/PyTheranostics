"""
PyTheranostics - A Python library for nuclear medicine processing and dosimetry
"""

__version__ = "0.1.0"
__author__ = "Carlos Uribe, PhD, MCCPM"
__email__ = "curibe@bccrc.ca"
__license__ = "MIT"

# Core functionality imports
from pytheranostics.qc.planar_qc import PlanarQC
from pytheranostics.qc.dosecal_qc import DosecalQC
from pytheranostics.qc.spect_qc import SPECTQC

# Shared utilities
from pytheranostics.shared.radioactive_decay import decay_act, get_activity_at_injection
from pytheranostics.shared.evaluation_metrics import perc_diff
from pytheranostics.shared.corrections import tew_scatt

# Calibration tools
from pytheranostics.calibrations.gamma_camera import GammaCamera

# Visualization
from pytheranostics.plots.plots import ewin_montage, plot_tac_residuals

# Image processing
from pytheranostics.segmentation.tools import rtst_to_mask

# Analysis tools
from pytheranostics.fits.fits import monoexp_fun, biexp_fun, triexp_fun

# DICOM handling
from pytheranostics.dicomtools.dicomtools import DicomModify

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
]
