"""
PyTheranostics - A Python library for nuclear medicine processing and dosimetry
"""

__version__ = "0.1.0"
__author__ = "Carlos Uribe, PhD, MCCPM"
__email__ = "curibe@bccrc.ca"
__license__ = "MIT"

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
