"""
PyTheranostics - A Python library for nuclear medicine processing and dosimetry
"""

try:
    from importlib.metadata import version
    __version__ = version("pytheranostics")
except ImportError:
    __version__ = "unknown"

__author__ = 'Carlos Uribe, PhD, MCCPM'
__email__ = 'curibe@bccrc.ca'
__license__ = 'MIT'

# Define what should be imported with "from pytheranostics import *"
__all__ = [
    'PlanarQC',
    'DosecalQC',
    'SPECTQC',
    'decay_act',
    'get_activity_at_injection',
    'perc_diff',
    'tew_scatt',
    'GammaCamera',
    'ewin_montage',
    'plot_tac_residuals',
    'rtst_to_mask',
    'monoexp_fun',
    'biexp_fun',
    'triexp_fun',
    'DicomModify',
    'OrganSDosimetry',
    'VoxelSDosimetry',
    'load_and_resample_RT',
    'create_logitudinal_from_dicom',
    'load_phantom_mass'
]

# Lazy imports
def __getattr__(name):
    if name == 'PlanarQC':
        from pytheranostics.qc.planar_qc import PlanarQC
        return PlanarQC
    elif name == 'DosecalQC':
        from pytheranostics.qc.dosecal_qc import DosecalQC
        return DosecalQC
    elif name == 'SPECTQC':
        from pytheranostics.qc.spect_qc import SPECTQC
        return SPECTQC
    elif name in ['decay_act', 'get_activity_at_injection']:
        from pytheranostics.shared.radioactive_decay import decay_act, get_activity_at_injection
        return globals()[name]
    elif name == 'perc_diff':
        from pytheranostics.shared.evaluation_metrics import perc_diff
        return perc_diff
    elif name == 'tew_scatt':
        from pytheranostics.shared.corrections import tew_scatt
        return tew_scatt
    elif name == 'GammaCamera':
        from pytheranostics.calibrations.gamma_camera import GammaCamera
        return GammaCamera
    elif name in ['ewin_montage', 'plot_tac_residuals']:
        from pytheranostics.plots.plots import ewin_montage, plot_tac_residuals
        return globals()[name]
    elif name == 'rtst_to_mask':
        from pytheranostics.segmentation.tools import rtst_to_mask
        return rtst_to_mask
    elif name in ['monoexp_fun', 'biexp_fun', 'triexp_fun']:
        from pytheranostics.fits.fits import monoexp_fun, biexp_fun, triexp_fun
        return globals()[name]
    elif name == 'DicomModify':
        from pytheranostics.dicomtools.dicomtools import DicomModify
        return DicomModify
    elif name == 'OrganSDosimetry':
        from pytheranostics.dosimetry.OrganSDosimetry import OrganSDosimetry
        return OrganSDosimetry
    elif name == 'VoxelSDosimetry':
        from pytheranostics.dosimetry.VoxelSDosimetry import VoxelSDosimetry
        return VoxelSDosimetry
    elif name == 'load_phantom_mass':
        from pytheranostics.dosimetry.olinda import load_phantom_mass
        return load_phantom_mass
    elif name == 'load_and_resample_RT':
        from pytheranostics.ImagingTools.Tools import load_and_resample_RT
        return load_and_resample_RT
    elif name == 'create_logitudinal_from_dicom':
        from pytheranostics.ImagingDS.LongStudy import create_logitudinal_from_dicom
        return create_logitudinal_from_dicom
    raise AttributeError(f"module 'pytheranostics' has no attribute '{name}'")