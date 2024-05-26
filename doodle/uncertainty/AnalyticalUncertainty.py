from typing import Any, Dict
from doodle.dosimetry.BaseDosimetry import BaseDosimetry
from doodle.uncertainty.BaseUncertainty import BaseUncertainty


class AnalyticalUncertainty(BaseUncertainty):
    """Implements and follows EANM Guidelines for uncertainty estimation:
    
        Gear, J.I., Cox, M.G., Gustafsson, J. et al. EANM practical guidance on uncertainty analysis for molecular radiotherapy absorbed dose calculations. 
        Eur J Nucl Med Mol Imaging 45, 2456–2474 (2018). https://doi.org/10.1007/s00259-018-4136-7

    """
    
    def __init__(self, unc_config: Dict[str, Any], dose: BaseDosimetry) -> None:
        super().__init__(unc_config, dose)
        
        
    def recon(self) -> None:
        """ Estimates uncertainty associated with image reconstruction algorithm. The following methods are available:
        
            - User defined region-specific uncertainty.
            - pyTomography region-based uncertainty estimator (requires projection data).
            - Look-up table. Currently contains uncertainty estimates for kidneys, liver, salivary glands, and lesions from simulations of an 
              XCAT phantom, and pre-computed using pyTomography region-based uncertainty estimator.
              
              
        """
        return None
    
    def volume_uncertainty(self) -> None:
        """Estimates uncertainty in volume (or mass) due to volume delineation. 
        """