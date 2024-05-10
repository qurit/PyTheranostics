from typing import Any, Dict
from doodle.dosimetry.BaseDosimetry import BaseDosimetry
from doodle.uncertainty.BaseUncertainty import BaseUncertainty


class OrganSUncertainty(BaseUncertainty):
    """

    Args:
        BaseUncertainty (_type_): _description_
    """
    
    def __init__(self, unc_config: Dict[str, Any], dose: BaseDosimetry) -> None:
        super().__init__(unc_config, dose)
        
    def recon(self) -> None:
        """ Model uncertainty associated with image reconstruction algorithm. The following methods are available:
        - User defined region-specific uncertainty.
        - pyTomography region-based uncertainty estimator (requires projection data).
        - Look-up table. Currently contains uncertainty estimates for kidneys, liver, salivary glands, and lesions from simulations of an 
        XCAT phantom, and pre-computed using pyTomography region-based uncertainty estimator.
        """
        