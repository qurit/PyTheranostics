from typing import Any, Dict
from doodle.dosimetry.BaseDosimetry import BaseDosimetry
from doodle.uncertainty.BaseUncertainty import BaseUncertainty

class MonteCarloUncertainty(BaseUncertainty):
    """Monte-Carlo Uncertainty Estimator class. Perform simulations of the dose calculation workflow by sampling each relevant parameter at each step according to
    the parameter specific variability.    
    """
    def __init__(self, unc_config: Dict[str, Any], dose: BaseDosimetry) -> None:
        """
        Args:
            unc_config (Dict[str, Any]): Uncertainty configuration parameters. See URL. # TODO
            dose (BaseDosimetry): Initialized Dosimetry Calculator class.
        """
        super().__init__(unc_config, dose)
        
    def check_config(self) -> None:
        """Check Configuration dictionary for mandatory parameters, and uses default values if not present.
        """
        
        mandatory_defaults = {"num_realizations": 20}
        
        for mandatory, default in mandatory_defaults.items():
            if mandatory not in self.config:
                self.config[mandatory] = default
                        
        return None
    
    def sample_masks(self) -> None:
        """For each region of interest, re-generate a new mask slightly different from the base mask to model uncertainty in image segmetnation.
        The re-generation is obtained by applying an elastic transformation on the mask.
        """
        
        
        
        