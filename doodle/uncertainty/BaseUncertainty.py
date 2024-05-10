import abc
from typing import Dict, List, Any
from doodle.dosimetry.BaseDosimetry import BaseDosimetry


class BaseUncertainty(metaclass=abc.ABCMeta):
    """Base Uncertainty Class. Holds the basic definitions to sample realizations of each component of the
    dosimetry workflow, based on pre-determined uncertainty.

    """
    
    def __init__(self, unc_config: Dict[str, Any], dose: BaseDosimetry) -> None:
        """_summary_

        Args:
            unc_config (Dict[str, Any]): _description_
        """
        self.config = unc_config
        self.dose_manager = dose
        self.estimates: Dict[str, List[float]] = {}
    
    
    @abc.abstractmethod
    def estimate_uncertainty(self, num_realizations: int = 100) -> None:
        """Abstract base method which that orchestrates the uncertainty estimation. Should run the 'simulation' of
        a dosimetry workflow, sampling each measured quantity according to its uncertainty.

        Args:
            num_realizations (int, optional): Number of simulations. Defaults to 100.

        """
        return None