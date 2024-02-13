from typing import List
from pathlib import Path
from doodle.ImagingDS.LongStudy import LongitudinalStudy
import json


class BaseDosimetry:
    """Base Dosimetry Class. Takes Nuclear Medicine Data and CT Data to perform patient-specific dosimetry."""
    def __init__(self, patient_id: str, cycle: int,
                 database_dir: str,
                 nm_data: LongitudinalStudy, ct_data: LongitudinalStudy,
                 ) -> None:

        # Store data
        self.patient_id = patient_id 
        self.cycle = cycle
        self.db_dir = Path(database_dir)
        self.check_patient_in_db() # TODO: Traceability/database?

        self.nm_data = nm_data
        self.ct_data = ct_data

        # Veryfy radionuclide information is present in nm_data.
        self.check_nm_data()


    def check_nm_data(self) -> None:
        """Verify that radionuclide info is present in NM data, and that radionuclide data (e.g., half-life) 
        is available in internal database."""

        # Load Radionuclide data
        with open("../data/isotopes.json", "r") as rad_data:
            radionuclide_data = json.load(rad_data)

        if "radionuclide" not in self.nm_data.meta:
            raise ValueError("Nuclear Medicine Data missing radionuclide")
        
        if self.nm_data.meta["radionuclide"] not in radionuclide_data:
            raise ValueError(f"Data for {self.nm_data.meta['radionuclide']} is not available.")
        
    def check_patient_in_db() -> None:
        """Check if prior dosimetry exists for this patient"""
        raise RuntimeWarning("Database search function not implemented. Dosimetry for this patient might "
                             "already exists...")
        return None
    
    def compute_tiac(self, method: str)