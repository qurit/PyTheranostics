from dataclasses import dataclass
from typing import Optional


@dataclass
class MetaDataType:
    """Metadata information for medical imaging datasets."""

    PatientID: str
    AcquisitionDate: str
    AcquisitionTime: str
    HoursAfterInjection: Optional[float]
    Radionuclide: Optional[str]
    Injected_Activity_MBq: Optional[float]
