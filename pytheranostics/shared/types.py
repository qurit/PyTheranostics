"""Shared type definitions for pytheranostics.

This module contains lightweight, dependency-free data structures that are
shared across subpackages to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ImagingMetadata:
    """Metadata information for medical imaging datasets.

    This dataclass is intentionally defined in a shared, neutral module so it
    can be imported by both imaging_tools and imaging_ds without creating
    circular dependencies.
    """

    PatientID: str
    AcquisitionDate: str
    AcquisitionTime: str
    HoursAfterInjection: Optional[float]
    Radionuclide: Optional[str]
    Injected_Activity_MBq: Optional[float]
