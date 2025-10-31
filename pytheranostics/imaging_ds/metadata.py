"""Metadata structures for imaging datasets.

This module re-exports shared metadata types to keep imaging_ds free of
cross-package dependencies and avoid circular imports.
"""

from pytheranostics.shared.types import ImagingMetadata

__all__ = ["ImagingMetadata"]
