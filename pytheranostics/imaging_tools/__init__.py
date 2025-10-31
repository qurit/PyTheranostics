"""Imaging tools and utilities for medical image processing.

Provides lazy access to the ``Tools`` submodule so patterns like
``pytheranostics.imaging_tools.Tools`` work without eager imports.
"""

import importlib

__all__ = ["Tools"]


def __getattr__(name: str):
    """Lazily import submodules on first attribute access."""
    if name == "Tools":
        return importlib.import_module(__name__ + ".Tools")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
