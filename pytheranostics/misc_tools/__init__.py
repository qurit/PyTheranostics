"""Miscellaneous tools and utilities.

Provides lazy access to the ``Tools`` submodule so patterns like
``pytheranostics.misc_tools.Tools`` work without eager imports.
"""

import importlib

__all__ = ["Tools"]


def __getattr__(name: str):
    """Lazily import submodules on first attribute access."""
    if name == "Tools":
        return importlib.import_module(__name__ + ".Tools")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
