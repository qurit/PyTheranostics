"""Utility helpers for accessing package data via importlib.resources."""

from __future__ import annotations

from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Iterator


@contextmanager
def resource_path(package: str, relative_path: str) -> Iterator[Path]:
    """Yield a filesystem path to a bundled resource.

    The helper works for both files and directories and hides the boilerplate
    of using ``importlib.resources.as_file``. It ensures compatibility when the
    package is installed as a wheel/zip where resources need to be extracted to
    a temporary location before accessing them by path.
    """
    resource = resources.files(package).joinpath(*relative_path.split("/"))
    with resources.as_file(resource) as path_obj:
        yield Path(path_obj)
