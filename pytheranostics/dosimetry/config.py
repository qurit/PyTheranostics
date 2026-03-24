"""Dosimetry fit configuration loader.

Provides a single API to build ROI fit parameter configuration from a project or
package template, merging organ defaults, organ overrides, and auto-discovered
lesions from a `LongitudinalStudy`.

The config file is auto-discovered in the following order:
1. Current working directory: dosimetry_fit_defaults.json
2. Parent directory of CWD: dosimetry_fit_defaults.json
3. Package template: pytheranostics.data/configuration_templates/dosimetry_fit_defaults.json

Notes
-----
- Bounds may include string values like "inf" and special expressions
  (e.g., "log2_over_(6.647*24)_per_hour") which are parsed to numeric values.
- Lesion discovery is controlled via the config's `lesions` section.

"""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _parse_special_value(val):
    """Parse special string values to numeric where applicable.

    Parameters
    ----------
    val : any
        Value from config that may be a special string.

    Returns
    -------
    any
        Parsed value.
    """
    if isinstance(val, str):
        if val == "inf":
            from math import inf

            return inf
        if val == "log2_over_(6.647*24)_per_hour":
            import numpy as np

            return float(np.log(2) / (6.647 * 24))
    return val


def _parse_bounds(bounds):
    """Parse bounds mapping converting special strings to numeric values.

    Parameters
    ----------
    bounds : dict | None
        Bounds mapping from the config.

    Returns
    -------
    dict | None
        Parsed bounds or None.
    """
    if bounds is None:
        return None
    parsed = {}
    for k, pair in bounds.items():
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            parsed[k] = (_parse_special_value(pair[0]), _parse_special_value(pair[1]))
        else:
            parsed[k] = pair
    return parsed


def _load_config(config_path: Optional[Path | str] = None) -> Dict:
    """Load dosimetry fit defaults JSON from project or package template.

    Parameters
    ----------
    config_path : Path | str | None
        Optional explicit path to a JSON config.

    Returns
    -------
    dict
        The loaded configuration mapping.
    """
    # Explicit path
    if config_path is not None:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Dosimetry config not found: {p}")
        with p.open("r") as f:
            return json.load(f)

    # Project search
    search_paths = [
        Path.cwd() / "dosimetry_fit_defaults.json",
        Path.cwd().parent / "dosimetry_fit_defaults.json",
    ]
    for p in search_paths:
        if p.exists():
            try:
                with p.open("r") as f:
                    return json.load(f)
            except Exception:
                continue

    # Package template fallback
    try:
        import importlib.resources as pkg_resources

        template = pkg_resources.files("pytheranostics.data").joinpath(
            "configuration_templates/dosimetry_fit_defaults.json"
        )
        with open(template, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load package dosimetry template: %s", e)
        raise


def build_roi_fit_config(
    longSPECT, config_path: Optional[Path | str] = None
) -> Dict[str, Dict]:
    """Build ROI fit configuration for organs and lesions.

    Parameters
    ----------
    longSPECT : LongitudinalStudy
        The SPECT longitudinal study; used to auto-discover lesion names from masks.
    config_path : Path | str | None
        Optional explicit path to a JSON config. If None, auto-discovery is used.

    Returns
    -------
    dict
        Mapping of ROI name to fit parameter dict.
    """
    cfg = _load_config(config_path)

    organ_defaults = deepcopy(cfg.get("organ_defaults", {}))
    lesion_defaults = deepcopy(cfg.get("lesion_defaults", {}))

    # Parse bounds for both defaults
    if "bounds" in organ_defaults:
        organ_defaults["bounds"] = _parse_bounds(organ_defaults.get("bounds"))
    if "bounds" in lesion_defaults:
        lesion_defaults["bounds"] = _parse_bounds(lesion_defaults.get("bounds"))

    # Get all masks actually available in the study
    all_masks = set()
    for tp_masks in getattr(longSPECT, "masks", {}).values():
        all_masks.update(tp_masks.keys())

    # Get valid organ names from config
    try:
        from pytheranostics.imaging_ds.longitudinal_study import LongitudinalStudy

        valid_names = LongitudinalStudy._get_valid_organ_names()
    except Exception:
        valid_names = []

    lesion_pattern_str = cfg.get("lesions", {}).get("pattern", r"^Lesion_(\\d+)$")
    lesion_pattern = re.compile(lesion_pattern_str)

    # Use intersection: organs that are both valid AND actually present in masks
    organ_names = [
        name
        for name in valid_names
        if name in all_masks and not lesion_pattern.match(name)
    ]

    roi_config: Dict[str, Dict] = {}

    # Initialize all organs with defaults
    for name in organ_names:
        roi_config[name] = deepcopy(organ_defaults)

    # Apply explicit organ overrides
    for name, override in cfg.get("organs", {}).items():
        base = roi_config.get(name, deepcopy(organ_defaults))
        merged = deepcopy(base)
        for k, v in override.items():
            if k == "bounds":
                merged[k] = _parse_bounds(v)
            else:
                merged[k] = v
        roi_config[name] = merged

    # Auto-discover lesions from masks
    lesions_cfg = cfg.get("lesions", {})
    if lesions_cfg.get("auto_discover", True):
        discovered: set[str] = set()
        for tp_masks in getattr(longSPECT, "masks", {}).values():
            for mask_name in tp_masks.keys():
                if lesion_pattern.match(mask_name):
                    discovered.add(mask_name)
        if discovered:
            logger.debug("Discovered lesions: %s", sorted(discovered))
        for lesion in sorted(discovered):
            lesion_entry = deepcopy(lesion_defaults)
            roi_config[lesion] = lesion_entry

    logger.info(
        "Configured %d organs + %d lesions",
        len([n for n in roi_config.keys() if not lesion_pattern.match(n)]),
        len([n for n in roi_config.keys() if lesion_pattern.match(n)]),
    )

    return roi_config
