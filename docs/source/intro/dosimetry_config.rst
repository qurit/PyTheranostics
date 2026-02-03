Dosimetry Fit Defaults
======================

PyTheranostics supports a config-driven workflow for ROI dosimetry fits.
This page describes the configuration format, auto-discovery logic,
and how to use the `build_roi_fit_config` API in notebooks and scripts.

Overview
--------

- Central config file controls default fit parameters for organs and lesions.
- Project-specific overrides are supported; otherwise a packaged template is used.
- Lesion ROIs are auto-discovered from `longSPECT.masks` (when enabled).

Project Initialization
----------------------

When creating a new PyTheranostics project, the dosimetry config is automatically
generated alongside other project templates:

.. code-block:: python

   from pytheranostics.project import init_project

   # Creates project directory with all config templates
   init_project("./my_dosimetry_project")

This copies `dosimetry_fit_defaults.json` to your project root so you can
**customize it without touching code**. Edit the file directly to adjust organ
defaults, BoneMarrow kinetics, lesion bounds, etc.

Config Discovery
----------------

The loader searches for `dosimetry_fit_defaults.json` in this order:

1. Current working directory
2. Parent directory of the current working directory
3. Packaged template: `pytheranostics.data/configuration_templates/dosimetry_fit_defaults.json`

If a project file is found, it overrides the packaged template.

Config Schema
-------------

Example JSON (packaged template):

.. code-block:: json

   {
     "organ_defaults": {
       "fit_order": 1,
       "with_uptake": false,
       "param_init": {"A1": 100, "A2": 0.01},
       "fixed_parameters": null,
       "bounds": null,
       "washout_ratio": null
     },
     "organs": {
       "BoneMarrow": {
         "fit_order": 2,
         "param_init": {"A1": 50},
         "fixed_parameters": {"A2": 0.045788, "B1": 561.376560, "B2": 0.213215},
         "with_uptake": null,
         "bounds": null,
         "washout_ratio": 4.656569406831483
       }
     },
     "lesion_defaults": {
       "fit_order": 1,
       "param_init": {"A1": 700, "A2": 0.1},
       "fixed_parameters": null,
       "bounds": {"A1": [0, "inf"], "A2": ["log2_over_(6.647*24)_per_hour", "inf"]},
       "with_uptake": false,
       "washout_ratio": null
     },
     "lesions": {
       "auto_discover": true,
       "pattern": "^Lesion_(\\d+)$"
     }
   }

Notes:
- Bounds support special values: "inf" and "log2_over_(6.647*24)_per_hour".
- Organ overrides only replace specified fields; unspecified fields inherit defaults.

API Usage
---------

.. code-block:: python

   import logging
   from pytheranostics.dosimetry import build_roi_fit_config

   logging.basicConfig(level=logging.INFO, format='%(message)s')

   roi_config = build_roi_fit_config(longSPECT)
   # roi_config["Liver"] -> {fixed_parameters, fit_order, param_init, ...}
   # roi_config["Lesion_3"] -> lesion defaults applied

Project Overrides
-----------------

After running `initialize_project_dosimetry_config()`, edit the generated
`dosimetry_fit_defaults.json` in your project root to customize:

- `organ_defaults`: Apply to all organs unless overridden
- `organs.<OrganName>`: Override specific organ parameters (e.g., BoneMarrow kinetics)
- `lesion_defaults`: Apply to all auto-discovered lesions
- `lesions.pattern`: Regex pattern for lesion ROI names
- `lesions.auto_discover`: Enable/disable automatic lesion discovery

The loader will pick up your project file automatically.

Validation & Best Practices
---------------------------

- Keep institution-specific values in project overrides, not code.
- Review bounds and fixed parameters for each organ.
- Version your overrides and document assumptions.
