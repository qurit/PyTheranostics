.. _logging_guide:

Logging and Verbosity Control
==============================

PyTheranostics uses Python's standard ``logging`` module to provide configurable output verbosity. This allows you to control how much information is displayed during processing, making it suitable for both interactive notebook use and automated pipelines.

Overview
--------

The logging system provides four severity levels:

.. list-table::
   :widths: 15 30 55
   :header-rows: 1

   * - Level
     - When to Use
     - What You'll See
   * - ``DEBUG``
     - Troubleshooting and development
     - Every step: individual timepoint loading, mask resampling operations, registration iterations
   * - ``INFO``
     - Normal production use
     - Important milestones: "Loaded 5 CT timepoints", "Applied mappings", file save operations
   * - ``WARNING``
     - Default (quiet mode)
     - Only important notices: mask overwrites, data quality issues, deprecated features
   * - ``ERROR``
     - Critical issues only
     - Fatal errors that prevent processing

Quick Start
-----------

Set Logging Level in Notebooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add this at the top of your notebook to control verbosity:

.. code-block:: python

   import logging
   
   # Quiet mode (recommended for notebooks) - only warnings and errors
   logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
   
   # Normal mode - see important milestones
   logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
   
   # Verbose mode - see every step (for debugging)
   logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

.. note::
   Call ``basicConfig()`` **before** importing PyTheranostics modules for the settings to take effect.

Example: Clean Notebook Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.WARNING)  # Quiet mode
   
   from pytheranostics.imaging_ds.cycle_loader import create_studies_with_masks
   
   # This will now run quietly, only showing warnings/errors
   longCT, longSPECT, inj, mappings = create_studies_with_masks(
       storage_root="./data",
       patient_id="PATIENT001",
       cycle_no=1,
       calibration_factor=106.0,
       auto_map=True
   )

Example: Debugging with Verbose Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s: %(message)s')
   
   from pytheranostics.imaging_ds.cycle_loader import create_studies_with_masks
   
   # This will show detailed progress:
   # - Loading timepoint 0 from CT_TP1...
   # - ✓ Timepoint 0 loaded
   # - Resampling Masks: Liver ...
   # - etc.
   longCT, longSPECT, inj, mappings = create_studies_with_masks(...)

Affected Modules
----------------

The following modules use the logging system:

Imaging & Data Loading
~~~~~~~~~~~~~~~~~~~~~~~

* ``pytheranostics.imaging_ds.longitudinal_study``
  
  - Logs timepoint loading progress (parallel and sequential)
  - Reports mask addition and overwrites
  - Shows registration iterations and Jaccard indices
  - Confirms file save operations

* ``pytheranostics.imaging_ds.mapping_summary``
  
  - Reports mapping statistics per timepoint
  - Shows applied mappings when verbose mode enabled
  - Confirms JSON export of mapping details

* ``pytheranostics.imaging_tools.tools``
  
  - Reports DICOM metadata extraction (injected activity, patient weight)
  - Warns about orientation matrix corrections
  - Shows mask resampling and registration progress

Common Use Cases
----------------

Case 1: Interactive Notebook (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For clean notebook output, use WARNING level:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.WARNING, format='%(message)s')

This gives you a clean interface while still alerting you to important issues like:

* Mask overwrites
* Missing DICOM metadata
* Data quality warnings

Case 2: Production Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For automated workflows, use INFO level with more context:

.. code-block:: python

   import logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
       filename='processing.log'
   )

This logs important milestones to a file for audit trails.

Case 3: Debugging Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

When troubleshooting, enable DEBUG with full context:

.. code-block:: python

   import logging
   logging.basicConfig(
       level=logging.DEBUG,
       format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s: %(message)s'
   )

This shows every operation with timestamps and source locations.

Advanced Configuration
----------------------

Module-Specific Levels
~~~~~~~~~~~~~~~~~~~~~~~

You can set different log levels for different modules:

.. code-block:: python

   import logging
   
   # Global default: WARNING
   logging.basicConfig(level=logging.WARNING)
   
   # But show INFO for longitudinal study operations
   logging.getLogger('pytheranostics.imaging_ds.longitudinal_study').setLevel(logging.INFO)
   
   # And DEBUG for mapping summary
   logging.getLogger('pytheranostics.imaging_ds.mapping_summary').setLevel(logging.DEBUG)

Custom Formatters
~~~~~~~~~~~~~~~~~

Customize the output format:

.. code-block:: python

   import logging
   
   # Simple format for notebooks
   logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
   
   # Detailed format for log files
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s | %(name)-40s | %(levelname)-8s | %(message)s',
       datefmt='%Y-%m-%d %H:%M:%S'
   )

Troubleshooting
---------------

"Logging settings not taking effect"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: You set the logging level but still see too much or too little output.

**Solution**: Make sure to call ``logging.basicConfig()`` **before** importing any PyTheranostics modules:

.. code-block:: python

   # ✓ CORRECT ORDER
   import logging
   logging.basicConfig(level=logging.WARNING)
   from pytheranostics.imaging_ds import LongitudinalStudy  # Now uses WARNING level
   
   # ✗ WRONG ORDER
   from pytheranostics.imaging_ds import LongitudinalStudy  # Uses default level
   import logging
   logging.basicConfig(level=logging.WARNING)  # Too late!

"Multiple log messages appearing"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Each message appears multiple times or from different loggers.

**Solution**: This happens when ``basicConfig()`` is called multiple times. In Jupyter notebooks, restart the kernel and configure logging once at the start:

.. code-block:: python

   # In your first notebook cell:
   import logging
   
   # Only call this once per kernel session
   if not logging.getLogger().hasHandlers():
       logging.basicConfig(level=logging.INFO, format='%(message)s')

Migration from Previous Versions
---------------------------------

If you're upgrading from a version that used ``print()`` statements, no code changes are required. The transition is automatic:

* All previous ``print()`` output is now logged at INFO or WARNING level
* To restore previous behavior (seeing all messages), use ``logging.INFO``
* To reduce output in notebooks, use ``logging.WARNING``

See Also
--------

* :ref:`overview` - General introduction to PyTheranostics
* :ref:`installation` - Installation instructions
* `Python Logging Documentation <https://docs.python.org/3/library/logging.html>`_ - Official Python logging guide
