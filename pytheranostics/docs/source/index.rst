.. PyTheranostics documentation master file, created by
   sphinx-quickstart on Thu May 23 12:47:39 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyTheranostics's documentation!
=======================================

PyTheranostics is a comprehensive Python library for nuclear medicine image processing and dosimetry calculations. It provides a complete workflow from image processing to absorbed dose calculations in target organs.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   modules
   api
   contributing
   changelog

Features
--------

* Image processing and analysis
* Dosimetry calculations
* DICOM handling and manipulation
* Calibration tools
* Quality control utilities
* Registration and segmentation tools
* Visualization and plotting capabilities

Installation
-----------

You can install PyTheranostics using pip:

.. code-block:: bash

   pip install pytheranostics

For development installation:

.. code-block:: bash

   pip install -e ".[dev]"

Quick Start
----------

.. code-block:: python

   import pytheranostics as pth

   # Load and process images
   image = pth.ImagingDS.load_dicom("path/to/dicom")

   # Perform dosimetry calculations
   dose = pth.dosimetry.calculate_absorbed_dose(image)

   # Visualize results
   pth.plots.plot_dose_distribution(dose)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
-------

This project is licensed under the terms of the MIT license. See the `LICENSE <https://github.com/your-repo/LICENSE>`_ file for details.

Acknowledgements
----------------

We would like to thank the following contributors for their work on this project:

..  contributors:: qurit/PyTheranostics
   .. :avatars:
   .. :exclude: dependabot[bot] 

.. footer::

   Made with 💖 by the Pytheranostics team.
