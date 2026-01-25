.. PyTheranostics documentation master file, created by
   sphinx-quickstart on Thu May 23 12:47:39 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyTheranostics Documentation
============================

PyTheranostics is a comprehensive Python library for nuclear medicine image processing and dosimetry calculations. It provides a complete workflow from image processing to absorbed dose calculations in target organs.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   intro/overview
   intro/installation
   intro/logging
   tutorials/getting_started/basic_usage

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   API/modules
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
------------

You can install PyTheranostics using pip:

.. code-block:: bash

   pip install pytheranostics

For development installation:

.. code-block:: bash

   pip install -e ".[dev]"

Quick Start
-----------

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

We would like to thank everyone who has contributed to PyTheranostics. Visit the
`GitHub contributors graph <https://github.com/qurit/PyTheranostics/graphs/contributors>`_
for the up-to-date list of collaborators.

.. footer::

   Made with 💖 by the Pytheranostics team.
