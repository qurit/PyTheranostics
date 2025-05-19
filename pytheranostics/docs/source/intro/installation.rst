.. _intro-install:

==================
Installation Guide
==================

.. _faq-python-versions:

Supported Python versions
=========================

PyTheranostics requires Python 3.8 or higher.

.. _intro-install-pytheranostics:

Installing PyTheranostics
==========================

You can install PyTheranostics and its dependencies from PyPI with::

    pip install pytheranostics

We strongly recommend that you install it in a dedicated virtual environment
to avoid conflicting with your system packages.


Things that are good to know
----------------------------

PyTheranostics is written in pure Python and depends on a few key Python packages (among others):

* `numpy`_, a powerful Python library for numerical computing, providing support for large multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.

* `matplotlib`_, a comprehensive Python library for creating static, animated, and interactive visualizations, offering a wide range of plotting functions to generate publication-quality graphs, charts, and figures for data analysis and scientific exploration.

* `pandas`_, a versatile Python library for data manipulation and analysis, offering intuitive data structures like DataFrames and Series, along with powerful tools for cleaning, transforming, and analyzing structured data from various sources, facilitating tasks such as data exploration, manipulation, and visualization.

* `pydicom`_, a Python library for working with DICOM (Digital Imaging and Communications in Medicine) files, providing tools for reading, writing, and manipulating medical image data, facilitating tasks such as parsing metadata, extracting pixel data, and performing various operations on medical images in a standardized format.

* `openpyxl`_, a Python library for working with Excel files, enabling users to read, write, and manipulate Excel spreadsheets programmatically, offering support for various Excel formats and features such as cell formatting, formulas, charts, and more, facilitating tasks such as data extraction, analysis, and reporting directly from Excel files.

* `rt-utils`_, RT-Utils allows you to create or load RT Structs, extract 3d masks from RT Struct ROIs, easily add one or more regions of interest, and save the resulting RT Struct in just a few lines.

* `scikit-image`_, a collection of algorithms for image processing and computer vision, providing a wide range of tools for image analysis, segmentation, feature extraction, and more, facilitating tasks such as image enhancement, restoration, and transformation for scientific and industrial applications.

* `simpleitk`_, a simplified layer built on top of the Insight Segmentation and Registration Toolkit (ITK), providing a simplified interface for medical image processing tasks, such as image registration, segmentation, and filtering, facilitating tasks such as image analysis, visualization, and processing in medical imaging applications.



.. _numpy: https://numpy.org/
.. _matplotlib: https://matplotlib.org/
.. _pandas: https://pandas.pydata.org/
.. _pydicom: https://pydicom.github.io/
.. _openpyxl: https://openpyxl.readthedocs.io/
.. _rt-utils: https://github.com/qurit/rt-utils
.. _scikit-image: https://scikit-image.org/
.. _simpleitk: https://simpleitk.readthedocs.io/en/master/