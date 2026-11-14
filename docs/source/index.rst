:hide-toc:

============================
PyTheranostics Documentation
============================

.. raw:: html

   <div class="pt-hero">
     <p class="pt-tagline">From nuclear medicine scans to absorbed dose in target organs</p>
     <p class="pt-badges">
       <a href="https://pypi.org/project/pytheranostics"><img src="https://img.shields.io/pypi/v/pytheranostics" alt="PyPI version" height="20"></a>
       <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+" height="20">
       <a href="https://github.com/qurit/PyTheranostics/blob/main/LICENSE"><img src="https://img.shields.io/github/license/qurit/PyTheranostics?color=g" alt="MIT licence" height="20"></a>
       <a href="https://www.qurit.ca/"><img src="https://img.shields.io/badge/built%20at-QURIT-0f766e" alt="Built at QURIT" height="20"></a>
     </p>
   </div>

PyTheranostics is a library of tools to process nuclear medicine scans and take
them through the dosimetry workflow: organising a study's DICOM, making a SPECT
quantitative, segmenting volumes of interest, fitting their time–activity
curves, and calculating the absorbed dose in target organs.

Install it with:

.. code-block:: bash

   pip install pytheranostics

----

Start here
==========

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket;1.2em` Overview
      :link: intro/overview
      :link-type: doc

      What the library does and how the dosimetry workflow fits together.
      **Read this first** if you are new to PyTheranostics.

   .. grid-item-card:: :octicon:`download;1.2em` Installation
      :link: intro/installation
      :link-type: doc

      Supported Python versions, the dependencies, and the optional extras.

   .. grid-item-card:: :octicon:`book;1.2em` Tutorials
      :link: tutorials/index
      :link-type: doc

      Runnable notebooks: project setup, DICOM organisation, data ingestion,
      segmentation, qSPECT and ROI mapping.

   .. grid-item-card:: :octicon:`code;1.2em` API reference
      :link: API/modules
      :link-type: doc

      Every module, generated from the docstrings in the source.

What is in it
=============

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Area
     - Tools
   * - Study organisation
     - ``imaging_ds`` — DICOM ingestion, longitudinal studies, cycle loading,
       ROI mapping
   * - Quality control
     - ``qc`` — radionuclide calibrator, planar and SPECT checks against a
       stored protocol
   * - Calibration
     - ``calibrations`` — camera sensitivity and calibration factors
   * - Quantitative imaging
     - ``dicomtools`` — counts to Bq/mL, plus the header a viewer needs for SUV
   * - Segmentation
     - ``segmentation`` — RTStruct masks, and TotalSegmentator integration
   * - Registration
     - ``registration`` — aligning the series a study spans
   * - Kinetics
     - ``fits`` — mono-, bi- and tri-exponential time–activity curve fitting
   * - Dosimetry
     - ``dosimetry`` — dose kernels, Monte Carlo and organ-level S-values
   * - Preclinical
     - ``preclinical_dosimetry`` — the same workflow for small-animal studies
   * - Visualisation
     - ``plots`` — energy-window montages and time–activity curve figures

.. toctree::
   :hidden:
   :caption: Getting Started

   intro/overview
   intro/installation
   intro/logging
   intro/dosimetry_config
   tutorials/getting_started/basic_usage

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/index

.. toctree::
   :hidden:
   :caption: Reference

   API/modules
   changelog

Citing and licence
==================

PyTheranostics is developed at the `Quantitative Radiomolecular Imaging and
Therapy Lab (QURIT) <https://www.qurit.ca/>`_ at BC Cancer Research Institute,
and is released under the MIT licence — see
`LICENSE <https://github.com/qurit/PyTheranostics/blob/main/LICENSE>`_.

We would like to thank everyone who has contributed. The up-to-date list is on
the `GitHub contributors graph
<https://github.com/qurit/PyTheranostics/graphs/contributors>`_.
