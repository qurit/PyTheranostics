# PyTheranostics workflows

This directory contains task-oriented notebooks for running common processing
steps. Unlike the notebooks under `docs/source/tutorials`, these workflows are
intended to be configured and executed on local data rather than read as guided
lessons.

## Available workflows

- `qSPECT/counts_to_bqml.ipynb`: convert reconstructed SPECT DICOM images from
  scanner counts to quantitative Bq/mL DICOM files.

## Usage conventions

- Review and edit each notebook's configuration cells before execution.
- Keep patient data, credentials, and generated outputs outside the repository.
- Do not commit executed cell outputs containing patient information.
- Move reusable processing logic into `pytheranostics`; notebooks should focus
  on configuration, orchestration, review, and quality control.

