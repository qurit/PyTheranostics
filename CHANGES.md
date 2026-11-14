# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - unreleased

This is the first release since 0.1 and covers roughly 350 commits, so the
notes below are grouped by theme rather than listing every change.

Version 0.2.0 was never published: the number was carried in the source tree
but no release was ever built from it, so it is skipped here to keep the
published history unambiguous.

### Added
- Quantitative SPECT conversion: GE `Pixel Scale` handling, an optional
  Siemens frame-duration fallback, and a selectable RTSTRUCT contour mode.
- Weighted time-activity curve fitting, with per-VOI `sigma` configured as a
  fraction, a square root, or an absolute value.
- Project scaffolding (`init_project`, configuration templates) and a study
  inventory script.
- TotalSegmentator integration and ROI mapping utilities.
- A documentation site built with Furo, including runnable tutorials for
  project setup, DICOM organisation, data ingestion, segmentation, qSPECT,
  ROI mapping and voxel dosimetry.
- A check that blocks medical image data from entering the repository, run
  both as a pre-commit hook and as a required CI job.

### Changed
- **Breaking:** the minimum supported Python is now 3.10. The package
  previously declared 3.8, which was never tested and is end of life.
- **Breaking:** `Radiomics` no longer carries hardcoded study paths.
  `output_dir` is now a required keyword argument and `params_path` defaults
  to the packaged `Params.yaml`.
- Continuous integration now tests Python 3.10 through 3.14.
- Releases are published from GitHub Releases through PyPI Trusted
  Publishing; the previous workflow could not publish at all.
- Development happens on a single `main` branch; the `dev` branch is retired.

### Removed
- Example data and hardcoded study paths from the package, the tests and the
  documentation.

### Fixed
- qSPECT UID generation for long DICOM prefixes.
- Scan datetime now derives from acquisition metadata.
- Signed pixel value representation is resolved before a qSPECT file is
  saved, and input pixel mappings are applied before conversion.
- DICOM pixel metadata after int16 quantization.
- RTSTRUCT validation when the observation sequence is absent.

## [0.1.0] - 2024-03-19

### Added
- Initial release of PyTheranostics
- Basic package structure and modules
- Core functionality for nuclear medicine image processing
- Dosimetry calculation tools
- DICOM handling capabilities
- Calibration utilities
- Quality control tools
- Registration and segmentation modules
- Visualization and plotting features

### Changed
- Migrated from doodle to PyTheranostics
- Updated package name and structure
- Improved documentation

### Fixed
- Initial bug fixes and improvements
