[![Documentation Status](https://readthedocs.org/projects/pytheranostics/badge/?version=latest)](https://docs.pytheranostics.qurit.ca/en/latest/?badge=latest)

# PyTheranostics

A comprehensive Python library for nuclear medicine image processing and dosimetry calculations.

## Overview

PyTheranostics is a powerful toolkit designed for processing nuclear medicine scans and performing dosimetry calculations. It provides a complete workflow from image processing to absorbed dose calculations in target organs.

## Features

- Image processing and analysis
- Dosimetry calculations
- DICOM handling and manipulation
- Calibration tools
- Quality control utilities
- Registration and segmentation tools
- Visualization and plotting capabilities

## Installation

```bash
pip install pytheranostics
```

## Quick Start

```python
import pytheranostics as tx

# Load and process images
image = tx.ImagingDS.load_dicom("path/to/dicom")

# Perform dosimetry calculations
dose = tx.dosimetry.calculate_absorbed_dose(image)

# Visualize results
tx.plots.plot_dose_distribution(dose)
```

## Documentation

For detailed documentation, visit our [documentation page](https://pytheranostics.readthedocs.io/).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Citation

If you use PyTheranostics in your research, please cite:

```
@software{pytheranostics2024,
  author = {Sara Kurkowska, Pedro Esquinas,Carlos Uribe},
  title = {PyTheranostics: A Python Library for Nuclear Medicine Processing and Dosimetry},
  year = {2024},
  url = {https://github.com/qurit/PyTheranostics}
}
```
