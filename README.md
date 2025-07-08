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

## Database

To use save to a locally hosted database, follow these instructions:

1. [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)
1. [Clone the AscintaDB repo](https://github.com/jasonspence/AscintaDB)
1. From this repo in your virtual environment, run `pip install ../path/to/AscintaDB`
1. Create a .env file, run `cp .env.template .env`
1. Turn on Docker Desktop
1. Run the startup script `ascintadb` - this turns on the database
1. Run the test script `python pytheranostics/tests/test_database_integration.py`

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

