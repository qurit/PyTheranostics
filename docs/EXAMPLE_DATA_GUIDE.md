# Managing Example Data for PyTheranostics

This guide explains how PyTheranostics manages example DICOM data for tutorials and testing.

## Current Data Source

PyTheranostics uses example data hosted in the **University of Michigan Deep Blue repository**:

- **DOI**: https://doi.org/10.7302/864r-tb45
- **Dataset**: Multi-timepoint Lu-177 SPECT/CT for Patient 004
- **Download URL**: https://deepblue.lib.umich.edu/data/downloads/tb09j589z
- **More details**: https://deepblue.lib.umich.edu/data/concern/data_sets/th83kz366?locale=en

## How Users Access Data

Users simply run:

```python
from pytheranostics.data_fetchers import fetch_snmmi_dosimetry_challenge

fetch_snmmi_dosimetry_challenge()
# Output: Data ready at ~/.pytheranostics_example_data/snmmi_dose_challenge

# Access multi-timepoint SPECT/CT data
from pathlib import Path
data_home = Path.home() / ".pytheranostics_example_data" / "snmmi_dose_challenge"
patient_004 = data_home / "Patient_004" / "SPECT_Cts"

# List available scans (4 timepoints)
scans = sorted([d.name for d in patient_004.iterdir() if d.is_dir()])
print(f"Available scans: {scans}")  # ['scan1', 'scan2', 'scan3', 'scan4']

# Access specific scan data
scan1_ct = list((patient_004 / "scan1" / "ct").glob("*.dcm"))
scan1_spect = list((patient_004 / "scan1" / "spect").glob("*.dcm"))
print(f"Scan 1 - CT: {len(scan1_ct)} slices, SPECT: {len(scan1_spect)} slices")
```

### Automatic Data Organization

PyTheranostics automatically extracts and organizes the Deep Blue dataset into a clean, scannable structure:

**Organized to multi-timepoint scan structure:**
```
~/.pytheranostics_example_data/
└── snmmi_dose_challenge/
    └── Patient_004/
        └── SPECT_Cts/
            ├── scan1/
            │   ├── ct/        (CT DICOM series)
            │   └── spect/     (SPECT DICOM series)
            ├── scan2/
            │   ├── ct/
            │   └── spect/
            ├── scan3/
            │   ├── ct/
            │   └── spect/
            └── scan4/
                ├── ct/
                └── spect/
```

This structure makes it easy to work with longitudinal/multi-timepoint data for dosimetry calculations!

## How to Update Example Data (For Maintainers)

If you need to update the example datasets:

### Uploading to GitHub Releases

1. **Prepare your data:**
   - Anonymize all DICOM files (remove PHI/PII)
   - Organize in logical directory structure
   - Create ZIP archive
   - Verify integrity

2. **Calculate checksum:**
   ```bash
   md5sum example_ct_data.zip
   # or on macOS:
   md5 example_ct_data.zip
   ```
   Save this checksum - you'll need it for the download function.

3. **Create GitHub Release:**
   - Go to: https://github.com/qurit/PyTheranostics/releases
   - Click "Create a new release"
   - Tag version: `data-v1.0` (or increment as needed)
   - Release title: "Example Data v1.0"
   - Description:
     ```
     Example anonymized CT and SPECT DICOM data for PyTheranostics tutorials.
     
     - Size: XX MB
     - Content: Anonymized medical imaging data for segmentation and dosimetry testing
     - Anonymization: All PHI removed per HIPAA/GDPR requirements
     - Source: University of Michigan
     - DOI: https://doi.org/10.7302/864r-tb45
     ```
   - Upload your ZIP files (both `example_ct_data.zip` and `example_spect_data.zip`)
   - Publish release

4. **Update download function:**
   Edit `pytheranostics/data/example_data.py`:
   ```python
   # Get actual download URL from release assets page
   ct_url = "https://github.com/qurit/PyTheranostics/releases/download/data-v1.0/example_ct_data.zip"
   spect_url = "https://github.com/qurit/PyTheranostics/releases/download/data-v1.0/example_spect_data.zip"
   
   # Use checksums from step 2
   ct_md5 = "abc123def456..."
   spect_md5 = "xyz789uvw123..."
   ```

5. **Test the download:**
   ```python
   from pytheranostics.data_fetchers import fetch_snmmi_dosimetry_challenge
   
   # Test download with force_download=True
   fetch_snmmi_dosimetry_challenge()
   
   print("Data download test successful!")
   ```

6. **Update documentation:**
   - Update links in README
   - Update this guide with new release version
   - Update citation information if needed
   - Update tutorial documentation

## Data Preparation Requirements

### Anonymization

Before uploading **any** medical data:

- [ ] Remove patient name, ID, date of birth
- [ ] Remove study dates (or shift consistently)
- [ ] Remove institution/physician names
- [ ] Strip all identifiable metadata
- [ ] Use DICOM anonymization tool to verify
- [ ] Obtain IRB approval if required
- [ ] Check institutional data sharing policy
- [ ] Document anonymization process

### DICOM Quality

**For CT Data:**
- Complete series (all slices)
- Proper DICOM metadata (ImagePositionPatient, etc.)
- Consistent spacing and orientation
- Suitable for TotalSegmentator
- Reasonable image quality

**For SPECT Data:**
- Multiple timepoints (0.5h, 6h, 24h)
- Proper calibration metadata
- Suitable for dosimetry calculations
- Energy windows properly set

### File Organization

```
data/
├── Patient_001/
│   ├── CT_0p5h/
│   │   ├── 001.dcm
│   │   ├── 002.dcm
│   │   └── ...
│   └── SPECT_0p5h/
│       ├── 001.dcm
│       └── ...
└── Patient_002/
    ├── CT_24h/
    └── SPECT_24h/
```

## Update Checklist

When adding/updating example data:

- [ ] Data properly anonymized and verified
- [ ] DICOM quality checked
- [ ] Compressed and organized
- [ ] Uploaded to repository (Deep Blue, Zenodo, etc.)
- [ ] DOI obtained
- [ ] URLs updated in `pytheranostics/data/example_data.py`
- [ ] Citation information updated
- [ ] Tutorial documentation updated
- [ ] Tested download and extraction
- [ ] Pre-commit checks pass
- [ ] Release notes updated

## Citing Example Data

If you use PyTheranostics with the provided example data, cite both:

```bibtex
@software{pytheranostics2024,
  author = {Uribe, Carlos and Esquinas, Pedro and Kurkowska, Sara},
  title = {PyTheranostics: A Python Library for Nuclear Medicine Processing and Dosimetry},
  year = {2024},
  url = {https://github.com/qurit/PyTheranostics}
}

@dataset{umich_imaging_data,
  title = {Example CT and SPECT DICOM Data for Medical Image Processing},
  doi = {10.7302/864r-tb45},
  url = {https://deepblue.lib.umich.edu/},
  note = {University of Michigan Deep Blue Repository}
}
```

Get citation programmatically:

```python
from pytheranostics.data import get_example_data_citation
print(get_example_data_citation())
```

## Accessing Data Programmatically

```python
from pathlib import Path
from pytheranostics.datasets import fetch_snmmi_dosimetry_challenge

# Download and organize data
fetch_snmmi_dosimetry_challenge()

# Access data location
data_home = Path.home() / ".pytheranostics_example_data" / "snmmi_dose_challenge"
patient_004 = data_home / "Patient_004" / "SPECT_Cts"

# Iterate through all scans
for scan_dir in sorted(patient_004.iterdir()):
    if scan_dir.is_dir():
        ct_files = list((scan_dir / "ct").glob("*.dcm"))
        spect_files = list((scan_dir / "spect").glob("*.dcm"))
        print(f"{scan_dir.name}: {len(ct_files)} CT + {len(spect_files)} SPECT")

# Access specific scan
scan2_ct_path = patient_004 / "scan2" / "ct"
scan2_spect_path = patient_004 / "scan2" / "spect"
```

## Example Anonymization Script

```python
import pydicom
from pathlib import Path

def anonymize_dicom(input_dir, output_dir):
    """Anonymize DICOM files by removing PHI."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for dcm_file in input_path.rglob("*.dcm"):
        ds = pydicom.dcmread(dcm_file)
        
        # Remove patient identifiers
        ds.PatientName = "ANONYMOUS"
        ds.PatientID = "ANON001"
        ds.PatientBirthDate = ""
        
        # Shift dates consistently
        if hasattr(ds, 'StudyDate'):
            ds.StudyDate = "20200101"
        if hasattr(ds, 'SeriesDate'):
            ds.SeriesDate = "20200101"
            
        # Remove institution info
        ds.InstitutionName = ""
        ds.InstitutionAddress = ""
        
        # Save anonymized file
        rel_path = dcm_file.relative_to(input_path)
        out_file = output_path / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(out_file)
        
    print(f"Anonymized {len(list(input_path.rglob('*.dcm')))} files")
    print(f"Saved to: {output_path}")

# Usage
# anonymize_dicom("raw_data/", "anonymized_data/")
```

## Questions?

Contact the PyTheranostics maintainers or open an issue on GitHub.
