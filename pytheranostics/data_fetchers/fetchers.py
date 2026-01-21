"""Download and cache example data for tutorials and testing."""

import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen


def get_data_home(data_home: Optional[str] = None) -> Path:
    """Return the path to the pytheranostics example data directory.

    By default, data is stored in the user's cache directory.

    Parameters
    ----------
    data_home : str, optional
        The path to the pytheranostics example data directory. If None, the default
        path is used: `~/.pytheranostics_example_data`.

    Returns
    -------
    Path
        The path to the data home directory.
    """
    if data_home is None:
        data_home = Path.home() / ".pytheranostics_example_data"
    else:
        data_home = Path(data_home)

    data_home.mkdir(parents=True, exist_ok=True)
    return data_home


def _verify_checksum(filepath: Path, expected_md5: str) -> bool:
    """Verify file integrity using MD5 checksum.

    Parameters
    ----------
    filepath : Path
        Path to the file to verify.
    expected_md5 : str
        Expected MD5 hash.

    Returns
    -------
    bool
        True if checksum matches, False otherwise.
    """
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest() == expected_md5


def clear_data_cache(data_home: Optional[str] = None):
    """Remove all cached example data.

    Parameters
    ----------
    data_home : str, optional
        The path to the pytheranostics data directory. If None, uses default.

    Examples
    --------
    >>> from pytheranostics.data import clear_data_cache
    >>> clear_data_cache()
    """
    data_home = get_data_home(data_home)
    if data_home.exists():
        shutil.rmtree(data_home)
        print(f"Cleared data cache at: {data_home}")
    else:
        print("No cached data to clear")


def list_cached_data(data_home: Optional[str] = None):
    """List all cached example datasets.

    Parameters
    ----------
    data_home : str, optional
        The path to the pytheranostics data directory. If None, uses default.

    Examples
    --------
    >>> from pytheranostics.data import list_cached_data
    >>> list_cached_data()
    """
    data_home = get_data_home(data_home)
    if not data_home.exists():
        print("No cached data found")
        return

    print(f"Cached data in {data_home}:")
    for item in data_home.iterdir():
        if item.is_dir():
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            size_mb = size / (1024 * 1024)
            print(f"  - {item.name}: {size_mb:.2f} MB")


def get_example_data_citation() -> str:
    """Get proper citation for example datasets.

    Returns
    -------
    str
        BibTeX citation for the example data.
    """
    citation = """@dataset{umich_imaging_data_2024,
  title = {Example CT and SPECT DICOM Data for Medical Image Processing},
  author = {Contributors},
  year = {2024},
  doi = {10.7302/864r-tb45},
  url = {https://deepblue.lib.umich.edu/},
  note = {University of Michigan Deep Blue Repository}
}"""
    return citation


def fetch_snmmi_dosimetry_challenge(
    data_home: Optional[str] = None, download: bool = True
) -> None:
    """Fetch the SNMMI Dosimetry Challenge dataset.

    This dataset contains anonymized CT and SPECT DICOM images suitable for
    testing segmentation and dosimetry workflows. Data is sourced from the
    University of Michigan Deep Blue repository (DOI: 10.7302/864r-tb45).

    Parameters
    ----------
    data_home : str, optional
        The path to store the data. If None, uses `~/.pytheranostics_example_data`.
    download : bool, optional
        If True (default), download the data if not already present.
        If False, raise an error if data is not found.

    Returns
    -------
    None
        Prints download information to console. Access data from the cache directory.

    Raises
    ------
    RuntimeError
        If download=False and data is not found locally.

    Examples
    --------
    >>> from pytheranostics.data_fetchers import fetch_snmmi_dosimetry_challenge
    >>> fetch_snmmi_dosimetry_challenge()
    Downloading multi-timepoint Lu-177 SPECT/CT data...
    Extracting...
    Extraction complete ✓

    Data ready at: ~/.pytheranostics_example_data/snmmi_dose_challenge

    >>> # Access multi-timepoint data
    >>> from pathlib import Path
    >>> data_home = Path.home() / ".pytheranostics_example_data" / "snmmi_dose_challenge"
    >>> patient_004 = data_home / "Patient_004" / "SPECT_Cts"
    >>>
    >>> # List available scans (scan1-scan4)
    >>> scans = sorted([d.name for d in patient_004.iterdir() if d.is_dir()])
    >>> print(scans)  # ['scan1', 'scan2', 'scan3', 'scan4']
    >>>
    >>> # Access specific scan data
    >>> scan1_ct = list((patient_004 / "scan1" / "ct").glob("*.dcm"))
    >>> scan1_spect = list((patient_004 / "scan1" / "spect").glob("*.dcm"))

    Notes
    -----
    - Data is automatically cached in `~/.pytheranostics_example_data/`
    - First download may take several minutes depending on connection speed
    - Subsequent calls use cached data instantly
    - Dataset contains multi-timepoint SPECT/CT for Patient_004

    References
    ----------
    Dataset DOI: https://doi.org/10.7302/864r-tb45
    Repository: https://deepblue.lib.umich.edu/
    """
    home = get_data_home(str(data_home) if data_home else None)

    dataset_base = "snmmi_dose_challenge"
    patient_dir = home / dataset_base / "Patient_004"

    if download:
        if patient_dir.exists():
            print(f"Example data already exists at: {patient_dir}")
            print("Use download=False to skip re-download")
        else:
            url = "https://deepblue.lib.umich.edu/data/downloads/tb09j589z"
            zip_path = home / "snmmi_dosimetry_challenge.zip"

            print("Downloading multi-timepoint Lu-177 SPECT/CT data...")
            try:
                request = Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "gzip, deflate, br",
                        "Referer": "https://deepblue.lib.umich.edu/",
                        "DNT": "1",
                        "Connection": "keep-alive",
                        "Upgrade-Insecure-Requests": "1",
                        "Sec-Fetch-Dest": "document",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": "same-origin",
                    },
                )
                with urlopen(request) as response:
                    with open(zip_path, "wb") as out_file:
                        out_file.write(response.read())
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download data from Deep Blue: {e}\n\n"
                    f"If you have the data files locally, place them in:\n"
                    f"~/.pytheranostics_example_data/snmmi_dose_challenge/"
                )

            print("Extracting...")
            temp_extract_dir = home / "snmmi_dosimetry_challenge_temp"
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_extract_dir)
            print("Extraction complete ✓")

            patient_dir.parent.mkdir(parents=True, exist_ok=True)
            if patient_dir.exists():
                shutil.rmtree(patient_dir)

            extracted_contents = list(temp_extract_dir.iterdir())
            if len(extracted_contents) == 1 and extracted_contents[0].is_dir():
                shutil.move(str(extracted_contents[0]), str(patient_dir))
            else:
                shutil.move(str(temp_extract_dir), str(patient_dir))

            zip_path.unlink()
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir)

        print(f"\nData ready at: {patient_dir.parent}")
        print("\nDataset citation:")
        print("  DOI: https://doi.org/10.7302/864r-tb45")
        print("  Repository: University of Michigan Deep Blue")
    else:
        if not patient_dir.exists():
            raise RuntimeError(
                f"Dataset not found in {home}. "
                "Use fetch_snmmi_dosimetry_challenge() to download, or download manually from: "
                "https://deepblue.lib.umich.edu/data/concern/data_sets/th83kz366"
            )
