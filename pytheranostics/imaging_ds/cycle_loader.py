"""
Cycle data loading utilities for PyTheranostics.

Load organized DICOM data (CT, SPECT, RTSTRUCT) for dosimetry processing
from the folder structure produced by the DICOM receiver or manual organization.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pydicom


def _extract_hhmmss(tm: Optional[str]) -> Optional[str]:
    """Convert a DICOM TM string to HHMMSS (6 digits)."""
    if not tm:
        return None
    digits = "".join(ch for ch in str(tm) if ch.isdigit())
    if not digits:
        return None
    return (digits + "000000")[:6]


def list_cycle_timepoints(
    storage_root: str | Path, patient_id: str, cycle_no: int
) -> Tuple[List[Optional[Path]], List[Optional[Path]], List[Optional[Path]]]:
    """List CT, SPECT, and RTSTRUCT paths for a given patient cycle.

    Assumes the layout: storage_root/PatientID/CycleX/tpY/<Modality> with RTSTRUCT
    under CT/RTstruct. Returns three lists aligned by timepoint index.

    Parameters
    ----------
    storage_root : str | Path
        Base directory where organized data lives.
    patient_id : str
        Patient identifier.
    cycle_no : int
        Cycle number (1-based).

    Returns
    -------
    tuple[list[Path | None], list[Path | None], list[Path | None]]
        Lists of CT directories, SPECT directories, and RTSTRUCT files (one per timepoint).
    """
    root = Path(storage_root)
    cycle_dir = root / patient_id / f"Cycle{int(cycle_no)}"
    if not cycle_dir.exists():
        raise FileNotFoundError(f"Cycle directory not found: {cycle_dir}")

    # Sort tp folders by numeric suffix
    tp_dirs: List[Tuple[int, Path]] = []
    for p in cycle_dir.iterdir():
        if p.is_dir() and p.name.startswith("tp"):
            m = re.search(r"(\d+)$", p.name)
            idx = int(m.group(1)) if m else 0
            tp_dirs.append((idx, p))
    tp_dirs.sort(key=lambda x: x[0])

    ct_paths: List[Optional[Path]] = []
    spect_paths: List[Optional[Path]] = []
    rtstruct_files: List[Optional[Path]] = []

    for _, tp in tp_dirs:
        ct_dir = tp / "CT"
        spect_dir = tp / "SPECT"
        # CT path (directory) if present
        ct_paths.append(ct_dir if ct_dir.exists() else None)
        # SPECT path (directory) if present
        spect_paths.append(spect_dir if spect_dir.exists() else None)
        # RTSTRUCT file: pick first .dcm in CT/RTstruct if present
        rtstruct_dir = ct_dir / "RTstruct"
        rt_file: Optional[Path] = None
        if rtstruct_dir.exists():
            for f in sorted(rtstruct_dir.glob("*.dcm")):
                # Trust folder name; otherwise verify Modality
                try:
                    ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                    if getattr(ds, "Modality", "") == "RTSTRUCT":
                        rt_file = f
                        break
                except Exception:
                    continue
        rtstruct_files.append(rt_file)

    return ct_paths, spect_paths, rtstruct_files


def extract_injection_from_first_tp_spect(
    storage_root: str | Path, patient_id: str, cycle_no: int
) -> Dict[str, Optional[str | int]]:
    """Extract injection info from the first time point SPECT DICOM in a cycle.

    Returns a dictionary with keys: InjectionDate (YYYYMMDD), InjectionTime (HHMMSS),
    InjectedActivity (Bq, int) and PatientWeight_g (int). Values may be None/empty
    if not present in the DICOM headers.

    Parameters
    ----------
    storage_root : str | Path
        Base directory where organized data lives.
    patient_id : str
        Patient identifier.
    cycle_no : int
        Cycle number (1-based).

    Returns
    -------
    dict[str, str | int | None]
        Dictionary with InjectionDate, InjectionTime, InjectedActivity, PatientWeight_g.
    """
    _, spect_paths, _ = list_cycle_timepoints(storage_root, patient_id, cycle_no)
    if not spect_paths:
        raise RuntimeError("No time points found for SPECT")

    tp1_spect = spect_paths[0]
    if tp1_spect is None or not tp1_spect.exists():
        raise FileNotFoundError("First time point SPECT directory not found")

    # Find a DICOM file
    dcm_file = next(iter(sorted(tp1_spect.glob("*.dcm"))), None)
    if dcm_file is None:
        raise FileNotFoundError("No DICOM files found in first SPECT time point")

    ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True, force=True)

    # Defaults from study if radiopharm sequence is missing
    inj_date = getattr(ds, "StudyDate", None)
    inj_time = _extract_hhmmss(getattr(ds, "StudyTime", None))
    injected_activity: Optional[int] = None

    if hasattr(ds, "RadiopharmaceuticalInformationSequence"):
        try:
            rp = ds.RadiopharmaceuticalInformationSequence[0]
            # Try RadiopharmaceuticalStartDateTime first (combined date/time)
            rp_datetime = getattr(rp, "RadiopharmaceuticalStartDateTime", None)
            if rp_datetime:
                # Format: YYYYMMDDHHMMSS.FFFFFF or YYYYMMDDHHMMSS
                dt_str = str(rp_datetime)
                if len(dt_str) >= 14:
                    inj_date = dt_str[:8]  # YYYYMMDD
                    inj_time = dt_str[8:14]  # HHMMSS
            else:
                # Fall back to separate Date/Time fields
                inj_date = getattr(rp, "RadiopharmaceuticalStartDate", inj_date)
                inj_time = _extract_hhmmss(
                    getattr(rp, "RadiopharmaceuticalStartTime", inj_time)
                )

            dose = getattr(rp, "RadionuclideTotalDose", None)
            if dose is not None:
                try:
                    injected_activity = int(round(float(dose)))  # Bq
                except Exception:
                    pass
        except Exception:
            pass

    # Patient weight in grams
    weight_g: Optional[int] = None
    pw = getattr(ds, "PatientWeight", None)  # kg
    if pw is not None:
        try:
            weight_g = int(round(float(pw) * 1000.0))
        except Exception:
            pass

    return {
        "InjectionDate": inj_date or "",
        "InjectionTime": inj_time or "",
        "InjectedActivity": injected_activity,  # Bq (int) or None
        "PatientWeight_g": weight_g,
    }


def prepare_cycle_inputs(
    storage_root: str | Path, patient_id: str, cycle_no: int
) -> Tuple[
    List[Optional[Path]],
    List[Optional[Path]],
    List[Optional[Path]],
    Dict[str, Optional[str | int]],
]:
    """Prepare inputs for longitudinal processing for a given cycle.

    One-liner to load CT, SPECT, RTSTRUCT paths and injection metadata for dosimetry.

    Parameters
    ----------
    storage_root : str | Path
        Base directory where organized data lives.
    patient_id : str
        Patient identifier.
    cycle_no : int
        Cycle number (1-based).

    Returns
    -------
    tuple
        (ct_paths, spect_paths, rtstruct_files, injection_info) where injection_info
        is a dict with InjectionDate, InjectionTime, InjectedActivity, PatientWeight_g.
    """
    ct_paths, spect_paths, rtstruct_files = list_cycle_timepoints(
        storage_root, patient_id, cycle_no
    )
    inj = extract_injection_from_first_tp_spect(storage_root, patient_id, cycle_no)
    return ct_paths, spect_paths, rtstruct_files, inj
