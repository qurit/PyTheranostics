"""
Cycle data loading utilities for PyTheranostics.

Load organized DICOM data (CT, SPECT, RTSTRUCT) for dosimetry processing
from the folder structure produced by the DICOM receiver or manual organization.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pydicom
import SimpleITK

from pytheranostics.imaging_ds.longitudinal_study import LongitudinalStudy
from pytheranostics.imaging_tools.tools import load_and_resample_RT_to_target


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


# --- New high-level orchestration API ---------------------------------------------------------


def _canonical_mask_name(name: str) -> str:
    """Map RTSTRUCT ROI names to canonical pyTheranostics mask names.

    Best-effort normalization used for auto mapping. Keeps unknown names as-is.
    """
    # Strip modality suffixes often used in notebooks (e.g., _m for CT-based, _a for activity)
    base = name
    if base.endswith("_m") or base.endswith("_a"):
        base = base[:-2]

    # Common synonyms/abbreviations
    replacements = {
        "Kidney_L": "Kidney_Left",
        "Kidney_R": "Kidney_Right",
        "Parotid_L": "ParotidGland_Left",
        "Parotid_R": "ParotidGland_Right",
        "Submandibular_L": "SubmandibularGland_Left",
        "Submandibular_R": "SubmandibularGland_Right",
        "WBCT": "WholeBody",
    }
    return replacements.get(base, base)


def _build_auto_mapping(mask_keys: List[str]) -> Dict[str, str]:
    """Build a mapping dict from available mask keys to canonical names."""
    mapping: Dict[str, str] = {}
    for key in mask_keys:
        mapping[key] = _canonical_mask_name(key)
    return mapping


def create_studies_with_masks(
    storage_root: str | Path,
    patient_id: str,
    cycle_no: int,
    *,
    calibration_factor: Optional[float] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    auto_map: bool = False,
    ct_mask_mapping: Optional[Dict[str, str]] = None,
    spect_mask_mapping: Optional[Dict[str, str]] = None,
    mapping_config: Optional[Union[str, Path, Dict[str, Dict[str, str]]]] = None,
) -> Tuple[
    LongitudinalStudy,
    LongitudinalStudy,
    Dict[str, Optional[str | int]],
    Dict[int, Dict[str, str]],
]:
    """Create longitudinal studies and load/resample masks in one pass.

    This function reads all DICOM data once to build CT and SPECT longitudinal studies,
    extracts injection information, and loads + resamples RTSTRUCT masks for each timepoint.
    By default, masks are imported under their original ROI names (no validation or
    canonical mapping). You can optionally pass explicit mappings or enable auto_map
    to normalize ROI names at load time.

    Parameters
    ----------
    storage_root : str | Path
        Base directory where organized data lives.
    patient_id : str
        Patient identifier.
    cycle_no : int
        Cycle number (1-based).
    calibration_factor : float, optional
        Optional SPECT calibration factor (Bq per count/sec) applied during image load.
    parallel : bool
        Load each timepoint in parallel when possible, defaults True.
    max_workers : int, optional
        Max threads for loading, defaults to sensible number when None.
    auto_map : bool
        If True, infer and APPLY mask mappings from available ROI names (e.g., Kidney_L -> Kidney_Left).
        Defaults to False to import masks "as-is" and normalize later.
    ct_mask_mapping : dict, optional
        Explicit mapping for CT masks; overrides auto mapping if provided.
    spect_mask_mapping : dict, optional
        Explicit mapping for SPECT masks; overrides auto mapping if provided.
    mapping_config : str | Path | dict, optional
        Either:
        - Path to a JSON file containing 'ct_mappings' and 'spect_mappings' keys
        - A dictionary with 'ct_mappings' and 'spect_mappings' keys
        If provided, loads mappings from this config. Individual ct_mask_mapping and
        spect_mask_mapping parameters override keys from the config.

    Returns
    -------
    (longCT, longSPECT, injection_info, used_mappings)
        - longCT: LongitudinalStudy of CT timepoints
        - longSPECT: LongitudinalStudy of SPECT timepoints
        - injection_info: Dict with InjectionDate, InjectionTime, InjectedActivity, PatientWeight_g
        - used_mappings: Dict[time_id, mapping_summary] of the mapping applied per timepoint
    """
    # Load mappings from config if provided
    config_ct_mapping = None
    config_spect_mapping = None

    if mapping_config is not None:
        if isinstance(mapping_config, dict):
            # Direct dict provided
            config_ct_mapping = mapping_config.get("ct_mappings", {})
            config_spect_mapping = mapping_config.get("spect_mappings", {})
        else:
            # Path to JSON file
            loaded = LongitudinalStudy.load_mappings_from_json(mapping_config)
            config_ct_mapping = loaded.get("ct_mappings", {})
            config_spect_mapping = loaded.get("spect_mappings", {})

    # Individual parameters override config
    final_ct_mapping = (
        ct_mask_mapping if ct_mask_mapping is not None else config_ct_mapping
    )
    final_spect_mapping = (
        spect_mask_mapping if spect_mask_mapping is not None else config_spect_mapping
    )

    # 1) Discover paths and injection metadata
    ct_paths, spect_paths, rtstruct_files, inj = prepare_cycle_inputs(
        storage_root, patient_id, cycle_no
    )

    # 2) Build longitudinal studies from DICOM once
    ct_dirs = [str(p) for p in ct_paths if p is not None]
    spect_dirs = [str(p) for p in spect_paths if p is not None]

    longCT = LongitudinalStudy.from_dicom(
        dicom_dirs=ct_dirs,
        modality="CT",
        parallel=parallel,
        max_workers=max_workers,
    )
    longSPECT = LongitudinalStudy.from_dicom(
        dicom_dirs=spect_dirs,
        modality="Lu177_SPECT",
        calibration_factor=calibration_factor,
        parallel=parallel,
        max_workers=max_workers,
    )

    # 3) Load and resample masks per timepoint, add to both studies
    used_mappings: Dict[int, Dict[str, str]] = {}
    for time_id, rt_file in enumerate(rtstruct_files):
        ct_dir = ct_paths[time_id]
        if ct_dir is None or rt_file is None:
            continue

        # Use the SPECT image at this timepoint as target for NM masks
        target_img: Optional[SimpleITK.Image] = longSPECT.images.get(time_id)
        if target_img is None:
            # No SPECT for this timepoint; still allow CT masks
            # Build NM masks only if target available
            target_img = next(iter(longSPECT.images.values()), None)

        ct_masks, nm_masks = load_and_resample_RT_to_target(
            ref_dicom_ct_dir=str(ct_dir),
            rt_struct_file=str(rt_file),
            target_img=target_img,
        )

        # Decide whether to apply mappings now or import raw names
        apply_ct_mapping = (final_ct_mapping is not None) or auto_map
        apply_spect_mapping = (final_spect_mapping is not None) or auto_map

        if apply_ct_mapping:
            # Build mappings: explicit overrides auto
            ct_map = (
                final_ct_mapping
                if final_ct_mapping is not None
                else _build_auto_mapping(list(ct_masks.keys()))
            )
            longCT.add_masks_to_time_point(
                time_id=time_id, masks=ct_masks, mask_mapping=ct_map
            )
        else:
            # Import as-is
            longCT.add_raw_masks_to_time_point(time_id=time_id, masks=ct_masks)
            ct_map = {k: k for k in ct_masks.keys()}

        if target_img is not None:
            if apply_spect_mapping:
                spect_map = (
                    final_spect_mapping
                    if final_spect_mapping is not None
                    else _build_auto_mapping(list(nm_masks.keys()))
                )
                longSPECT.add_masks_to_time_point(
                    time_id=time_id, masks=nm_masks, mask_mapping=spect_map
                )
            else:
                longSPECT.add_raw_masks_to_time_point(time_id=time_id, masks=nm_masks)
                spect_map = {k: k for k in nm_masks.keys()}
        else:
            spect_map = {}

        used_mappings[time_id] = {**ct_map, **spect_map}

    return longCT, longSPECT, inj, used_mappings
