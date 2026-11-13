"""
Cycle data loading utilities for PyTheranostics.

Load organized DICOM data (CT, SPECT, RTSTRUCT) for dosimetry processing
from the folder structure produced by the DICOM receiver or manual organization.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    height_cm: Optional[int] = None
    pw = getattr(ds, "PatientSize", None)  # cm
    if pw is not None:
        try:
            height_cm = int(round(float(pw) * 100.0))
        except Exception:
            pass
    return {
        "InjectionDate": inj_date or "",
        "InjectionTime": inj_time or "",
        "InjectedActivity": injected_activity,  # Bq (int) or None
        "PatientWeight_g": weight_g,
        "PatientHeight_cm": height_cm,
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


def _get_canonical_mappings() -> Dict[str, str]:
    """Load canonical name mappings from config file.

    Searches for voi_mappings_config.json in order:
    1. Current directory (project-specific)
    2. One level up (project root)
    3. Package template (defaults)

    Returns
    -------
    Dict[str, str]
        Mapping of abbreviated/common names to canonical names.
        Returns empty dict if no config found.
    """
    search_paths = [
        Path.cwd() / "voi_mappings_config.json",
        Path.cwd().parent / "voi_mappings_config.json",
    ]

    for config_path in search_paths:
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    if "canonical_mappings" in config:
                        canon_config = config["canonical_mappings"]
                        if isinstance(canon_config, dict):
                            return canon_config.get("mappings", {})
            except Exception:
                continue

    # Try package template
    try:
        import importlib.resources as pkg_resources

        template_path = pkg_resources.files("pytheranostics.data").joinpath(
            "configuration_templates/voi_mappings_config.json"
        )
        with open(template_path, "r") as f:
            config = json.load(f)
            if "canonical_mappings" in config:
                canon_config = config["canonical_mappings"]
                if isinstance(canon_config, dict):
                    return canon_config.get("mappings", {})
    except Exception:
        pass

    return {}


def _canonical_mask_name(name: str) -> str:
    """Apply canonical name mappings from config.

    Best-effort normalization used for auto mapping. Keeps unknown names as-is.
    Mappings are loaded from voi_mappings_config.json.
    """
    # Strip modality suffixes often used in notebooks (e.g., _m for CT-based, _a for activity)
    base = name
    if base.endswith("_m") or base.endswith("_a"):
        base = base[:-2]

    # Load canonical mappings from config
    replacements = _get_canonical_mappings()
    return replacements.get(base, base)


def _build_auto_mapping(mask_keys: List[str]) -> Dict[str, str]:
    """Build a mapping dict from available mask keys to canonical names."""
    mapping: Dict[str, str] = {}
    for key in mask_keys:
        mapping[key] = _canonical_mask_name(key)
    return mapping


def create_studies_with_masks(
    storage_root: Optional[str | Path] = None,
    patient_id: Optional[str] = None,
    cycle_no: Optional[int] = None,
    *,
    study_info: Optional[Dict[str, Any]] = None,
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
    storage_root : str | Path, optional
        Base directory where organized data lives. Used with patient_id + cycle_no
        for the legacy folder-structured loading behavior.
    patient_id : str, optional
        Patient identifier used with storage_root.
    cycle_no : int, optional
        Cycle number (1-based), used with storage_root.
    study_info : dict, optional
        Study dictionary returned by auto_setup_dosimetry_study() or
        auto_setup_dosimetry_study_inventory(). If provided, CT/SPECT/RTSTRUCT
        paths are loaded from this dictionary and storage_root/patient_id/cycle_no
        are not required.
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
    if study_info is not None:
        tp_items = study_info.get("time_points", [])
        if tp_items:
            ct_paths = [
                Path(tp["ct_path"]) if tp.get("ct_path") else None for tp in tp_items
            ]
            spect_paths = [
                Path(tp["spect_path"]) if tp.get("spect_path") else None
                for tp in tp_items
            ]
            rtstruct_files = [
                Path(tp["rtstruct_file"]) if tp.get("rtstruct_file") else None
                for tp in tp_items
            ]
            first_tp_with_inj = next(
                (
                    tp
                    for tp in tp_items
                    if isinstance(tp.get("injection_info"), dict)
                    and tp.get("injection_info")
                ),
                None,
            )
            inj_info = (
                first_tp_with_inj.get("injection_info", {})
                if first_tp_with_inj is not None
                else {}
            )
        else:
            ct_paths = [Path(p) if p else None for p in study_info.get("ct_paths", [])]
            spect_paths = [
                Path(p) if p else None for p in study_info.get("spect_paths", [])
            ]
            rtstruct_files = [
                Path(p) if p else None for p in study_info.get("rtstruct_files", [])
            ]
            inj_info = {}

        max_len = max(len(ct_paths), len(spect_paths), len(rtstruct_files), 0)
        ct_paths.extend([None] * (max_len - len(ct_paths)))
        spect_paths.extend([None] * (max_len - len(spect_paths)))
        rtstruct_files.extend([None] * (max_len - len(rtstruct_files)))

        weight_g = inj_info.get("PatientWeight_g")
        if weight_g is None:
            weight_g = inj_info.get("patient_weight_g")
        if weight_g is None and inj_info.get("patient_weight_kg") is not None:
            try:
                weight_g = int(round(float(inj_info["patient_weight_kg"]) * 1000.0))
            except Exception:
                weight_g = None

        injected_activity = (
            inj_info.get("InjectedActivity")
            if "InjectedActivity" in inj_info
            else inj_info.get("injected_activity")
        )
        if injected_activity is not None:
            try:
                injected_activity = int(round(float(injected_activity)))
            except Exception:
                pass

        inj = {
            "InjectionDate": inj_info.get("InjectionDate")
            or inj_info.get("injection_date")
            or "",
            "InjectionTime": _extract_hhmmss(
                inj_info.get("InjectionTime") or inj_info.get("injection_time")
            )
            or "",
            "InjectedActivity": injected_activity,
            "PatientWeight_g": weight_g,
            "PatientHeight_cm": inj_info.get("PatientHeight_cm"),
        }
    else:
        if storage_root is None or patient_id is None or cycle_no is None:
            raise ValueError(
                "Either provide study_info, or provide storage_root + patient_id + cycle_no."
            )
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
    # Track mappings separately by study origin
    used_mappings: Dict[int, Dict[str, Dict[str, str]]] = {}
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

        if target_img is None:
            raise RuntimeError(
                f"No SPECT image available for timepoint {time_id} to resample NM masks"
            )

        ct_masks, nm_masks = load_and_resample_RT_to_target(
            ref_dicom_ct_dir=str(ct_dir),
            rt_struct_file=str(rt_file),
            target_img=target_img,
        )

        # Decide whether to apply mappings now or import raw names
        apply_ct_mapping = (final_ct_mapping is not None) or auto_map
        apply_spect_mapping = (final_spect_mapping is not None) or auto_map

        def _is_valid_target(name: str) -> bool:
            if name in LongitudinalStudy._get_valid_organ_names():
                return True
            return re.match(r"^Lesion_([1-9]\d*)$", name) is not None

        # --- CT masks
        if apply_ct_mapping:
            ct_map_valid: Dict[str, str] = {}
            ct_raw_keys: List[str] = []
            if final_ct_mapping is not None:
                for k in ct_masks.keys():
                    dst = final_ct_mapping.get(k)
                    if dst is not None and _is_valid_target(dst):
                        ct_map_valid[k] = dst
                    else:
                        ct_raw_keys.append(k)
            else:
                # auto map
                for k in ct_masks.keys():
                    dst = _canonical_mask_name(k)
                    if _is_valid_target(dst):
                        ct_map_valid[k] = dst
                    else:
                        ct_raw_keys.append(k)

            if ct_map_valid:
                longCT.add_masks_to_time_point(
                    time_id=time_id, masks=ct_masks, mask_mapping=ct_map_valid
                )
            if ct_raw_keys:
                longCT.add_raw_masks_to_time_point(
                    time_id=time_id,
                    masks={k: ct_masks[k] for k in ct_raw_keys},
                )
            # Track used mapping (identity for raw keys)
            ct_map = {
                **{k: v for k, v in ct_map_valid.items()},
                **{k: k for k in ct_raw_keys},
            }
        else:
            # Import as-is
            longCT.add_raw_masks_to_time_point(time_id=time_id, masks=ct_masks)
            ct_map = {k: k for k in ct_masks.keys()}

        if target_img is not None:
            if apply_spect_mapping:
                spect_map_valid: Dict[str, str] = {}
                spect_raw_keys: List[str] = []
                if final_spect_mapping is not None:
                    for k in nm_masks.keys():
                        dst = final_spect_mapping.get(k)

                        if dst is not None and _is_valid_target(dst):
                            spect_map_valid[k] = dst
                        else:
                            spect_raw_keys.append(k)
                else:
                    # auto map
                    for k in nm_masks.keys():
                        dst = _canonical_mask_name(k)
                        if _is_valid_target(dst):
                            spect_map_valid[k] = dst
                        else:
                            spect_raw_keys.append(k)

                if spect_map_valid:
                    longSPECT.add_masks_to_time_point(
                        time_id=time_id, masks=nm_masks, mask_mapping=spect_map_valid
                    )
                if spect_raw_keys:
                    longSPECT.add_raw_masks_to_time_point(
                        time_id=time_id,
                        masks={k: nm_masks[k] for k in spect_raw_keys},
                    )
                # Track used mapping (identity for raw keys)
                spect_map = {
                    **{k: v for k, v in spect_map_valid.items()},
                    **{k: k for k in spect_raw_keys},
                }
            else:
                longSPECT.add_raw_masks_to_time_point(time_id=time_id, masks=nm_masks)
                spect_map = {k: k for k in nm_masks.keys()}
        else:
            spect_map = {}

        # Store with study origin labels
        used_mappings[time_id] = {"ct": ct_map, "spect": spect_map}

    return longCT, longSPECT, inj, used_mappings
