"""Utilities for organizing DICOM files by patient, cycle, and timepoint."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pydicom

logger = logging.getLogger(__name__)


def _parse_dt(date_str: Optional[str], time_str: Optional[str]) -> Optional[datetime]:
    """Parse common DICOM date/time fields to a datetime object.

    Parameters
    ----------
    date_str : str | None
        DICOM DA (YYYYMMDD)
    time_str : str | None
        DICOM TM (HHMMSS.frac)

    Returns
    -------
    datetime | None
        Parsed datetime or None if not enough info
    """
    if not date_str:
        return None
    try:
        y = int(date_str[0:4])
        m = int(date_str[4:6])
        d = int(date_str[6:8])
        if time_str:
            hh = int(time_str[0:2]) if len(time_str) >= 2 else 0
            mm = int(time_str[2:4]) if len(time_str) >= 4 else 0
            ss = int(time_str[4:6]) if len(time_str) >= 6 else 0
            micro = 0
            if len(time_str) > 7 and "." in time_str:
                frac = time_str.split(".")[-1]
                # pad/cut to microseconds
                frac = (frac + "000000")[:6]
                micro = int(frac)
            return datetime(y, m, d, hh, mm, ss, micro)
        return datetime(y, m, d)
    except Exception:
        return None


def _series_datetime_from_any(dcm: pydicom.Dataset) -> Optional[datetime]:
    """Best-effort extraction of a datetime for a DICOM series instance.

    Tries AcquisitionDate/Time first (most accurate), then SeriesDate/Time,
    then ContentDate/Time, finally falls back to StudyDate/Time.
    """
    # Acquisition (most accurate for actual scan time)
    dt = _parse_dt(
        getattr(dcm, "AcquisitionDate", None), getattr(dcm, "AcquisitionTime", None)
    )
    if dt:
        return dt
    # Series
    dt = _parse_dt(getattr(dcm, "SeriesDate", None), getattr(dcm, "SeriesTime", None))
    if dt:
        return dt
    # Content
    dt = _parse_dt(getattr(dcm, "ContentDate", None), getattr(dcm, "ContentTime", None))
    if dt:
        return dt
    # Study
    return _parse_dt(getattr(dcm, "StudyDate", None), getattr(dcm, "StudyTime", None))


def organize_folder_by_cycles(
    storage_root: Path | str,
    output_base: Path | str | None = None,
    *,
    cycle_gap_days: float = 15,
    timepoint_separation_days: float = 1,
    move: bool = True,
    patient_id_filter: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, List[Path]]]]:
    """Organize a folder of DICOM files into Patient/Cycle/Timepoint structure.

    This scans ``storage_root`` recursively for ``*.dcm`` files, groups them by
    PatientID and StudyDate, then creates folders like:

        PatientID/Cycle1/tp1/CT
        PatientID/Cycle1/tp1/SPECT
        PatientID/Cycle1/tp1/CT/RTstruct

    Behavior mirrors ``organize_by_cycles()`` but does not require
    a running receiver nor a metadata file; grouping is inferred from DICOM tags.

    Parameters
    ----------
    storage_root : Path | str
        Root directory to scan for DICOM files (searched recursively).
    output_base : Path | str | None
        Base directory where organized output will be created. Defaults to
        ``storage_root`` when None.
    cycle_gap_days : float
        New cycle if consecutive study dates differ by >= this many days.
    timepoint_separation_days : float
        New timepoint when datetime gap is >= this many days (can be fractional, e.g., 0.2 ≈ 4.8 hours).
    move : bool
        If True, move files (and prune emptied dirs opportunistically). If False, copy files.
    patient_id_filter : list[str] | None
        If provided, only organize these PatientIDs.

    Returns
    -------
    dict
        Mapping: {PatientID: {"CycleX": {"tpY": [Path, ...]}}}
    """
    storage_root = Path(storage_root)
    if output_base is None:
        output_base = storage_root
    output_base = Path(output_base)

    index: Dict[str, Dict[tuple, List[Path]]] = {}
    rep_dt_by_series: Dict[str, Dict[tuple, List[datetime]]] = {}

    def _read_minimal(dcm_path: Path) -> Optional[pydicom.Dataset]:
        try:
            return pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
        except Exception:
            return None

    candidates: set[Path] = set()
    for pattern in ("*.dcm", "*.DCM"):
        candidates.update(storage_root.rglob(pattern))

    for dcm_file in sorted(candidates):
        ds = _read_minimal(dcm_file)
        if ds is None:
            continue

        patient_id = getattr(ds, "PatientID", None) or "UNKNOWN"
        if patient_id_filter and patient_id not in patient_id_filter:
            continue

        dt = _series_datetime_from_any(ds)
        study_date = getattr(ds, "StudyDate", None)
        if dt is None:
            # Last-resort fallback to file modification time to help split same-day scans
            try:
                dt = datetime.fromtimestamp(dcm_file.stat().st_mtime)
            except Exception:
                dt = None

        if not study_date:
            if dt:
                study_date = dt.strftime("%Y%m%d")
            else:
                study_date = "00000000"

        modality = getattr(ds, "Modality", None) or "UNKNOWN"
        series_number = getattr(ds, "SeriesNumber", None)
        try:
            series_number = int(series_number) if series_number is not None else -1
        except Exception:
            series_number = -1

        key = (study_date, modality, series_number)
        index.setdefault(patient_id, {}).setdefault(key, []).append(dcm_file)

        if dt is None:
            try:
                dt = datetime.strptime(study_date, "%Y%m%d")
            except Exception:
                dt = datetime(1900, 1, 1)

        rep_dt_by_series.setdefault(patient_id, {}).setdefault(key, []).append(dt)

    results: Dict[str, Dict[str, Dict[str, List[Path]]]] = {}

    for patient_id, series_map in index.items():
        series_entries: List[Dict[str, object]] = []
        for key, files in series_map.items():
            study_date, modality, series_number = key
            dt_list = rep_dt_by_series.get(patient_id, {}).get(key, [])
            rep_dt = min(dt_list) if dt_list else datetime(1900, 1, 1)

            # Sub-group files within this series by datetime gaps to split same series_number across timepoints
            # Sort files by their datetime and split into subgroups when gap >= timepoint_separation_days
            file_dts: List[tuple[Path, datetime]] = []
            for f in files:
                try:
                    ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                    fdt = _series_datetime_from_any(ds)
                    if fdt is None:
                        try:
                            fdt = datetime.fromtimestamp(f.stat().st_mtime)
                        except Exception:
                            fdt = datetime.strptime(study_date, "%Y%m%d")
                except Exception:
                    try:
                        fdt = datetime.fromtimestamp(f.stat().st_mtime)
                    except Exception:
                        fdt = datetime.strptime(study_date, "%Y%m%d")
                file_dts.append((f, fdt))

            file_dts = sorted(file_dts, key=lambda x: x[1])

            # Split into subgroups when gap >= timepoint_separation_days
            subgroups: List[List[Path]] = []
            current_group: List[Path] = []
            prev_dt: Optional[datetime] = None
            for f, fdt in file_dts:
                if prev_dt is not None and (fdt - prev_dt) >= timedelta(
                    days=timepoint_separation_days
                ):
                    # Start new subgroup
                    subgroups.append(current_group)
                    current_group = [f]
                else:
                    current_group.append(f)
                prev_dt = fdt

            if current_group:
                subgroups.append(current_group)

            # Create a series_entry per subgroup
            for sg_idx, sg_files in enumerate(subgroups):
                sg_dts = [fdt for f, fdt in file_dts if f in sg_files]
                sg_rep_dt = min(sg_dts) if sg_dts else rep_dt
                series_entries.append(
                    {
                        "study_date": study_date,
                        "datetime": sg_rep_dt,
                        "modality": modality,
                        "series_number": series_number,
                        "files": sg_files,
                    }
                )

        series_entries = sorted(series_entries, key=lambda s: s["datetime"])
        if not series_entries:
            continue

        patient_root = output_base / patient_id
        patient_root.mkdir(parents=True, exist_ok=True)

        cycle_idx = 1
        tp_idx = 1
        prev_dt = series_entries[0]["datetime"]
        src_dirs_for_cleanup: set[Path] = set()

        for i, s in enumerate(series_entries):
            this_dt = s["datetime"]
            if i > 0:
                if (this_dt - prev_dt) >= timedelta(days=cycle_gap_days):
                    cycle_idx += 1
                    tp_idx = 1
                elif (this_dt - prev_dt) >= timedelta(days=timepoint_separation_days):
                    tp_idx += 1

            cycle_dir = patient_root / f"Cycle{cycle_idx}" / f"tp{tp_idx}"
            cycle_dir.mkdir(parents=True, exist_ok=True)

            cycle_key = f"Cycle{cycle_idx}"
            tp_key = f"tp{tp_idx}"
            results.setdefault(patient_id, {}).setdefault(cycle_key, {}).setdefault(
                tp_key, []
            )

            modality = s["modality"]
            series_number = s["series_number"]
            files = s["files"]

            if modality in ["NM", "PT"]:
                dest_dir = cycle_dir / "SPECT"
            elif modality == "RTSTRUCT":
                dest_dir = cycle_dir / "CT" / "RTstruct"
            else:
                dest_dir = cycle_dir / (modality or "UNKNOWN")

            dest_dir.mkdir(parents=True, exist_ok=True)

            imported = 0
            for src_file in files:
                try:
                    target = dest_dir / src_file.name
                    if target.exists():
                        continue
                    if move:
                        import shutil

                        shutil.move(str(src_file), str(target))
                        src_dirs_for_cleanup.add(src_file.parent)
                    else:
                        import shutil

                        shutil.copy2(str(src_file), str(target))
                    imported += 1
                except Exception:
                    continue

            logger.info(
                f"Organized {imported} files -> {dest_dir} ({modality}, Series{series_number}, {this_dt})"
            )

            results[patient_id][cycle_key][tp_key].append(dest_dir)

            prev_dt = this_dt

        if move:
            for src_dir in list(src_dirs_for_cleanup):
                try:
                    if src_dir.exists() and not any(src_dir.iterdir()):
                        src_dir.rmdir()
                except Exception:
                    pass
                try:
                    parent1 = src_dir.parent
                    if parent1.exists() and not any(parent1.iterdir()):
                        parent1.rmdir()
                except Exception:
                    pass

    return results


def summarize_timepoints(
    storage_root: Path | str,
    *,
    patient_id_filter: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, object]]]:
    """Summarize detected series (one per modality/series_number per StudyDate) for debugging.

    Scans ``storage_root`` with the same datetime extraction logic used by
    ``organize_folder_by_cycles`` and returns, per patient, the ordered list of
    all distinct (study_date, modality, series_number) with their representative
    datetimes and gaps in hours to the previous entry.
    """
    storage_root = Path(storage_root)

    # Track all unique (study_date, modality, series_number) per patient with min datetime
    index: Dict[str, Dict[tuple, List[datetime]]] = {}

    def _read_minimal(dcm_path: Path) -> Optional[pydicom.Dataset]:
        try:
            return pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
        except Exception:
            return None

    candidates: set[Path] = set()
    for pattern in ("*.dcm", "*.DCM"):
        candidates.update(storage_root.rglob(pattern))

    for dcm_file in sorted(candidates):
        ds = _read_minimal(dcm_file)
        if ds is None:
            continue
        patient_id = getattr(ds, "PatientID", None) or "UNKNOWN"
        if patient_id_filter and patient_id not in patient_id_filter:
            continue

        dt = _series_datetime_from_any(ds)
        study_date = getattr(ds, "StudyDate", None)
        if dt is None:
            try:
                dt = datetime.fromtimestamp(dcm_file.stat().st_mtime)
            except Exception:
                dt = None

        if not study_date:
            study_date = dt.strftime("%Y%m%d") if dt else "00000000"

        modality = getattr(ds, "Modality", None) or "UNKNOWN"
        series_number = getattr(ds, "SeriesNumber", None)
        try:
            series_number = int(series_number) if series_number is not None else -1
        except Exception:
            series_number = -1

        if dt is None:
            try:
                dt = datetime.strptime(study_date, "%Y%m%d")
            except Exception:
                dt = datetime(1900, 1, 1)

        key = (study_date, modality, series_number)
        index.setdefault(patient_id, {}).setdefault(key, []).append(dt)

    summary: Dict[str, List[Dict[str, object]]] = {}
    for patient_id, by_key in index.items():
        entries: List[Dict[str, object]] = []
        for key, dts in sorted(by_key.items(), key=lambda kv: min(kv[1])):
            sd, mod, sn = key
            rep_dt = min(dts)
            entries.append(
                {
                    "study_date": sd,
                    "modality": mod,
                    "series_number": sn,
                    "datetime": rep_dt,
                }
            )

        # Compute deltas after sorting by datetime
        entries = sorted(entries, key=lambda e: e["datetime"])
        prev_dt: Optional[datetime] = None
        for entry in entries:
            delta_hours = None
            if prev_dt is not None:
                delta_hours = (entry["datetime"] - prev_dt).total_seconds() / 3600.0
            entry["delta_hours"] = delta_hours
            prev_dt = entry["datetime"]

        summary[patient_id] = entries

    return summary
