"""
DICOM ingestion utilities for PyTheranostics.

Simplifies data ingestion for dosimetry workflows.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pydicom
from pydicom.misc import is_dicom

logger = logging.getLogger(__name__)


def _split_dicom_datetime(dt_val: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Split a DICOM DT value into date (YYYYMMDD) and time (HHMMSS)."""
    if not dt_val:
        return None, None
    val = str(dt_val).split(".")[0]
    if len(val) < 8:
        return None, None
    date_part = val[:8]
    time_part = val[8:14] if len(val) > 8 else None
    if time_part:
        time_part = time_part.ljust(6, "0")
    return date_part, time_part


class DosimetryStudyOrganizer:
    """
    Organize DICOM studies for dosimetry analysis.

    Automatically handles multiple time points, extracts metadata, and structures files.
    """

    def __init__(self, base_dir: Path):
        """
        Initialize the study organizer.

        Parameters
        ----------
        base_dir : Path
            Base directory containing DICOM files
        """
        self.base_dir = Path(base_dir)
        self.patient_info = {}
        self.time_points = []

    def scan_directory(self) -> Dict:
        """
        Scan directory structure and identify time points automatically.

        Returns
        -------
        dict
            Dictionary with organized study information
        """
        study_info = {
            "patient_id": None,
            "time_points": [],
            "ct_paths": [],
            "spect_paths": [],
            "rtstruct_files": [],
            "injection_info": {},
        }

        # Look for time point directories (tp1, tp2, etc.) or organize by series time
        tp_dirs = sorted(self.base_dir.glob("tp*"))

        if not tp_dirs:
            # Try to auto-detect time points from DICOM metadata
            logger.info("No tp* directories found, attempting auto-detection")
            tp_dirs = self._auto_detect_time_points()

        for tp_dir in tp_dirs:
            tp_info = self._process_time_point(tp_dir)
            if tp_info:
                study_info["time_points"].append(tp_info)
                study_info["ct_paths"].append(tp_info.get("ct_path"))
                study_info["spect_paths"].append(tp_info.get("spect_path"))

                if tp_info.get("rtstruct_file"):
                    study_info["rtstruct_files"].append(tp_info["rtstruct_file"])

        # Extract patient info from first available DICOM
        if study_info["time_points"]:
            first_tp = study_info["time_points"][0]
            study_info["patient_id"] = first_tp.get("patient_id")
            study_info["injection_info"] = first_tp.get("injection_info", {})

        return study_info

    def _process_time_point(self, tp_dir: Path) -> Optional[Dict]:
        """
        Process a single time point directory.

        Parameters
        ----------
        tp_dir : Path
            Time point directory

        Returns
        -------
        dict or None
            Time point information
        """
        tp_info = {
            "name": tp_dir.name,
            "path": tp_dir,
            "ct_path": None,
            "spect_path": None,
            "rtstruct_file": None,
            "patient_id": None,
            "study_date": None,
            "injection_info": {},
        }

        # Look for CT directory
        ct_dir = tp_dir / "CT"
        if ct_dir.exists():
            tp_info["ct_path"] = str(ct_dir)

            # Look for RT struct
            rtstruct_dir = ct_dir / "RTstruct"
            if rtstruct_dir.exists():
                rtstruct_files = list(rtstruct_dir.glob("*.dcm"))
                if rtstruct_files:
                    tp_info["rtstruct_file"] = str(rtstruct_files[0])

            # Extract patient info from CT
            ct_files = list(ct_dir.glob("*.dcm"))
            if ct_files:
                try:
                    ds = pydicom.dcmread(ct_files[0], stop_before_pixels=True)
                    tp_info["patient_id"] = getattr(ds, "PatientID", None)
                    tp_info["study_date"] = getattr(ds, "StudyDate", None)
                except Exception as e:
                    logger.warning(f"Could not read DICOM metadata: {e}")

        # Look for SPECT/NM directory
        spect_dir = tp_dir / "SPECT"
        if spect_dir.exists():
            tp_info["spect_path"] = str(spect_dir)

            # Extract injection information from SPECT
            spect_files = list(spect_dir.glob("*.dcm"))
            if spect_files:
                try:
                    ds = pydicom.dcmread(spect_files[0], stop_before_pixels=True)
                    tp_info["injection_info"] = self._extract_injection_info(ds)
                except Exception as e:
                    logger.warning(f"Could not extract injection info: {e}")

        return tp_info if (tp_info["ct_path"] or tp_info["spect_path"]) else None

    def _extract_injection_info(self, ds: pydicom.Dataset) -> Dict:
        """
        Extract injection information from a DICOM dataset.

        Parameters
        ----------
        ds : pydicom.Dataset
            DICOM dataset (typically SPECT/NM)

        Returns
        -------
        dict
            Injection information
        """
        info = {
            "patient_weight_kg": getattr(ds, "PatientWeight", None),
            "injection_date": None,
            "injection_time": None,
            "injected_activity": None,
            "radiopharmaceutical": None,
        }

        # Convert patient weight to grams
        if info["patient_weight_kg"]:
            info["patient_weight_g"] = int(info["patient_weight_kg"] * 1000)

        # Extract from RadiopharmaceuticalInformationSequence
        if hasattr(ds, "RadiopharmaceuticalInformationSequence"):
            rp_seq = ds.RadiopharmaceuticalInformationSequence
            if len(rp_seq) > 0:
                rp_info = rp_seq[0]
                info["radiopharmaceutical"] = getattr(
                    rp_info, "Radiopharmaceutical", None
                )
                info["injected_activity"] = getattr(
                    rp_info, "RadionuclideTotalDose", None
                )

                inj_dt = getattr(rp_info, "RadiopharmaceuticalStartDateTime", None)
                inj_date, inj_time = _split_dicom_datetime(inj_dt)
                if inj_date:
                    info["injection_date"] = inj_date
                if inj_time:
                    info["injection_time"] = inj_time

                # Backward-compatible fallback when DateTime is not available.
                if not info["injection_date"]:
                    inj_date_legacy = getattr(
                        rp_info, "RadiopharmaceuticalStartDate", None
                    )
                    if inj_date_legacy:
                        info["injection_date"] = inj_date_legacy
                if not info["injection_time"]:
                    inj_time_legacy = getattr(
                        rp_info, "RadiopharmaceuticalStartTime", None
                    )
                    if inj_time_legacy:
                        info["injection_time"] = inj_time_legacy.split(".")[0]

        return info

    def _auto_detect_time_points(self) -> List[Path]:
        """
        Auto-detect time points from a flat directory structure.

        Returns
        -------
        list of Path
            Detected time point directories
        """
        # This would implement logic to group DICOM files by acquisition time
        # and create virtual time points
        logger.warning(
            "Auto-detection not fully implemented. Please use tp1, tp2, etc. structure"
        )
        return []

    def cleanup_nested_folders(self, directory: Path):
        """
        Clean up nested folder structures (removes single-child folders).

        Parameters
        ----------
        directory : Path
            Directory to clean up
        """
        import shutil

        for subfolder in directory.iterdir():
            if subfolder.is_dir():
                inner_subfolders = list(subfolder.iterdir())
                if len(inner_subfolders) == 1 and inner_subfolders[0].is_dir():
                    # Move contents up one level
                    for item in inner_subfolders[0].iterdir():
                        if item.exists():
                            shutil.move(str(item), str(directory))

                    # Remove empty nested folders
                    if inner_subfolders[0].exists():
                        shutil.rmtree(str(inner_subfolders[0]))
                    if subfolder.exists():
                        shutil.rmtree(str(subfolder))


def auto_setup_dosimetry_study(
    base_dir: Path, patient_id: Optional[str] = None, cleanup: bool = True
) -> Tuple[Dict, List[str], List[str], List[str]]:
    """
    Automatically set up a dosimetry study from a directory of DICOM files.

    Parameters
    ----------
    base_dir : Path
        Base directory containing the study data
    patient_id : str, optional
        Patient ID (if None, will be extracted from DICOM)
    cleanup : bool
        Whether to clean up nested folder structures

    Returns
    -------
    tuple
        (study_info, ct_paths, spect_paths, rtstruct_files)
    """
    organizer = DosimetryStudyOrganizer(base_dir)

    # Clean up if requested
    if cleanup:
        for tp_dir in base_dir.glob("tp*"):
            if (tp_dir / "CT").exists():
                organizer.cleanup_nested_folders(tp_dir / "CT")
                if (tp_dir / "CT" / "RTstruct").exists():
                    organizer.cleanup_nested_folders(tp_dir / "CT" / "RTstruct")
            if (tp_dir / "SPECT").exists():
                organizer.cleanup_nested_folders(tp_dir / "SPECT")

    # Scan and organize
    study_info = organizer.scan_directory()

    # Override patient_id if provided
    if patient_id:
        study_info["patient_id"] = patient_id

    return (
        study_info,
        study_info["ct_paths"],
        study_info["spect_paths"],
        study_info["rtstruct_files"],
    )


def auto_setup_dosimetry_study_inventory(
    base_dir: Path, patient_id: Optional[str] = None
) -> Tuple[Dict[str, Any], List[str], List[str], List[str]]:
    """
    Build a DICOM inventory of SPECT (NM/OT/PT), CT, and RTSTRUCT series.

    Parameters
    ----------
    base_dir : Path
        Base directory containing DICOM files.
    patient_id : str, optional
        Patient ID override. If None, the first detected PatientID is used.

    Returns
    -------
    tuple
        (study_info, ct_paths, spect_paths, rtstruct_files)
    """

    def _iter_dicom_files(root: Path) -> Iterable[Path]:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".dcm" or is_dicom(str(path)):
                yield path

    def _parse_datetime(
        date_str: Optional[str], time_str: Optional[str]
    ) -> Optional[datetime]:
        if not date_str:
            return None
        time_val = (time_str or "000000").split(".")[0]
        if len(time_val) < 6:
            time_val = time_val.ljust(6, "0")
        try:
            return datetime.strptime(f"{date_str}{time_val}", "%Y%m%d%H%M%S")
        except ValueError:
            return None

    def _dicom_datetime(ds: pydicom.Dataset) -> Optional[datetime]:
        for date_key, time_key in [
            ("AcquisitionDate", "AcquisitionTime"),
            ("SeriesDate", "SeriesTime"),
            ("ContentDate", "ContentTime"),
            ("StudyDate", "StudyTime"),
        ]:
            date_val = getattr(ds, date_key, None)
            time_val = getattr(ds, time_key, None)
            dt = _parse_datetime(date_val, time_val)
            if dt:
                return dt
        return None

    def _common_parent(files: List[Path]) -> Optional[str]:
        if not files:
            return None
        common = Path(os.path.commonpath([str(p) for p in files]))
        if common.is_file():
            return str(common.parent)
        return str(common)

    def _select_best_match(
        target: Dict[str, Any], candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        target_study = target.get("study_uid")
        filtered = [c for c in candidates if c.get("study_uid") == target_study]
        if filtered:
            candidates = filtered
        target_dt = target.get("series_datetime")
        dated = [c for c in candidates if c.get("series_datetime")]
        if target_dt and dated:
            return min(
                dated,
                key=lambda c: abs((c["series_datetime"] - target_dt).total_seconds()),
            )
        return candidates[0]

    def _extract_injection_info(ds: pydicom.Dataset) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "patient_weight_kg": getattr(ds, "PatientWeight", None),
            "injection_date": None,
            "injection_time": None,
            "injected_activity": None,
            "radiopharmaceutical": None,
        }

        if info["patient_weight_kg"]:
            info["patient_weight_g"] = int(info["patient_weight_kg"] * 1000)

        if hasattr(ds, "RadiopharmaceuticalInformationSequence"):
            rp_seq = ds.RadiopharmaceuticalInformationSequence
            if len(rp_seq) > 0:
                rp_info = rp_seq[0]
                info["radiopharmaceutical"] = getattr(
                    rp_info, "Radiopharmaceutical", None
                )
                info["injected_activity"] = getattr(
                    rp_info, "RadionuclideTotalDose", None
                )

                inj_dt = getattr(rp_info, "RadiopharmaceuticalStartDateTime", None)
                inj_date, inj_time = _split_dicom_datetime(inj_dt)
                if inj_date:
                    info["injection_date"] = inj_date
                if inj_time:
                    info["injection_time"] = inj_time

                # Backward-compatible fallback when DateTime is not available.
                if not info["injection_date"]:
                    inj_date_legacy = getattr(
                        rp_info, "RadiopharmaceuticalStartDate", None
                    )
                    if inj_date_legacy:
                        info["injection_date"] = inj_date_legacy
                if not info["injection_time"]:
                    inj_time_legacy = getattr(
                        rp_info, "RadiopharmaceuticalStartTime", None
                    )
                    if inj_time_legacy:
                        info["injection_time"] = inj_time_legacy.split(".")[0]

        return info

    series_map: Dict[str, Dict[str, Any]] = {}
    detected_patient_id: Optional[str] = None

    tag_list = [
        # Core indexing/matching fields
        "Modality",
        "SeriesInstanceUID",
        "StudyInstanceUID",
        "SOPClassUID",
        "SOPInstanceUID",
        "StudyDate",
        "StudyTime",
        "SeriesDate",
        "SeriesTime",
        "AcquisitionDate",
        "AcquisitionTime",
        "ContentDate",
        "ContentTime",
        "PatientID",
        "PatientName",
        "PatientSex",
        "PatientBirthDate",
        "PatientWeight",
        # Common geometric/acquisition fields
        "FrameOfReferenceUID",
        "ImagePositionPatient",
        "ImageOrientationPatient",
        "PixelSpacing",
        "SliceThickness",
        "KVP",
        "ConvolutionKernel",
        "RescaleIntercept",
        "RescaleSlope",
        # NM/PT quantitative + injection fields
        "SeriesDescription",
        "AcquisitionDuration",
        "Units",
        "DecayCorrection",
        "CorrectedImage",
        "CountsSource",
        "NumberOfSlices",
        "RadiopharmaceuticalInformationSequence",
        "RadiopharmaceuticalStartDateTime",
        # RTSTRUCT linkage/content fields
        "StructureSetLabel",
        "StructureSetDate",
        "StructureSetTime",
        "ReferencedFrameOfReferenceSequence",
        "StructureSetROISequence",
        "ROIContourSequence",
        "RTROIObservationsSequence",
    ]

    for dcm_path in _iter_dicom_files(Path(base_dir)):
        try:
            ds = pydicom.dcmread(
                dcm_path, stop_before_pixels=True, force=True, specific_tags=tag_list
            )
        except Exception as exc:
            logger.warning(f"Skipping unreadable DICOM: {dcm_path} ({exc})")
            continue

        modality = getattr(ds, "Modality", None)
        if modality not in {"CT", "RTSTRUCT", "NM", "OT", "PT"}:
            continue

        series_uid = getattr(ds, "SeriesInstanceUID", None)
        study_uid = getattr(ds, "StudyInstanceUID", None)
        series_key = series_uid or f"{study_uid}|{modality}|{dcm_path.parent}"

        if series_key not in series_map:
            series_map[series_key] = {
                "modality": modality,
                "study_uid": study_uid,
                "series_uid": series_uid,
                "files": [],
                "series_datetime": _dicom_datetime(ds),
                "patient_id": getattr(ds, "PatientID", None),
                "representative_ds": ds,
            }

        series_map[series_key]["files"].append(dcm_path)

        if not detected_patient_id:
            detected_patient_id = getattr(ds, "PatientID", None)

    spect_series = [
        s for s in series_map.values() if s["modality"] in {"NM", "OT", "PT"}
    ]
    ct_series = [s for s in series_map.values() if s["modality"] == "CT"]
    rt_series = [s for s in series_map.values() if s["modality"] == "RTSTRUCT"]

    logger.info(
        "DICOM inventory: %d SPECT series, %d CT series, %d RTSTRUCT series",
        len(spect_series),
        len(ct_series),
        len(rt_series),
    )

    for series in spect_series:
        logger.info(
            "SPECT series: study_uid=%s series_uid=%s datetime=%s files=%d root=%s",
            series.get("study_uid"),
            series.get("series_uid"),
            series.get("series_datetime"),
            len(series.get("files", [])),
            _common_parent(series.get("files", [])) or "UNKNOWN",
        )

    for series in ct_series:
        logger.info(
            "CT series: study_uid=%s series_uid=%s datetime=%s files=%d root=%s",
            series.get("study_uid"),
            series.get("series_uid"),
            series.get("series_datetime"),
            len(series.get("files", [])),
            _common_parent(series.get("files", [])) or "UNKNOWN",
        )

    for series in rt_series:
        logger.info(
            "RTSTRUCT series: study_uid=%s series_uid=%s datetime=%s files=%d root=%s",
            series.get("study_uid"),
            series.get("series_uid"),
            series.get("series_datetime"),
            len(series.get("files", [])),
            _common_parent(series.get("files", [])) or "UNKNOWN",
        )

    time_points: List[Dict[str, Any]] = []
    ct_paths: List[str] = []
    spect_paths: List[str] = []
    rtstruct_files: List[str] = []

    def _spect_sort_key(series: Dict[str, Any]) -> Tuple[int, datetime]:
        dt_val = series.get("series_datetime")
        if dt_val:
            return (0, dt_val)
        return (1, datetime.max)

    for idx, spect in enumerate(sorted(spect_series, key=_spect_sort_key)):
        ct_match = _select_best_match(spect, ct_series)
        rt_match = _select_best_match(spect, rt_series)

        spect_path = _common_parent(spect["files"])
        ct_path = _common_parent(ct_match["files"]) if ct_match else None
        rt_file = str(rt_match["files"][0]) if rt_match and rt_match["files"] else None

        logger.info(
            "Time point %s: SPECT=%s CT=%s RTSTRUCT=%s",
            f"tp{idx + 1}",
            spect_path or "NONE",
            ct_path or "NONE",
            rt_file or "NONE",
        )
        if ct_match:
            logger.info(
                "Time point %s CT match: study_uid=%s series_uid=%s datetime=%s",
                f"tp{idx + 1}",
                ct_match.get("study_uid"),
                ct_match.get("series_uid"),
                ct_match.get("series_datetime"),
            )
        if rt_match:
            logger.info(
                "Time point %s RTSTRUCT match: study_uid=%s series_uid=%s datetime=%s",
                f"tp{idx + 1}",
                rt_match.get("study_uid"),
                rt_match.get("series_uid"),
                rt_match.get("series_datetime"),
            )

        tp_info: Dict[str, Any] = {
            "name": f"tp{idx + 1}",
            "spect_path": spect_path,
            "ct_path": ct_path,
            "rtstruct_file": rt_file,
            "series_datetime": spect.get("series_datetime"),
            "study_uid": spect.get("study_uid"),
            "series_uid": spect.get("series_uid"),
            "patient_id": spect.get("patient_id"),
            "injection_info": (
                _extract_injection_info(spect["representative_ds"])
                if spect.get("representative_ds")
                else {}
            ),
        }

        time_points.append(tp_info)
        if ct_path:
            ct_paths.append(ct_path)
        if spect_path:
            spect_paths.append(spect_path)
        if rt_file:
            rtstruct_files.append(rt_file)

    study_info: Dict[str, Any] = {
        "patient_id": patient_id or detected_patient_id,
        "time_points": time_points,
        "ct_paths": ct_paths,
        "spect_paths": spect_paths,
        "rtstruct_files": rtstruct_files,
    }

    return study_info, ct_paths, spect_paths, rtstruct_files


def extract_patient_metadata(dicom_dir: Path) -> Dict:
    """
    Extract patient metadata from the first DICOM file in a directory.

    Parameters
    ----------
    dicom_dir : Path
        Directory containing DICOM files

    Returns
    -------
    dict
        Patient metadata
    """
    dcm_files = list(Path(dicom_dir).glob("**/*.dcm"))

    if not dcm_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)

    metadata = {
        "patient_id": getattr(ds, "PatientID", "UNKNOWN"),
        "patient_name": str(getattr(ds, "PatientName", "UNKNOWN")),
        "patient_weight_kg": getattr(ds, "PatientWeight", None),
        "study_date": getattr(ds, "StudyDate", None),
        "study_time": getattr(ds, "StudyTime", None),
    }

    # Try to extract injection info for NM modality
    if getattr(ds, "Modality", "") in ["NM", "PT"]:
        if hasattr(ds, "RadiopharmaceuticalInformationSequence"):
            rp_seq = ds.RadiopharmaceuticalInformationSequence
            if len(rp_seq) > 0:
                rp_info = rp_seq[0]
                metadata["injected_activity"] = getattr(
                    rp_info, "RadionuclideTotalDose", None
                )
                inj_dt = getattr(rp_info, "RadiopharmaceuticalStartDateTime", None)
                inj_date, inj_time = _split_dicom_datetime(inj_dt)
                metadata["injection_date"] = inj_date
                metadata["injection_time"] = inj_time

                if not metadata["injection_date"]:
                    metadata["injection_date"] = getattr(
                        rp_info, "RadiopharmaceuticalStartDate", None
                    )
                if not metadata["injection_time"]:
                    inj_time_legacy = getattr(
                        rp_info, "RadiopharmaceuticalStartTime", None
                    )
                    if inj_time_legacy:
                        metadata["injection_time"] = inj_time_legacy.split(".")[0]

    return metadata
