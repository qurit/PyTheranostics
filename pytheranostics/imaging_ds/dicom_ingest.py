"""
DICOM ingestion utilities for PyTheranostics.

Simplifies data ingestion for dosimetry workflows.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pydicom

logger = logging.getLogger(__name__)


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

                # Extract date and time
                inj_date = getattr(rp_info, "RadiopharmaceuticalStartDate", None)
                inj_time = getattr(rp_info, "RadiopharmaceuticalStartTime", None)

                if inj_date:
                    info["injection_date"] = inj_date
                if inj_time:
                    # Format time to HHMMSS
                    info["injection_time"] = inj_time.split(".")[
                        0
                    ]  # Remove fractional seconds

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
                metadata["injection_date"] = getattr(
                    rp_info, "RadiopharmaceuticalStartDate", None
                )
                metadata["injection_time"] = getattr(
                    rp_info, "RadiopharmaceuticalStartTime", None
                )
                if metadata["injection_time"]:
                    metadata["injection_time"] = metadata["injection_time"].split(".")[
                        0
                    ]

    return metadata
