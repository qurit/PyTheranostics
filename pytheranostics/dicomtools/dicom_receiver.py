"""
DICOM receiver node for PyTheranostics.

Automatically receives and organizes DICOM images for dosimetry workflows.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pydicom
from pydicom.uid import (
    UID,
    DeflatedExplicitVRLittleEndian,
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    RLELossless,
)
from pynetdicom import AE, AllStoragePresentationContexts, evt
from pynetdicom.sop_class import Verification

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DICOMReceiver:
    """
    DICOM C-STORE SCP for receiving and organizing DICOM images for theranostic dosimetry workflows.

    Features:
    - Automatic organization by Patient ID, Study, and Modality
    - Metadata extraction for dosimetry parameters
    - Support for CT, SPECT/NM, PET, and RT Structure Sets
    - Configurable storage paths and callbacks
    """

    def __init__(
        self,
        ae_title: str = "PYTHERANOSTICS",
        port: int = 11112,
        storage_root: str = "./dicom_data",
        structured_storage: bool = True,
        allowed_calling_aets: Optional[List[str]] = None,
        auto_organize: bool = False,
        auto_organize_output_base: Optional[str] = None,
        auto_organize_cycle_gap_days: int = 15,
        auto_organize_timepoint_separation_days: int = 1,
        auto_organize_debounce_seconds: int = 60,
    ):
        """
        Initialize DICOM receiver.

        Parameters
        ----------
        ae_title : str
            Application Entity title for this DICOM node
        port : int
            Port to listen on for incoming DICOM connections
        storage_root : str
            Root directory for storing received DICOM files
        structured_storage : bool
            If True, organize files by PatientID/StudyDate/Modality/SeriesNumber
        auto_organize : bool
            If True, automatically organize received series into Cycle/Timepoint folders after a period of inactivity.
        auto_organize_output_base : str | None
            Base directory to write organized output. Defaults to storage_root if None.
        auto_organize_cycle_gap_days : int
            New cycle if consecutive scans are >= this many days apart (default 15).
        auto_organize_timepoint_separation_days : int
            New timepoint if date changes by this many days (default 1).
        auto_organize_debounce_seconds : int
            Wait time after the last received file before organizing (per-patient).
        """
        self.ae_title = ae_title
        self.port = port
        self.storage_root = Path(storage_root)
        self.structured_storage = structured_storage
        self.storage_root.mkdir(parents=True, exist_ok=True)
        # Auto organize configuration
        self.auto_organize = auto_organize
        self.auto_organize_output_base = (
            Path(auto_organize_output_base)
            if auto_organize_output_base
            else self.storage_root
        )
        self.auto_organize_cycle_gap_days = auto_organize_cycle_gap_days
        self.auto_organize_timepoint_separation_days = (
            auto_organize_timepoint_separation_days
        )
        self.auto_organize_debounce_seconds = auto_organize_debounce_seconds
        self._organize_timers = {}

        # Initialize Application Entity
        self.ae = AE(ae_title=ae_title)

        # Optionally restrict which Calling AE Titles are accepted
        if allowed_calling_aets:
            # Exact string match, max 16 chars each
            self.ae.require_calling_aet = [
                aet.strip()[:16] for aet in allowed_calling_aets
            ]

        # Add supported presentation contexts with a broad set of transfer syntaxes
        # Robust set of common transfer syntaxes, using UIDs directly for compatibility
        JPEG_BASELINE = UID("1.2.840.10008.1.2.4.50")  # JPEG Baseline (Process 1)
        JPEG_EXTENDED = UID("1.2.840.10008.1.2.4.51")  # JPEG Extended (Process 2 & 4)
        JPEG_LOSSLESS_P14 = UID(
            "1.2.840.10008.1.2.4.57"
        )  # JPEG Lossless, Non-Hierarchical (Process 14)
        JPEG_LOSSLESS = UID(
            "1.2.840.10008.1.2.4.70"
        )  # JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])
        JPEG2000_LOSSLESS = UID(
            "1.2.840.10008.1.2.4.90"
        )  # JPEG 2000 Image Compression (Lossless Only)
        JPEG2000 = UID("1.2.840.10008.1.2.4.91")  # JPEG 2000 Image Compression

        transfer_syntaxes = [
            ImplicitVRLittleEndian,
            ExplicitVRLittleEndian,
            ExplicitVRBigEndian,
            DeflatedExplicitVRLittleEndian,
            RLELossless,
            JPEG_BASELINE,
            JPEG_EXTENDED,
            JPEG_LOSSLESS_P14,
            JPEG_LOSSLESS,
            JPEG2000,
            JPEG2000_LOSSLESS,
        ]

        for cx in AllStoragePresentationContexts:
            self.ae.add_supported_context(cx.abstract_syntax, transfer_syntaxes)

        # Support C-ECHO
        self.ae.add_supported_context(Verification)

        # Storage for metadata
        self.metadata_file = self.storage_root / "received_studies.json"
        self.metadata = self._load_metadata()

        # Callbacks
        self.on_study_complete_callback: Optional[Callable] = None

    # Internal: schedule organization after inactivity per patient
    def _schedule_auto_organize(self, patient_id: str):
        if not self.auto_organize:
            return
        try:
            import threading

            # Cancel existing timer if present
            t = self._organize_timers.get(patient_id)
            if t and t.is_alive():
                t.cancel()

            def _runner():
                try:
                    logger.info(
                        f"Auto-organizing cycles for patient {patient_id} after {self.auto_organize_debounce_seconds}s idle"
                    )
                    self.organize_by_cycles(
                        patient_id=patient_id,
                        output_base=self.auto_organize_output_base,
                        cycle_gap_days=self.auto_organize_cycle_gap_days,
                        timepoint_separation_days=self.auto_organize_timepoint_separation_days,
                    )
                except Exception as e:
                    logger.exception(f"Auto-organize failed for {patient_id}: {e}")

            timer = threading.Timer(self.auto_organize_debounce_seconds, _runner)
            timer.daemon = True
            self._organize_timers[patient_id] = timer
            timer.start()
        except Exception as e:
            logger.exception(f"Failed to schedule auto-organize for {patient_id}: {e}")

    def _load_metadata(self) -> Dict:
        """Load existing metadata from a JSON file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save metadata to a JSON file."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def _extract_patient_info(self, ds: pydicom.Dataset) -> Dict:
        """
        Extract relevant patient and injection information from a DICOM dataset.

        Parameters
        ----------
        ds : pydicom.Dataset
            DICOM dataset

        Returns
        -------
        dict
            Extracted patient information
        """
        info = {
            "PatientID": getattr(ds, "PatientID", "UNKNOWN"),
            "PatientName": str(getattr(ds, "PatientName", "UNKNOWN")),
            "PatientWeight": getattr(ds, "PatientWeight", None),  # in kg
            "StudyDate": getattr(ds, "StudyDate", None),
            "StudyTime": getattr(ds, "StudyTime", None),
            "StudyDescription": getattr(ds, "StudyDescription", ""),
            "SeriesDescription": getattr(ds, "SeriesDescription", ""),
            "Modality": getattr(ds, "Modality", "UNKNOWN"),
            "SeriesNumber": getattr(ds, "SeriesNumber", 0),
            "InstanceNumber": getattr(ds, "InstanceNumber", 0),
        }

        # Extract nuclear medicine specific information
        if info["Modality"] in ["NM", "PT"]:
            info["Radiopharmaceutical"] = None
            info["InjectedActivity"] = None
            info["InjectionDateTime"] = None

            # Check for RadiopharmaceuticalInformationSequence
            if hasattr(ds, "RadiopharmaceuticalInformationSequence"):
                rp_seq = ds.RadiopharmaceuticalInformationSequence
                if len(rp_seq) > 0:
                    rp_info = rp_seq[0]
                    info["Radiopharmaceutical"] = getattr(
                        rp_info, "Radiopharmaceutical", None
                    )
                    info["InjectedActivity"] = getattr(
                        rp_info, "RadionuclideTotalDose", None
                    )

                    # Combine date and time
                    inj_date = getattr(rp_info, "RadiopharmaceuticalStartDate", None)
                    inj_time = getattr(rp_info, "RadiopharmaceuticalStartTime", None)
                    if inj_date and inj_time:
                        info["InjectionDateTime"] = f"{inj_date} {inj_time}"

        return info

    def _get_storage_path(self, ds: pydicom.Dataset) -> Path:
        """
        Determine storage path for a DICOM file.

        Parameters
        ----------
        ds : pydicom.Dataset
            DICOM dataset

        Returns
        -------
        Path
            Directory path where file should be stored
        """
        if not self.structured_storage:
            return self.storage_root

        patient_id = getattr(ds, "PatientID", "UNKNOWN")
        study_date = getattr(ds, "StudyDate", "UNKNOWN")
        modality = getattr(ds, "Modality", "UNKNOWN")

        # Create structure: PatientID/StudyDate/Modality
        path = self.storage_root / patient_id / study_date / modality

        # Special handling for RT Structure Sets
        if modality == "RTSTRUCT":
            path = self.storage_root / patient_id / study_date / "CT" / "RTstruct"

        path.mkdir(parents=True, exist_ok=True)
        return path

    # --------------------------
    # Post-processing utilities
    # --------------------------
    @staticmethod
    def _parse_dt(
        date_str: Optional[str], time_str: Optional[str]
    ) -> Optional[datetime]:
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

    @staticmethod
    def _series_datetime_from_any(dcm: pydicom.Dataset) -> Optional[datetime]:
        """Best-effort extraction of a datetime for a DICOM series instance.

        Tries SeriesDate/Time, then AcquisitionDate/Time, then ContentDate/Time,
        finally falls back to StudyDate/Time.
        """
        # Series
        dt = DICOMReceiver._parse_dt(
            getattr(dcm, "SeriesDate", None), getattr(dcm, "SeriesTime", None)
        )
        if dt:
            return dt
        # Acquisition
        dt = DICOMReceiver._parse_dt(
            getattr(dcm, "AcquisitionDate", None), getattr(dcm, "AcquisitionTime", None)
        )
        if dt:
            return dt
        # Content
        dt = DICOMReceiver._parse_dt(
            getattr(dcm, "ContentDate", None), getattr(dcm, "ContentTime", None)
        )
        if dt:
            return dt
        # Study
        return DICOMReceiver._parse_dt(
            getattr(dcm, "StudyDate", None), getattr(dcm, "StudyTime", None)
        )

    @staticmethod
    def _get_any_dicom_datetime_in_path(path: Path) -> Optional[datetime]:
        """Find any DICOM file in a directory and return its best-effort datetime.

        Parameters
        ----------
        path : Path
            Directory containing DICOM files

        Returns
        -------
        datetime | None
        """
        try:
            for dcm_file in sorted(path.glob("*.dcm")):
                try:
                    ds = pydicom.dcmread(
                        str(dcm_file), stop_before_pixels=True, force=True
                    )
                    dt = DICOMReceiver._series_datetime_from_any(ds)
                    if dt:
                        return dt
                except Exception:
                    continue
            return None
        except Exception:
            return None

    def _collect_patient_series(self, patient_id: str) -> List[Dict]:
        """Collect all known series for a patient across all studies.

        Returns list of dicts with keys: modality, series_number, series_description,
        path (Path), datetime (datetime | None), study_date (str | None).
        """
        series_list: List[Dict] = []
        for key, info in self.metadata.items():
            if not key.startswith(f"{patient_id}_"):
                continue
            study_date = info.get("patient_info", {}).get("StudyDate")
            series = info.get("series", {})
            for s_key, s in series.items():
                src_path = Path(s.get("path", self.storage_root))
                # Determine a representative datetime for the series
                rep_dt = self._get_any_dicom_datetime_in_path(src_path)
                if rep_dt is None and study_date:
                    # Fallback to study_date
                    rep_dt = self._parse_dt(
                        study_date, info.get("patient_info", {}).get("StudyTime")
                    )
                series_list.append(
                    {
                        "modality": s.get("modality", "UNKNOWN"),
                        "series_number": s.get("series_number", 0),
                        "series_description": s.get("series_description", ""),
                        "path": src_path,
                        "datetime": rep_dt,
                        "study_date": study_date,
                    }
                )
        # Filter out those without any path
        return [x for x in series_list if x.get("path") is not None]

    def organize_by_cycles(
        self,
        patient_id: str,
        output_base: Path,
        cycle_gap_days: int = 15,
        timepoint_separation_days: int = 1,
    ) -> Dict[str, Dict[str, List[Path]]]:
        """Post-process received DICOMs into Cycle/Timepoint structure.

                Creates folders like:
                    PatientID/Cycle1/tp1/CT/Series3
                    PatientID/Cycle1/tp1/SPECT/Series5
                    PatientID/Cycle1/tp2/CT/Series2

                RTSTRUCT will be placed under the corresponding CT timepoint:
                    PatientID/Cycle1/tp1/CT/RTstruct/Series7

        Parameters
        ----------
        patient_id : str
            Patient identifier
        output_base : Path
            Directory under which the new structure will be created
        cycle_gap_days : int
            Start a new cycle if the gap since the previous scan is >= this many days (default 15 days).
        timepoint_separation_days : int
            Start a new timepoint when acquisition date changes by this many days or more (default 1 day)

        Returns
        -------
        dict
            Nested dict with created directories per cycle and timepoint
        """
        series_list = self._collect_patient_series(patient_id)
        if not series_list:
            raise ValueError(f"No series found for patient '{patient_id}'.")

        # Ensure we have datetimes; if some missing, use file mtime as last resort
        for s in series_list:
            if s["datetime"] is None:
                try:
                    any_file = next(iter(sorted(s["path"].glob("*.dcm"))))
                    mtime = datetime.fromtimestamp(any_file.stat().st_mtime)
                    s["datetime"] = mtime
                except StopIteration:
                    # No files present - skip later
                    s["datetime"] = None

        # Drop any without datetime ultimately
        series_list = [s for s in series_list if s["datetime"] is not None]

        # Group series by StudyDate to define timepoints, so RTSTRUCT doesn't create new cycles
        # Build mapping: study_date -> list[series]
        tp_by_date: Dict[str, List[Dict]] = {}
        for s in series_list:
            sd = s.get("study_date") or s["datetime"].strftime("%Y%m%d")
            tp_by_date.setdefault(sd, []).append(s)

        # Sort timepoints by study date
        sorted_dates = sorted(tp_by_date.keys())

        out: Dict[str, Dict[str, List[Path]]] = {}
        patient_root = Path(output_base) / patient_id
        patient_root.mkdir(parents=True, exist_ok=True)

        if not sorted_dates:
            return out

        # Compute cycles from consecutive study date gaps
        cycle_idx = 1
        tp_idx = 1
        prev_date_dt = datetime.strptime(sorted_dates[0], "%Y%m%d")

        for i, sd in enumerate(sorted_dates):
            this_date_dt = datetime.strptime(sd, "%Y%m%d")
            if i > 0:
                if (this_date_dt - prev_date_dt) >= timedelta(days=cycle_gap_days):
                    # New cycle
                    cycle_idx += 1
                    tp_idx = 1
                else:
                    # Same cycle, next timepoint (optionally collapse same-day scans if needed)
                    if (
                        this_date_dt.date() - prev_date_dt.date()
                    ).days >= timepoint_separation_days:
                        tp_idx += 1

            # For all series in this study date, place under tp folder
            cycle_dir = patient_root / f"Cycle{cycle_idx}" / f"tp{tp_idx}"
            cycle_dir.mkdir(parents=True, exist_ok=True)

            # Track source modality directories seen for cleanup after moving
            src_dirs_for_cleanup: set[Path] = set()

            for s in tp_by_date[sd]:
                modality = s["modality"]
                # Normalize modality names for destination
                if modality in ["NM", "PT"]:
                    modality_folder = "SPECT"
                elif modality == "RTSTRUCT":
                    modality_folder = "CT"  # RTSTRUCT under CT/RTstruct
                else:
                    modality_folder = modality

                series_number = s.get("series_number", 0) or 0
                # Destination folders drop the Series subfolder; put instances directly under modality
                if modality == "RTSTRUCT":
                    dest_dir = cycle_dir / "CT" / "RTstruct"
                else:
                    dest_dir = cycle_dir / modality_folder

                dest_dir.mkdir(parents=True, exist_ok=True)

                # Copy only files belonging to this SeriesNumber
                src_path: Path = s["path"]
                src_dirs_for_cleanup.add(src_path)
                copied = 0
                for dcm_file in src_path.glob("*.dcm"):
                    try:
                        ds = pydicom.dcmread(
                            str(dcm_file), stop_before_pixels=True, force=True
                        )
                        if int(getattr(ds, "SeriesNumber", -1) or -1) == int(
                            series_number
                        ):
                            import shutil

                            dest_file = dest_dir / dcm_file.name
                            if dest_file.exists():
                                # Skip if already present to avoid accidental overwrite
                                continue
                            # Move instead of copy to avoid duplication
                            shutil.move(str(dcm_file), str(dest_file))
                            copied += 1
                    except Exception:
                        continue
                logger.info(
                    f"Organized {copied} files -> {dest_dir} ({modality}, Series{int(series_number)}, {sd})"
                )

                # Record in output mapping
                cycle_key = f"Cycle{cycle_idx}"
                tp_key = f"tp{tp_idx}"
                out.setdefault(cycle_key, {}).setdefault(tp_key, []).append(dest_dir)

            # After processing all series for this StudyDate, prune empty source directories
            try:
                for src_dir in src_dirs_for_cleanup:
                    # Remove dir if empty
                    try:
                        if src_dir.exists() and not any(src_dir.iterdir()):
                            src_dir.rmdir()
                    except Exception:
                        pass
                    # Attempt to remove parent StudyDate dir if empty
                    try:
                        study_parent = src_dir.parent
                        if study_parent.exists() and not any(study_parent.iterdir()):
                            study_parent.rmdir()
                    except Exception:
                        pass
                    # Attempt to remove patient dir if now empty (rare)
                    try:
                        patient_dir = study_parent.parent
                        if patient_dir.exists() and not any(patient_dir.iterdir()):
                            patient_dir.rmdir()
                    except Exception:
                        pass
            except Exception:
                logger.debug("Cleanup after move encountered issues; continuing.")

            prev_date_dt = this_date_dt

        logger.info(
            f"Cycle/Timepoint organization complete for patient {patient_id} at {patient_root}"
        )
        return out

    def _handle_store(self, event):
        """
        Handle an incoming C-STORE request.

        Parameters
        ----------
        event : pynetdicom.events.Event
            The event corresponding to the C-STORE request

        Returns
        -------
        int
            DICOM status code (0x0000 for success)
        """
        try:
            ds = event.dataset
            ds.file_meta = event.file_meta

            # Extract information
            patient_info = self._extract_patient_info(ds)
            storage_path = self._get_storage_path(ds)

            # Generate filename
            sop_instance_uid = ds.SOPInstanceUID
            filename = storage_path / f"{sop_instance_uid}.dcm"

            # Save DICOM file (avoid deprecation; enforce standard file format)
            ds.save_as(filename, enforce_file_format=True)

            logger.info(
                f"Received and stored: {patient_info['Modality']} - "
                f"Patient {patient_info['PatientID']} - "
                f"Series {patient_info['SeriesNumber']}"
            )

            # Update metadata
            study_key = f"{patient_info['PatientID']}_{patient_info['StudyDate']}"
            if study_key not in self.metadata:
                self.metadata[study_key] = {
                    "patient_info": patient_info,
                    "series": {},
                    "received_date": datetime.now().isoformat(),
                }

            series_key = f"{patient_info['Modality']}_{patient_info['SeriesNumber']}"
            if series_key not in self.metadata[study_key]["series"]:
                self.metadata[study_key]["series"][series_key] = {
                    "modality": patient_info["Modality"],
                    "series_number": patient_info["SeriesNumber"],
                    "series_description": patient_info["SeriesDescription"],
                    "instance_count": 0,
                    "path": str(storage_path),
                }

            self.metadata[study_key]["series"][series_key]["instance_count"] += 1
            self._save_metadata()

            # Schedule auto-organize for this patient (debounced)
            try:
                self._schedule_auto_organize(patient_info["PatientID"])
            except Exception as e:
                logger.exception(f"Failed to schedule auto-organize: {e}")

            # Return success status
            return 0x0000
        except Exception as e:
            logger.exception(f"C-STORE processing failed: {e}")
            # Processing failure
            return 0xC000

    @staticmethod
    def _handle_echo(event):
        """Respond to a C-ECHO request with Success."""
        return 0x0000

    @staticmethod
    def _handle_accepted(event):
        """Log details when an association is accepted."""
        try:
            assoc = event.assoc
            calling = assoc.requestor.ae_title.decode("ascii", errors="ignore").strip()
            called = assoc.acceptor.ae_title.decode("ascii", errors="ignore").strip()
            addr = f"{assoc.requestor.address}:{assoc.requestor.port}"
            logger.info(
                f"Association accepted: CallingAE='{calling}' -> CalledAE='{called}' from {addr}"
            )
        except Exception:
            logger.info("Association accepted")

    def start(self, blocking: bool = True):
        """
        Start the DICOM receiver.

        Parameters
        ----------
        blocking : bool
            If True, blocks until server is stopped. If False, runs in background.
        """
        handlers = [
            (evt.EVT_C_STORE, self._handle_store),
            (evt.EVT_C_ECHO, self._handle_echo),
            (evt.EVT_ACCEPTED, self._handle_accepted),
        ]

        logger.info(f"Starting DICOM Receiver: {self.ae_title} on port {self.port}")
        logger.info(f"Storage location: {self.storage_root.absolute()}")

        if blocking:
            self.ae.start_server(("", self.port), evt_handlers=handlers)
        else:
            import threading

            self.server_thread = threading.Thread(
                target=self.ae.start_server,
                args=(("", self.port),),
                kwargs={"evt_handlers": handlers},
                daemon=True,
            )
            self.server_thread.start()
            logger.info("DICOM Receiver started in background")

    def stop(self):
        """Stop the DICOM receiver."""
        # Cancel any pending auto-organize timers
        try:
            for t in list(self._organize_timers.values()):
                try:
                    if t and getattr(t, "is_alive", lambda: False)():
                        t.cancel()
                except Exception:
                    pass
            self._organize_timers.clear()
        except Exception:
            pass
        self.ae.shutdown()
        logger.info("DICOM Receiver stopped")

    def get_study_info(self, patient_id: str, study_date: str = None) -> Dict:
        """
        Get information about received studies for a patient.

        Parameters
        ----------
        patient_id : str
            Patient ID
        study_date : str, optional
            Specific study date (YYYYMMDD format)

        Returns
        -------
        dict
            Study information
        """
        if study_date:
            study_key = f"{patient_id}_{study_date}"
            return self.metadata.get(study_key, {})
        else:
            # Return all studies for this patient
            return {k: v for k, v in self.metadata.items() if k.startswith(patient_id)}

    def organize_for_dosimetry(
        self,
        patient_id: str,
        study_date: str,
        output_base: Path,
        cycle_name: str = "cycle01",
    ) -> Dict[str, Path]:
        """
        Organize received DICOM files into pytheranostics expected structure.

        Parameters
        ----------
        patient_id : str
            Patient ID
        study_date : str
            Study date (YYYYMMDD)
        output_base : Path
            Base output directory
        cycle_name : str
            Cycle name (e.g., 'cycle01')

        Returns
        -------
        dict
            Paths to organized data: {'ct': [...], 'spect': [...], 'rtstruct': [...]}
        """
        study_info = self.get_study_info(patient_id, study_date)
        if not study_info:
            raise ValueError(f"No study found for {patient_id} on {study_date}")

        output_dir = Path(output_base) / patient_id / cycle_name
        output_dir.mkdir(parents=True, exist_ok=True)

        organized_paths = {"ct": [], "spect": [], "rtstruct": []}

        # Group series by time point (based on series time or number)
        time_point = 1

        for series_key, series_info in study_info["series"].items():
            modality = series_info["modality"]
            source_path = Path(series_info["path"])

            if modality == "CT":
                dest_path = output_dir / f"tp{time_point}" / "CT"
                dest_path.mkdir(parents=True, exist_ok=True)
                organized_paths["ct"].append(dest_path)

                # Copy files
                for dcm_file in source_path.glob("*.dcm"):
                    import shutil

                    shutil.copy2(dcm_file, dest_path / dcm_file.name)

            elif modality in ["NM", "PT"]:
                dest_path = output_dir / f"tp{time_point}" / "SPECT"
                dest_path.mkdir(parents=True, exist_ok=True)
                organized_paths["spect"].append(dest_path)

                # Copy files
                for dcm_file in source_path.glob("*.dcm"):
                    import shutil

                    shutil.copy2(dcm_file, dest_path / dcm_file.name)

            elif modality == "RTSTRUCT":
                dest_path = output_dir / f"tp{time_point}" / "CT" / "RTstruct"
                dest_path.mkdir(parents=True, exist_ok=True)
                organized_paths["rtstruct"].append(dest_path)

                # Copy files
                for dcm_file in source_path.glob("*.dcm"):
                    import shutil

                    shutil.copy2(dcm_file, dest_path / dcm_file.name)

        logger.info(f"Organized data for {patient_id} into {output_dir}")
        return organized_paths


def create_receiver(
    ae_title: str = "PYTHERANOSTICS",
    port: int = 11112,
    storage_root: str = "./dicom_data",
    allowed_calling_aets: Optional[List[str]] = None,
    auto_organize: bool = False,
    auto_organize_output_base: Optional[str] = None,
    auto_organize_cycle_gap_days: int = 15,
    auto_organize_timepoint_separation_days: int = 1,
    auto_organize_debounce_seconds: int = 60,
) -> DICOMReceiver:
    """
    Create a DICOM receiver.

    Parameters
    ----------
    ae_title : str
        Application Entity title
    port : int
        Port number
    storage_root : str
        Root directory for storage
    allowed_calling_aets : list[str] | None
        Optional list of allowed Calling AE Titles (whitelist). If None, accept any.

    Returns
    -------
    DICOMReceiver
        Configured DICOM receiver instance
    """
    return DICOMReceiver(
        ae_title=ae_title,
        port=port,
        storage_root=storage_root,
        allowed_calling_aets=allowed_calling_aets,
        auto_organize=auto_organize,
        auto_organize_output_base=auto_organize_output_base,
        auto_organize_cycle_gap_days=auto_organize_cycle_gap_days,
        auto_organize_timepoint_separation_days=auto_organize_timepoint_separation_days,
        auto_organize_debounce_seconds=auto_organize_debounce_seconds,
    )
