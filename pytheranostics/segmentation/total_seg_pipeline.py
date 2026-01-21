"""High-level pipeline to run segmentation, RT-STRUCT conversion, and ROI CSV export.

With minimal notebook code.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pydicom

from .rtst_utilities import RTStructConverter, export_multiple_rtstructs_to_csv
from .total_seg_segmentation import SegmentationProcessor


def _seg_one_worker(args: Tuple[str, str, str, str, str]):
    """Top-level worker function for process pool (must be picklable).

    Args tuple: (ct_path, patient_id, timepoint, out_dir, device)
    """
    ct_path_str, patient_id, tp, out_dir_str, device = args
    ct_path = Path(ct_path_str)
    out_dir = Path(out_dir_str)
    print(
        f"Segmentation: patient={patient_id} timepoint={tp}\n  CT={ct_path}\n  OUT={out_dir}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    from totalsegmentator.python_api import totalsegmentator

    totalsegmentator(str(ct_path), str(out_dir), device=device)


def _discover_ct_series(root_dir: str | Path) -> List[Path]:
    """Recursively find CT series folders under a root directory.

    Heuristics:
    - Directory name contains 'CT'
    - Directory name does NOT contain 'RTst' or 'NM'
    - Timepoint token like '-CT.<...>h' is present (validated later)
    """
    root = Path(root_dir)
    candidates: List[Path] = []
    for p in root.rglob("*"):
        if p.is_dir():
            name = p.name
            if "CT" in name and "RTst" not in name and "NM" not in name:
                candidates.append(p)
    return candidates


def _read_patient_id_from_series(series_dir: Path) -> Optional[str]:
    """Read PatientID from any DICOM file within a CT series folder.

    Returns a sanitized PatientID or None if not found.
    """
    try:
        # Try a few files in the series directory
        for f in series_dir.iterdir():
            if f.is_file():
                try:
                    ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
                    pid = getattr(ds, "PatientID", None)
                    if pid:
                        return _sanitize_id(str(pid))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return None


def _sanitize_id(text: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text
    ).strip("_")


def run_full_pipeline(
    input_folders: Optional[List[str]] = None,
    base_output_dir: str | Path = ".",
    rtstruct_output_dir: str | Path = "./RTStructs",
    *,
    root_dir: Optional[str | Path] = None,
    device: str = "mps",
    parallel: bool = False,
    max_workers: int = 2,
    export_csv: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """Run the complete workflow.

    1) Discover CT DICOM series under a root directory (if provided) or use input_folders
    2) Segment each input CT series with TotalSegmentator into per-patient/per-timepoint subfolders
    3) Convert all masks to RT-STRUCT per timepoint per patient
    4) Optionally export ROI inventory CSV (recursively) under rtstruct_output_dir

    Parameters
    ----------
    input_folders : List[str], optional
        Explicit list of CT DICOM series folders (overrides discovery if given).
    base_output_dir : str | Path, optional
        Base directory where TotalSegmentator results are written, by default '.'.
    rtstruct_output_dir : str | Path, optional
        Directory where RT-STRUCT files will be saved, by default './RTStructs'.
    root_dir : str | Path, optional
        Root folder to discover CT series (if input_folders is None).
    device : str, optional
        'mps' (Apple), 'cuda', or 'cpu', by default "mps".
    parallel : bool, optional
        If True, run segmentation in parallel, by default False.
    max_workers : int, optional
        Number of workers for parallel processing, by default 2.
    export_csv : bool, optional
        If True, writes all_rois.csv in rtstruct_output_dir (recursive), by default True.

    Returns
    -------
    Dict[str, Dict[str, Path]]
        Mapping {patient_id -> {timepoint -> rtstruct_path}}.
    """
    base_output_dir = Path(base_output_dir)
    rtstruct_output_dir = Path(rtstruct_output_dir)
    rtstruct_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine and announce effective device
    resolved_device = device
    try:
        import torch  # local import to avoid hard dependency at import time

        mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        cuda_avail = torch.cuda.is_available()

        if device.lower() == "auto":
            if cuda_avail:
                resolved_device = "cuda"
            elif mps_avail:
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        elif device.lower() == "cuda" and not cuda_avail:
            print("Requested CUDA but not available; falling back to CPU")
            resolved_device = "cpu"
        elif device.lower() == "mps" and not mps_avail:
            print("Requested MPS but not available; falling back to CPU")
            resolved_device = "cpu"
        else:
            resolved_device = device.lower()
    except Exception:
        resolved_device = device.lower()

    print(f"Using device: {resolved_device.upper()}")

    # 0) Determine input series
    if input_folders is None or len(input_folders) == 0:
        if root_dir is None:
            raise ValueError("Provide either input_folders or root_dir for discovery.")
        discovered = _discover_ct_series(root_dir)
        if not discovered:
            raise FileNotFoundError(f"No CT series found under root: {root_dir}")
        input_paths = discovered
    else:
        input_paths = [Path(p) for p in input_folders]

    # 1) Prepare segmentation tasks: compute patient_id and timepoint
    sp = SegmentationProcessor(str(base_output_dir), device=resolved_device)
    tasks: List[Tuple[Path, str, str, Path]] = (
        []
    )  # (ct_path, patient_id, timepoint, out_dir)

    for ct_path in input_paths:
        tp = sp.extract_timepoint(ct_path)
        if tp == "unknown":
            # Skip entries without a parsable timepoint
            continue
        patient_id = _read_patient_id_from_series(ct_path) or _sanitize_id(
            ct_path.parent.name
        )
        out_dir = base_output_dir / patient_id / tp
        tasks.append((ct_path, patient_id, tp, out_dir))

    if not tasks:
        raise RuntimeError("No valid CT series with timepoints found.")

    # 2) Run segmentation (optionally in parallel)
    if parallel:
        from concurrent.futures import ProcessPoolExecutor

        # Convert tasks into simple tuples of strings for pickling safety
        safe_tasks = [
            (str(ct_path), patient_id, tp, str(out_dir), resolved_device)
            for ct_path, patient_id, tp, out_dir in tasks
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(_seg_one_worker, safe_tasks))
    else:
        for ct_path, patient_id, tp, out_dir in tasks:
            _seg_one_worker(
                (str(ct_path), patient_id, tp, str(out_dir), resolved_device)
            )

    # 3) Convert masks to RT-STRUCT per patient/timepoint
    patient_map: Dict[str, Dict[str, Path]] = {}
    for ct_path, patient_id, tp, out_dir in tasks:
        if not out_dir.exists():
            print(f"⚠️  Skipping RT-STRUCT for {patient_id}/{tp}: {out_dir} not found")
            continue
        rt_out_dir = rtstruct_output_dir / patient_id
        rt_out_dir.mkdir(parents=True, exist_ok=True)
        out_file = rt_out_dir / f"rtstruct_{tp}.dcm"
        print(f"RT-STRUCT: patient={patient_id} timepoint={tp} -> {out_file}")
        converter = RTStructConverter(ct_dicom_folder=str(ct_path))
        converter.add_masks_from_folder(str(out_dir), permute_axes=True, flip_x=True)
        converter.save(str(out_file))
        patient_map.setdefault(patient_id, {})[tp] = out_file

    # 4) Export a single CSV for all RT-STRUCTs (recursive)
    if export_csv:
        export_multiple_rtstructs_to_csv(
            str(rtstruct_output_dir), str(rtstruct_output_dir / "all_rois.csv")
        )

    return patient_map
