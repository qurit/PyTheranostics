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

    # Include the root itself if it looks like a CT folder
    root_name = root.name.lower()
    if (
        root.is_dir()
        and "ct" in root_name
        and "rtst" not in root_name
        and "nm" not in root_name
    ):
        candidates.append(root)

    for p in root.rglob("*"):
        if p.is_dir():
            name_lower = p.name.lower()
            if (
                "ct" in name_lower
                and "rtst" not in name_lower
                and "nm" not in name_lower
            ):
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


def run_segmentation_pipeline(
    input_folders: Optional[List[str]] = None,
    base_output_dir: str | Path = ".",
    *,
    root_dir: Optional[str | Path] = None,
    device: str = "mps",
    parallel: bool = False,
    max_workers: int = 2,
) -> Dict[str, Dict[str, Path]]:
    """Run CT segmentation with TotalSegmentator.

    This performs steps 1-2: discovery and segmentation. The output masks can then
    be converted to RT-STRUCT multiple times with different configurations.

    Parameters
    ----------
    input_folders : List[str], optional
        Explicit list of CT DICOM series folders (overrides discovery if given).
    base_output_dir : str | Path, optional
        Base directory where TotalSegmentator results are written, by default '.'.
    root_dir : str | Path, optional
        Root folder to discover CT series (if input_folders is None).
    device : str, optional
        'mps' (Apple), 'cuda', or 'cpu', by default "mps".
    parallel : bool, optional
        If True, run segmentation in parallel, by default False.
    max_workers : int, optional
        Number of workers for parallel processing, by default 2.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'segmentations': {patient_id -> {timepoint -> segmentation_output_dir}}
        - 'ct_paths': {patient_id -> {timepoint -> ct_dicom_folder}}
    """
    base_output_dir = Path(base_output_dir)

    # Determine and announce effective device
    resolved_device = device
    try:
        import torch

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

    # 1) Prepare segmentation tasks
    sp = SegmentationProcessor(str(base_output_dir), device=resolved_device)
    tasks: List[Tuple[Path, str, str, Path]] = []

    for ct_path in input_paths:
        tp = sp.extract_timepoint(ct_path)
        if tp == "unknown":
            continue
        patient_id = _read_patient_id_from_series(ct_path) or _sanitize_id(
            ct_path.parent.name
        )
        out_dir = base_output_dir / patient_id / tp
        tasks.append((ct_path, patient_id, tp, out_dir))

    if not tasks:
        raise RuntimeError("No valid CT series with timepoints found.")

    # 2) Run segmentation
    if parallel:
        from concurrent.futures import ProcessPoolExecutor

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

    # Return mapping of segmentation outputs and CT paths
    seg_map: Dict[str, Dict[str, Path]] = {}
    ct_paths: Dict[str, Dict[str, str]] = {}
    for ct_path, patient_id, tp, out_dir in tasks:
        seg_map.setdefault(patient_id, {})[tp] = out_dir
        ct_paths.setdefault(patient_id, {})[tp] = str(ct_path)

    return {"segmentations": seg_map, "ct_paths": ct_paths}


def convert_masks_to_rtstruct(
    segmentation_base_dir: str | Path,
    ct_series_paths: Dict[str, Dict[str, str]],
    rtstruct_output_dir: str | Path = "./RTStructs",
    config_path: Optional[str | Path] = None,
    export_csv: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """Convert NIfTI segmentation masks to RT-STRUCT files.

    This performs steps 3-4: RT-STRUCT conversion and CSV export. Can be run
    multiple times with different configs to generate different RT-STRUCTs.

    Parameters
    ----------
    segmentation_base_dir : str | Path
        Base directory containing segmentation outputs (patient_id/timepoint/*.nii.gz).
    ct_series_paths : Dict[str, Dict[str, str]]
        Mapping {patient_id -> {timepoint -> ct_dicom_folder}}.
    rtstruct_output_dir : str | Path, optional
        Directory where RT-STRUCT files will be saved, by default './RTStructs'.
    config_path : str | Path, optional
        Path to total_seg_config.json. If None, checks current directory.
    export_csv : bool, optional
        If True, writes all_rois.csv in rtstruct_output_dir, by default True.

    Returns
    -------
    Dict[str, Dict[str, Path]]
        Mapping {patient_id -> {timepoint -> rtstruct_path}}.
    """
    segmentation_base_dir = Path(segmentation_base_dir)
    rtstruct_output_dir = Path(rtstruct_output_dir)
    rtstruct_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine config path
    if config_path is None:
        config_path = Path.cwd() / "total_seg_config.json"
    else:
        config_path = Path(config_path)

    use_config = config_path.exists()
    if use_config:
        print(f"Using config: {config_path}")
    else:
        print("No config found, adding all masks")

    # Convert masks to RT-STRUCT
    patient_map: Dict[str, Dict[str, Path]] = {}

    for patient_id, timepoints in ct_series_paths.items():
        for tp, ct_path in timepoints.items():
            mask_dir = segmentation_base_dir / patient_id / tp
            if not mask_dir.exists():
                print(f"⚠️  Skipping {patient_id}/{tp}: {mask_dir} not found")
                continue

            rt_out_dir = rtstruct_output_dir / patient_id
            rt_out_dir.mkdir(parents=True, exist_ok=True)
            out_file = rt_out_dir / f"rtstruct_{tp}.dcm"
            print(f"RT-STRUCT: patient={patient_id} timepoint={tp} -> {out_file}")

            converter = RTStructConverter(ct_dicom_folder=str(ct_path))

            if use_config:
                converter.add_masks_from_folder_with_config(
                    str(mask_dir), str(config_path), permute_axes=True, flip_x=True
                )
            else:
                converter.add_masks_from_folder(
                    str(mask_dir), permute_axes=True, flip_x=True
                )

            converter.save(str(out_file))
            patient_map.setdefault(patient_id, {})[tp] = out_file

    # Export CSV
    if export_csv:
        export_multiple_rtstructs_to_csv(
            str(rtstruct_output_dir), str(rtstruct_output_dir / "all_rois.csv")
        )

    return patient_map


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
    """Run the complete workflow (segmentation + RT-STRUCT conversion).

    Convenience function that runs both segmentation and RT-STRUCT conversion.
    For more control, use run_segmentation_pipeline() and convert_masks_to_rtstruct()
    separately.

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
    # Step 1-2: Run segmentation
    result = run_segmentation_pipeline(
        input_folders=input_folders,
        base_output_dir=base_output_dir,
        root_dir=root_dir,
        device=device,
        parallel=parallel,
        max_workers=max_workers,
    )

    # Step 3-4: Convert to RT-STRUCT and export CSV
    return convert_masks_to_rtstruct(
        segmentation_base_dir=base_output_dir,
        ct_series_paths=result["ct_paths"],
        rtstruct_output_dir=rtstruct_output_dir,
        config_path=None,  # Will auto-detect in cwd
        export_csv=export_csv,
    )
