"""Slow validation test for the tutorial-like voxel dosimetry pipeline."""

from __future__ import annotations

import json
import os
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pydicom
import pytest
from pandas.testing import assert_frame_equal
from pydicom.sequence import Sequence

from pytheranostics import init_project
from pytheranostics.data_fetchers import fetch_snmmi_dosimetry_challenge
from pytheranostics.dosimetry import build_roi_fit_config
from pytheranostics.dosimetry.voxel_s_dosimetry import VoxelSDosimetry
from pytheranostics.imaging_ds import create_studies_with_masks
from pytheranostics.imaging_ds.dicom_ingest import auto_setup_dosimetry_study_inventory

_VALIDATION_ASSETS: Final[Path] = (
    Path(__file__).resolve().parent / "data" / "voxel_dosimetry_validation"
)
_EXPECTED_RESULTS_CSV: Final[Path] = _VALIDATION_ASSETS / "expected_results.csv"
_EXPECTED_DF_AD_CSV: Final[Path] = _VALIDATION_ASSETS / "expected_df_ad.csv"
_RTSTRUCT_ZENODO_RECORD_ID: Final[str] = "21893683"
_RTSTRUCT_ZENODO_RECORD_API_URL: Final[str] = (
    f"https://zenodo.org/api/records/{_RTSTRUCT_ZENODO_RECORD_ID}"
)
_RTSTRUCT_DOWNLOAD_TIMEOUT_S: Final[int] = 120
_ZENODO_USER_AGENT: Final[str] = "PyTheranostics test suite"
_RTSTRUCT_SOP_CLASS_UID: Final[str] = "1.2.840.10008.5.1.4.1.1.481.3"
_EXPECTED_RTSTRUCT_FILENAMES: Final[tuple[str, ...]] = (
    "rtstruct_scan1.dcm",
    "rtstruct_scan2.dcm",
    "rtstruct_scan3.dcm",
    "rtstruct_scan4.dcm",
)
_OPTIONAL_CONFIG_FILES: Final[tuple[str, ...]] = (
    "voi_mappings_config.json",
    "dosimetry_fit_defaults.json",
)
_LIST_LIKE_RESULTS_COLUMNS: Final[set[str]] = {
    "Time_hr",
    "Volume_CT_mL",
    "Activity_MBq",
    "Density_HU",
    "Fit_params",
    "R_squared_AIC",
    "Lambda_eff",
}


def _skip_snmmi_fetch_failure_in_ci_or_fail_locally(reason: str) -> None:
    message = (
        f"{reason}\n\n"
        "GitHub Actions is allowed to skip this external validation test because "
        "the SNMMI Deep Blue dataset can reject CI-hosted downloads. Before "
        "opening a PR, run this test locally with:\n"
        "  pytest tests/test_voxel_dosimetry_validation.py -rs"
    )
    if os.environ.get("CI", "").lower() == "true":
        pytest.skip(message)
    pytest.fail(message)


def _skip_if_validation_reference_assets_missing() -> None:
    missing = [
        path.name
        for path in (_EXPECTED_RESULTS_CSV, _EXPECTED_DF_AD_CSV)
        if not path.exists()
    ]
    if missing:
        pytest.skip(
            "Voxel dosimetry validation reference assets are not available. Missing: "
            + ", ".join(missing)
            + f". Populate {_VALIDATION_ASSETS} to enable this test."
        )


def _copy_optional_validation_configs(project_base: Path) -> None:
    for config_name in _OPTIONAL_CONFIG_FILES:
        src = _VALIDATION_ASSETS / config_name
        if src.exists():
            shutil.copy2(src, project_base / config_name)


def _download_url_to_file(url: str, output_path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": _ZENODO_USER_AGENT})
    with urllib.request.urlopen(
        request, timeout=_RTSTRUCT_DOWNLOAD_TIMEOUT_S
    ) as response, output_path.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)


def _validate_rtstruct_file(rtstruct_path: Path) -> None:
    try:
        ds = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
    except Exception as exc:
        raise ValueError(f"{rtstruct_path.name} is not a readable DICOM file.") from exc

    # RTROIObservationsSequence is optional RTSTRUCT metadata and is not used by
    # the mask-loading pipeline.  Some valid, anonymized RT Structure Sets omit
    # it, including the validation assets hosted on Zenodo.
    missing_sequences = [
        sequence_name
        for sequence_name in (
            "ROIContourSequence",
            "StructureSetROISequence",
        )
        if not hasattr(ds, sequence_name)
    ]
    sop_class_uid = str(getattr(ds, "SOPClassUID", ""))
    if sop_class_uid != _RTSTRUCT_SOP_CLASS_UID or missing_sequences:
        raise ValueError(
            f"{rtstruct_path.name} is not a valid RT Structure Set DICOM. "
            f"Modality={getattr(ds, 'Modality', None)!r}, "
            f"SOPClassUID={sop_class_uid!r}, "
            f"missing sequences={missing_sequences or 'none'}."
        )


def _ensure_rt_utils_compatible(rtstruct_path: Path) -> None:
    """Add optional metadata whose presence is required by ``rt-utils``."""
    ds = pydicom.dcmread(rtstruct_path)
    if not hasattr(ds, "RTROIObservationsSequence"):
        # This sequence is optional in an RT Structure Set and is not needed to
        # construct masks. rt-utils nevertheless requires the attribute to be
        # present when loading an existing RTSTRUCT.
        ds.RTROIObservationsSequence = Sequence()
        ds.save_as(rtstruct_path)


def _download_rtstruct_assets(project_base: Path) -> list[Path]:
    target_dir = project_base / "rtstructs"
    target_dir.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        _RTSTRUCT_ZENODO_RECORD_API_URL,
        headers={"User-Agent": _ZENODO_USER_AGENT},
    )
    with urllib.request.urlopen(
        request, timeout=_RTSTRUCT_DOWNLOAD_TIMEOUT_S
    ) as response:
        record = json.load(response)

    file_records = record.get("files", [])
    if not file_records:
        raise FileNotFoundError(
            f"Zenodo record {_RTSTRUCT_ZENODO_RECORD_ID} does not list any files."
        )

    files_by_name = {
        str(file_record.get("key", "")): file_record for file_record in file_records
    }
    missing = [
        filename
        for filename in _EXPECTED_RTSTRUCT_FILENAMES
        if filename not in files_by_name
    ]
    if missing:
        discovered = sorted(filename for filename in files_by_name if filename)
        raise FileNotFoundError(
            "Zenodo RTSTRUCT assets are incomplete. Missing: "
            + ", ".join(missing)
            + ". Discovered: "
            + (", ".join(discovered) if discovered else "none")
        )

    for filename in _EXPECTED_RTSTRUCT_FILENAMES:
        links = files_by_name[filename].get("links", {})
        download_url = (
            links.get("content") or links.get("download") or links.get("self")
        )
        if not download_url:
            raise FileNotFoundError(
                f"Zenodo file '{filename}' does not include a download link."
            )
        rtstruct_path = target_dir / filename
        _download_url_to_file(download_url, rtstruct_path)
        _validate_rtstruct_file(rtstruct_path)
        _ensure_rt_utils_compatible(rtstruct_path)

    return [target_dir / filename for filename in _EXPECTED_RTSTRUCT_FILENAMES]


def _load_reference_frame(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, index_col=0)


def _parse_numeric_list_cell(value: object) -> object:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return value

    inner = stripped[1:-1].strip()
    if not inner:
        return []

    cleaned = re.sub(r"np\.float64\(([^()]*)\)", r"\1", inner)
    return [float(item.strip()) for item in cleaned.split(",")]


def _normalize_sequence_cell(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return value


def _normalize_frame_cells(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in normalized.columns:
        if column in _LIST_LIKE_RESULTS_COLUMNS:
            normalized[column] = normalized[column].map(_parse_numeric_list_cell)
            normalized[column] = normalized[column].map(_normalize_sequence_cell)
    return normalized


def _assert_sequence_columns_close(
    actual: pd.DataFrame, expected: pd.DataFrame, *, rtol: float, atol: float
) -> None:
    for column in expected.columns:
        if column not in _LIST_LIKE_RESULTS_COLUMNS:
            continue
        for roi_name in expected.index:
            expected_value = expected.at[roi_name, column]
            actual_value = actual.at[roi_name, column]

            if isinstance(expected_value, list):
                if not isinstance(actual_value, list):
                    raise AssertionError(
                        f"Column '{column}' for ROI '{roi_name}' is not list-like in actual results."
                    )
                if len(actual_value) != len(expected_value):
                    raise AssertionError(
                        f"Column '{column}' for ROI '{roi_name}' has different lengths: "
                        f"{len(actual_value)} != {len(expected_value)}"
                    )
                if not np.allclose(actual_value, expected_value, rtol=rtol, atol=atol):
                    raise AssertionError(
                        f"Column '{column}' for ROI '{roi_name}' differs. "
                        f"Actual={actual_value}, Expected={expected_value}"
                    )


def _prepare_frame_for_comparison(
    actual: pd.DataFrame, expected: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_columns = [col for col in expected.columns if col not in actual.columns]
    if missing_columns:
        raise AssertionError(
            "Actual dataframe is missing expected columns: "
            + ", ".join(missing_columns)
        )

    actual_prepared = actual.loc[:, list(expected.columns)].copy()
    expected_prepared = expected.copy()

    actual_prepared.index = actual_prepared.index.map(str)
    expected_prepared.index = expected_prepared.index.map(str)

    missing_index = [
        idx for idx in expected_prepared.index if idx not in actual_prepared.index
    ]
    if missing_index:
        raise AssertionError(
            "Actual dataframe is missing expected index entries: "
            + ", ".join(missing_index)
        )

    actual_prepared = actual_prepared.loc[list(expected_prepared.index)]
    actual_prepared = _normalize_frame_cells(actual_prepared)
    expected_prepared = _normalize_frame_cells(expected_prepared)

    for column in expected_prepared.columns:
        if column in _LIST_LIKE_RESULTS_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(expected_prepared[column]):
            actual_prepared[column] = pd.to_numeric(actual_prepared[column])

    actual_prepared = actual_prepared.sort_index()
    expected_prepared = expected_prepared.sort_index()
    return actual_prepared, expected_prepared


@pytest.mark.integration
@pytest.mark.slow
def test_voxel_dosimetry_pipeline_matches_reference(tmp_path: Path) -> None:
    _skip_if_validation_reference_assets_missing()

    project_base = tmp_path / "snmmi_dosimetry_validation_project"
    init_project(project_base)
    _copy_optional_validation_configs(project_base)

    try:
        fetch_snmmi_dosimetry_challenge(data_home=str(project_base))
    except RuntimeError as exc:
        _skip_snmmi_fetch_failure_in_ci_or_fail_locally(
            f"SNMMI dosimetry dataset could not be fetched: {exc}"
        )

    try:
        downloaded_rtstruct_files = _download_rtstruct_assets(project_base)
    except (
        FileNotFoundError,
        OSError,
        urllib.error.URLError,
        ValueError,
    ) as exc:
        pytest.fail(
            "RTSTRUCT validation assets could not be fetched or validated: " f"{exc}"
        )

    study_info, ct_paths, spect_paths, rtstruct_files = (
        auto_setup_dosimetry_study_inventory(
            base_dir=project_base,
            patient_id=None,
        )
    )

    assert study_info.get("patient_id") is not None
    assert ct_paths, "No CT timepoints were discovered by the validation pipeline."
    assert (
        spect_paths
    ), "No SPECT timepoints were discovered by the validation pipeline."
    assert (
        rtstruct_files
    ), "No RTSTRUCT files were discovered by the validation pipeline."
    assert len(study_info["time_points"]) == len(downloaded_rtstruct_files)

    study_info["rtstruct_files"] = [str(path) for path in downloaded_rtstruct_files]
    for time_point, rtstruct_file in zip(
        study_info["time_points"], downloaded_rtstruct_files
    ):
        time_point["rtstruct_file"] = str(rtstruct_file)

    long_ct, long_spect, _, _ = create_studies_with_masks(
        patient_id=study_info["patient_id"],
        cycle_no=1,
        parallel=True,
        mapping_config=project_base / "voi_mappings_config.json",
        study_info=study_info,
    )

    roi_config = build_roi_fit_config(
        longSPECT=long_spect,
        config_path=project_base / "dosimetry_fit_defaults.json",
    )

    first_tp_injection = study_info["time_points"][0]["injection_info"]
    database_dir = project_base / "dosimetry_database"
    results_dir = project_base / "results"
    database_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    dosimetry_config = {
        "PatientID": study_info["patient_id"],
        "Cycle": 1,
        "DatabaseDir": str(database_dir),
        "results_path": str(results_dir),
        "VOIs": roi_config,
        "InjectionDate": first_tp_injection["injection_date"],
        "InjectionTime": first_tp_injection["injection_time"],
        "InjectedActivity": long_spect.meta[0].Injected_Activity_MBq,
        "Radionuclide": long_spect.meta[0].Radionuclide,
        "PatientWeight_g": first_tp_injection["patient_weight_g"],
        "Level": "Voxel",
        "Method": "Voxel-S-value",
        "ScaleDoseByDensity": False,
        "ReferenceTimePoint": 0,
    }

    dosimetry = VoxelSDosimetry(
        config=dosimetry_config,
        nm_data=long_spect,
        ct_data=long_ct,
    )
    dosimetry.compute_dose()

    expected_results = _load_reference_frame(_EXPECTED_RESULTS_CSV)
    expected_df_ad = _load_reference_frame(_EXPECTED_DF_AD_CSV)

    actual_results, expected_results = _prepare_frame_for_comparison(
        dosimetry.results, expected_results
    )
    actual_df_ad, expected_df_ad = _prepare_frame_for_comparison(
        dosimetry.df_ad, expected_df_ad
    )

    _assert_sequence_columns_close(
        actual_results,
        expected_results,
        rtol=1e-4,
        atol=1e-6,
    )
    assert_frame_equal(
        actual_results.drop(
            columns=list(_LIST_LIKE_RESULTS_COLUMNS & set(actual_results.columns))
        ),
        expected_results.drop(
            columns=list(_LIST_LIKE_RESULTS_COLUMNS & set(expected_results.columns))
        ),
        check_dtype=False,
        rtol=1e-4,
        atol=1e-6,
    )
    assert_frame_equal(
        actual_df_ad,
        expected_df_ad,
        check_dtype=False,
        rtol=1e-4,
        atol=1e-6,
    )
