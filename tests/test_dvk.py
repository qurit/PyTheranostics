"""Tests for DoseVoxelKernel CSV loading and selection."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import pytheranostics.dosimetry.dvk as dvk
from pytheranostics.dosimetry.dvk import (
    DoseVoxelKernel,
    KernelMetadata,
    _build_full_kernel_from_octant,
    _crop_kernel_around_center,
    _select_closest_kernel,
)


def _write_kernel_csv(path: Path, octant: np.ndarray) -> None:
    """Write a synthetic kernel octant using the Zenodo CSV layout."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file_obj:
        for section_idx in range(octant.shape[2]):
            block = octant[:, :, section_idx].T
            for row in block:
                file_obj.write(",".join(f"{value:.6f}" for value in row))
                file_obj.write("\n")

            if section_idx != octant.shape[2] - 1:
                file_obj.write(",".join([""] * octant.shape[1]))
                file_obj.write("\n")


def test_select_closest_kernel_warns_for_large_mismatch(
    caplog: pytest.LogCaptureFixture,
):
    """A warning should be emitted when the requested voxel size is far from a match."""
    kernels = [
        KernelMetadata(Path("a.csv"), "Lu177", 4.7952, 173),
        KernelMetadata(Path("b.csv"), "Lu177", 10.0, 83),
    ]

    with caplog.at_level("WARNING"):
        selected = _select_closest_kernel(4.5, kernels)

    assert selected.voxel_size_mm == pytest.approx(4.7952)
    assert "does not closely match" in caplog.text


def test_dose_voxel_kernel_reconstructs_octant_and_crops_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """DoseVoxelKernel should rebuild the full kernel and crop it near 25 mm."""
    kernel_dir = tmp_path / "voxel_kernels"
    octant = np.arange(1, 65, dtype=np.float64).reshape(4, 4, 4)
    kernel_path = kernel_dir / "177Lu" / "177LU_6.647D_99%_10mm_7x7x7.csv"
    _write_kernel_csv(kernel_path, octant)

    monkeypatch.setattr(dvk, "_DATA_DIR", kernel_dir)

    kernel = DoseVoxelKernel(isotope="Lu177", voxel_size_mm=10.0)
    expected_full = _build_full_kernel_from_octant(octant)
    expected_cropped = _crop_kernel_around_center(expected_full, 10.0, 25.0)

    assert kernel.kernel.shape == (3, 3, 3)
    assert np.array_equal(kernel.kernel, expected_cropped)
    assert kernel.kernel[1, 1, 1] == pytest.approx(octant[0, 0, 0])
    assert np.array_equal(kernel.kernel, np.flip(kernel.kernel, axis=0))
    assert np.array_equal(kernel.kernel, np.flip(kernel.kernel, axis=1))
    assert np.array_equal(kernel.kernel, np.flip(kernel.kernel, axis=2))


def test_dose_voxel_kernel_downloads_when_no_csv_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The loader should invoke the download helper when no CSV kernels are present."""
    kernel_dir = tmp_path / "voxel_kernels"
    octant = np.arange(1, 9, dtype=np.float64).reshape(2, 2, 2)
    created = {"called": False}

    def _fake_download(isotope: str, kernel_dir_arg: Path) -> None:
        created["called"] = True
        assert isotope == "Lu177"
        _write_kernel_csv(
            kernel_dir_arg / "177Lu" / "177LU_6.647D_99%_5mm_3x3x3.csv",
            octant,
        )

    monkeypatch.setattr(dvk, "_DATA_DIR", kernel_dir)
    monkeypatch.setattr(dvk, "_download_kernels_from_zenodo", _fake_download)

    kernel = DoseVoxelKernel(
        isotope="Lu177", voxel_size_mm=5.0, crop_kernel_size_mm=None
    )

    assert created["called"] is True
    assert kernel.kernel.shape == (3, 3, 3)


def test_kernel_download_retries_truncated_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A truncated Zenodo response should be retried before ZIP extraction."""
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w") as archive:
        archive.writestr(
            "177Lu/177LU_6.647D_99%_5mm_3x3x3.csv",
            "1,2\n3,4\n",
        )
    archive_bytes = archive_buffer.getvalue()
    checksum = hashlib.md5(archive_bytes).hexdigest()
    record = {
        "files": [
            {
                "key": "177Lu.zip",
                "size": len(archive_bytes),
                "checksum": f"md5:{checksum}",
                "links": {"self": "https://example.test/177Lu.zip"},
            }
        ]
    }
    responses = [
        json.dumps(record).encode(),
        archive_bytes[: len(archive_bytes) // 2],
        archive_bytes,
    ]

    def fake_urlopen(request, timeout):
        assert timeout == dvk._ZENODO_DOWNLOAD_TIMEOUT_S
        return io.BytesIO(responses.pop(0))

    monkeypatch.setattr(dvk.urllib.request, "urlopen", fake_urlopen)

    kernel_dir = tmp_path / "voxel_kernels"
    dvk._download_kernels_from_zenodo("Lu177", kernel_dir)

    assert not responses
    assert (kernel_dir / "177Lu" / "177LU_6.647D_99%_5mm_3x3x3.csv").is_file()
