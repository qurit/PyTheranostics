"""Dose voxel kernel module for convolution-based dosimetry."""

from __future__ import annotations

import csv
import logging
import re
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy
from scipy import signal

from pytheranostics.misc_tools.tools import hu_to_rho

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "voxel_kernels"
_ZENODO_RECORD_ID = "7596345"
_DEFAULT_CROP_SIZE_MM = 25.0
_CSV_PATTERN = re.compile(
    r"^(?P<isotope>[0-9]+[A-Za-z]+)"
    r"_(?P<half_life>[^_]+)"
    r"_(?P<energy_fraction>[^_]+)"
    r"_(?P<voxel_size>[0-9]+(?:\.[0-9]+)?)mm"
    r"_(?P<matrix_size>\d+)x(?P=matrix_size)x(?P=matrix_size)\.csv$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class KernelMetadata:
    """Metadata parsed from a dose kernel CSV filename."""

    path: Path
    isotope: str
    voxel_size_mm: float
    matrix_size: int


def _split_isotope(isotope: str) -> Tuple[str, str]:
    """Return isotope mass number and symbol from either `Lu177` or `177Lu`."""
    match = re.fullmatch(r"(?P<symbol>[A-Za-z]+)(?P<mass>\d+)", isotope)
    if match is None:
        match = re.fullmatch(r"(?P<mass>\d+)(?P<symbol>[A-Za-z]+)", isotope)

    if match is None:
        raise ValueError(
            "Isotope must be formatted like 'Lu177' or '177Lu'. "
            f"Received '{isotope}'."
        )

    return match.group("mass"), match.group("symbol")


def _canonical_isotope(isotope: str) -> str:
    """Convert an isotope string to a canonical representation for comparisons."""
    mass, symbol = _split_isotope(isotope)
    return f"{symbol.capitalize()}{mass}"


def _zenodo_isotope_name(isotope: str) -> str:
    """Convert an isotope string to Zenodo's archive naming convention."""
    mass, symbol = _split_isotope(isotope)
    return f"{mass}{symbol.capitalize()}"


def _discover_kernel_files(kernel_dir: Path, isotope: str) -> List[KernelMetadata]:
    """Return kernel CSVs matching the requested isotope."""
    requested_isotope = _canonical_isotope(isotope)
    kernels: List[KernelMetadata] = []

    for path in sorted(kernel_dir.rglob("*.csv")):
        match = _CSV_PATTERN.match(path.name)
        if match is None:
            continue

        parsed_isotope = _canonical_isotope(match.group("isotope"))
        if parsed_isotope != requested_isotope:
            continue

        kernels.append(
            KernelMetadata(
                path=path,
                isotope=parsed_isotope,
                voxel_size_mm=float(match.group("voxel_size")),
                matrix_size=int(match.group("matrix_size")),
            )
        )

    return kernels


def _select_closest_kernel(
    requested_voxel_size_mm: float, available_kernels: Sequence[KernelMetadata]
) -> KernelMetadata:
    """Return the kernel whose voxel size is closest to the requested size."""
    if not available_kernels:
        raise ValueError("No kernels were provided for voxel-size selection.")

    closest_kernel = min(
        available_kernels,
        key=lambda kernel: abs(kernel.voxel_size_mm - requested_voxel_size_mm),
    )
    delta_mm = abs(closest_kernel.voxel_size_mm - requested_voxel_size_mm)

    if delta_mm > 0.09:
        logger.warning(
            "Requested voxel size %.3f mm does not closely match an available "
            "kernel. Using %.4f mm instead.",
            requested_voxel_size_mm,
            closest_kernel.voxel_size_mm,
        )

    return closest_kernel


def _download_kernels_from_zenodo(isotope: str, kernel_dir: Path) -> None:
    """Download and extract dose-kernel CSVs for the requested isotope."""
    zenodo_name = _zenodo_isotope_name(isotope)
    url = (
        f"https://zenodo.org/records/{_ZENODO_RECORD_ID}/files/"
        f"{zenodo_name}.zip?download=1"
    )

    kernel_dir.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
                shutil.copyfileobj(response, tmp_file)
                archive_path = Path(tmp_file.name)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            logger.error(
                "No dose voxel kernel archive is available on Zenodo for isotope %s.",
                isotope,
            )
            raise NotImplementedError(
                f"Dose voxel kernels for isotope '{isotope}' are not available on Zenodo."
            ) from exc
        raise

    try:
        with zipfile.ZipFile(archive_path) as archive:
            csv_members = [
                member
                for member in archive.infolist()
                if member.filename.lower().endswith(".csv")
            ]
            if not csv_members:
                raise FileNotFoundError(
                    f"Downloaded archive for isotope '{isotope}' does not contain CSV kernels."
                )

            for member in csv_members:
                archive.extract(member, path=kernel_dir)
    finally:
        archive_path.unlink(missing_ok=True)

    logger.info("Downloaded dose voxel kernels for %s from %s", isotope, url)


def _ensure_kernel_files_available(
    kernel_dir: Path, isotope: str
) -> List[KernelMetadata]:
    """Ensure that kernel CSVs for the requested isotope are available locally."""
    kernels = _discover_kernel_files(kernel_dir=kernel_dir, isotope=isotope)
    if kernels:
        return kernels

    has_any_csv_kernels = any(
        _CSV_PATTERN.match(path.name) is not None for path in kernel_dir.rglob("*.csv")
    )
    if not has_any_csv_kernels:
        logger.info(
            "No kernel CSV files found under %s. Downloading %s kernels.",
            kernel_dir,
            isotope,
        )
    else:
        logger.info(
            "No local kernel CSVs found for %s under %s. Downloading isotope archive.",
            isotope,
            kernel_dir,
        )

    _download_kernels_from_zenodo(isotope=isotope, kernel_dir=kernel_dir)
    kernels = _discover_kernel_files(kernel_dir=kernel_dir, isotope=isotope)
    if not kernels:
        raise NotImplementedError(
            f"Dose voxel kernels for isotope '{isotope}' were not found after download."
        )

    return kernels


def _load_octant_kernel(csv_path: Path) -> numpy.ndarray:
    """Load the positive octant stored in a kernel CSV."""
    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = [
            row for row in csv.reader(file_obj) if any(cell.strip() for cell in row)
        ]

    if not rows:
        raise ValueError(f"Kernel CSV '{csv_path}' is empty.")

    section_size = len(rows[0])
    if section_size == 0:
        raise ValueError(f"Kernel CSV '{csv_path}' does not contain numeric data.")

    if any(len(row) != section_size for row in rows):
        raise ValueError(f"Kernel CSV '{csv_path}' has inconsistent row lengths.")

    if len(rows) % section_size != 0:
        raise ValueError(
            f"Kernel CSV '{csv_path}' cannot be reshaped into stacked square sections."
        )

    num_sections = len(rows) // section_size
    octant = numpy.empty(
        (section_size, section_size, num_sections), dtype=numpy.float64
    )

    for section_idx in range(num_sections):
        start = section_idx * section_size
        stop = start + section_size
        block = numpy.asarray(rows[start:stop], dtype=numpy.float64)
        octant[:, :, section_idx] = block.T

    return octant


def _mirror_axis_from_center(half_kernel: numpy.ndarray, axis: int) -> numpy.ndarray:
    """Mirror a kernel axis where index 0 is the center voxel."""
    slicer = [slice(None)] * half_kernel.ndim
    slicer[axis] = slice(1, None)
    positive_side = half_kernel[tuple(slicer)]
    mirrored_side = numpy.flip(positive_side, axis=axis)
    return numpy.concatenate((mirrored_side, half_kernel), axis=axis)


def _build_full_kernel_from_octant(octant: numpy.ndarray) -> numpy.ndarray:
    """Expand the stored positive octant into a full 3-D kernel."""
    kernel = octant
    for axis in range(3):
        kernel = _mirror_axis_from_center(kernel, axis=axis)
    return kernel


def _odd_voxel_count_for_physical_size(size_mm: float, voxel_size_mm: float) -> int:
    """Return the odd voxel count closest to a requested physical crop size."""
    if size_mm <= 0:
        raise ValueError("Crop size must be positive when cropping is enabled.")

    target_voxels = max(1.0, size_mm / voxel_size_mm)
    rounded_voxels = max(1, int(round(target_voxels)))
    if rounded_voxels % 2 == 1:
        return rounded_voxels

    lower = max(1, rounded_voxels - 1)
    upper = rounded_voxels + 1
    if abs(lower - target_voxels) <= abs(upper - target_voxels):
        return lower
    return upper


def _crop_kernel_around_center(
    kernel: numpy.ndarray, voxel_size_mm: float, crop_size_mm: Optional[float]
) -> numpy.ndarray:
    """Crop a kernel symmetrically around its center voxel."""
    if crop_size_mm is None:
        return kernel

    target_voxels = _odd_voxel_count_for_physical_size(crop_size_mm, voxel_size_mm)
    if target_voxels >= kernel.shape[0]:
        return kernel

    center = kernel.shape[0] // 2
    half_width = target_voxels // 2
    start = center - half_width
    stop = center + half_width + 1
    return kernel[start:stop, start:stop, start:stop]


class DoseVoxelKernel:
    """Dose Voxel Kernel for convolution-based dosimetry calculations."""

    def __init__(
        self,
        isotope: str,
        voxel_size_mm: float,
        crop_kernel_size_mm: Optional[float] = _DEFAULT_CROP_SIZE_MM,
    ) -> None:
        """Initialize the DoseVoxelKernel.

        Args
        ----
            isotope (str): The isotope name (e.g., 'Lu177').
            voxel_size_mm (float): Requested voxel size in millimeters.
            crop_kernel_size_mm (float | None): Physical cubic crop size in mm.
                Use ``None`` to disable cropping.
        """
        available_kernels = _ensure_kernel_files_available(_DATA_DIR, isotope)
        selected_kernel = _select_closest_kernel(
            requested_voxel_size_mm=voxel_size_mm,
            available_kernels=available_kernels,
        )

        octant = _load_octant_kernel(selected_kernel.path)
        full_kernel = _build_full_kernel_from_octant(octant)

        expected_shape = (
            selected_kernel.matrix_size,
            selected_kernel.matrix_size,
            selected_kernel.matrix_size,
        )
        if full_kernel.shape != expected_shape:
            raise ValueError(
                "Kernel shape reconstructed from CSV does not match the filename "
                f"metadata for '{selected_kernel.path.name}'."
            )

        self.kernel = _crop_kernel_around_center(
            kernel=full_kernel,
            voxel_size_mm=selected_kernel.voxel_size_mm,
            crop_size_mm=crop_kernel_size_mm,
        ).astype(numpy.float64)
        self.voxel_size_mm = float(selected_kernel.voxel_size_mm)
        self.matrix_size = int(selected_kernel.matrix_size)
        self.isotope = selected_kernel.isotope

    def tia_to_dose(
        self, tia_mbq_s: numpy.ndarray, ct: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        """Convert Time-Integrated Activity to dose.

        Parameters
        ----------
        tia_mbq_s : numpy.ndarray
            Time-integrated activity in MBq*s.
        ct : numpy.ndarray, optional
            CT image in HU for density weighting.

        Returns
        -------
        numpy.ndarray
            Dose map in mGy.
        """
        dose_mGy = signal.fftconvolve(tia_mbq_s, self.kernel, mode="same", axes=None)

        if ct is not None:
            logger.warning(
                "Scaling dose by density will yield erroneous dose values in very "
                "low-density voxels (for example air inside the body). Please use "
                "at your own risk."
            )
            dose_mGy = self.weight_dose_by_density(dose_map=dose_mGy, ct=ct)

        return dose_mGy

    def weight_dose_by_density(
        self, dose_map: numpy.ndarray, ct: numpy.ndarray
    ) -> numpy.ndarray:
        """Scale dose per voxel by voxel density.

        This is only valid for voxels of density similar to that of soft tissue and will also improve results for voxels
        with higher density of soft tissue in some instances. However, it will over-estimate doses in voxels with lower density than soft tissue.
        To prevent dose to shoot-up in areas of air where there is activity present (e.g., in the patient's gut), we do not apply scaling based on density in those voxels (i.e., we apply a factor of 1, which is equivalent to saying
        the tissue is ~ soft tissue).

        Args:
            dose_map (numpy.ndarray): Dose-map obtained from convolution of TIA map and Dose Kernel.
            ct (numpy.ndarray): CT image, in HU.

        Returns
        -------
        numpy.ndarray
            Modified Dose-map with dose per voxel scaled-up by density.
        """
        return 1 / hu_to_rho(hu=numpy.clip(ct, 0, 99999)) * dose_map
