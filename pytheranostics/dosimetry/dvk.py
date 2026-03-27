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
    """Metadata parsed from a dose-kernel CSV filename.

    Attributes
    ----------
    path : Path
        Path to the CSV file on disk.
    isotope : str
        Canonical isotope label, for example ``"Lu177"``.
    voxel_size_mm : float
        Isotropic voxel size represented by the kernel, in millimeters.
    matrix_size : int
        Edge length of the cubic full kernel matrix.
    """

    path: Path
    isotope: str
    voxel_size_mm: float
    matrix_size: int


def _split_isotope(isotope: str) -> Tuple[str, str]:
    """Split an isotope label into mass number and element symbol.

    Parameters
    ----------
    isotope : str
        Isotope label formatted as ``"Lu177"`` or ``"177Lu"``.

    Returns
    -------
    tuple of str
        Mass number followed by the element symbol.

    Raises
    ------
    ValueError
        If ``isotope`` does not match one of the supported label formats.
    """
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
    """Convert an isotope label to the canonical comparison format.

    Parameters
    ----------
    isotope : str
        Isotope label formatted as ``"Lu177"`` or ``"177Lu"``.

    Returns
    -------
    str
        Canonical label with the element symbol first, for example ``"Lu177"``.
    """
    mass, symbol = _split_isotope(isotope)
    return f"{symbol.capitalize()}{mass}"


def _zenodo_isotope_name(isotope: str) -> str:
    """Convert an isotope label to Zenodo's archive naming convention.

    Parameters
    ----------
    isotope : str
        Isotope label formatted as ``"Lu177"`` or ``"177Lu"``.

    Returns
    -------
    str
        Archive label with the mass number first, for example ``"177Lu"``.
    """
    mass, symbol = _split_isotope(isotope)
    return f"{mass}{symbol.capitalize()}"


def _discover_kernel_files(kernel_dir: Path, isotope: str) -> List[KernelMetadata]:
    """Discover local kernel CSV files matching an isotope.

    Parameters
    ----------
    kernel_dir : Path
        Root directory searched recursively for kernel CSV files.
    isotope : str
        Requested isotope label.

    Returns
    -------
    list of KernelMetadata
        Parsed metadata for matching kernel files.
    """
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
    """Select the available kernel whose voxel size is closest to the request.

    Parameters
    ----------
    requested_voxel_size_mm : float
        Requested isotropic voxel size in millimeters.
    available_kernels : sequence of KernelMetadata
        Candidate kernels for the isotope of interest.

    Returns
    -------
    KernelMetadata
        Metadata for the closest matching kernel.

    Raises
    ------
    ValueError
        If ``available_kernels`` is empty.
    """
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
    """Download and extract dose-kernel CSV files for an isotope.

    Parameters
    ----------
    isotope : str
        Requested isotope label.
    kernel_dir : Path
        Destination directory where the downloaded archive is extracted.

    Raises
    ------
    NotImplementedError
        If no Zenodo archive exists for the requested isotope.
    FileNotFoundError
        If the downloaded archive does not contain any CSV kernel files.
    urllib.error.URLError
        If the archive download fails for a network-related reason.
    """
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
    """Ensure that kernel CSV files for an isotope are available locally.

    Parameters
    ----------
    kernel_dir : Path
        Root directory containing local kernel files.
    isotope : str
        Requested isotope label.

    Returns
    -------
    list of KernelMetadata
        Available kernels for the requested isotope after local discovery and any
        required download.

    Raises
    ------
    NotImplementedError
        If matching kernels are still unavailable after the download attempt.
    """
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

    _download_kernels_from_zenodo(isotope, kernel_dir)
    kernels = _discover_kernel_files(kernel_dir=kernel_dir, isotope=isotope)
    if not kernels:
        raise NotImplementedError(
            f"Dose voxel kernels for isotope '{isotope}' were not found after download."
        )

    return kernels


def _load_octant_kernel(csv_path: Path) -> numpy.ndarray:
    """Load a stored positive octant from a kernel CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to a kernel CSV file using the stacked-square Zenodo layout.

    Returns
    -------
    numpy.ndarray
        Three-dimensional positive octant of the kernel.

    Raises
    ------
    ValueError
        If the CSV file is empty, malformed, or cannot be reshaped into stacked
        square sections.
    """
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
    """Mirror a kernel axis around the center voxel.

    Parameters
    ----------
    half_kernel : numpy.ndarray
        Kernel array where index 0 along ``axis`` corresponds to the center voxel.
    axis : int
        Axis to mirror.

    Returns
    -------
    numpy.ndarray
        Kernel with the requested axis mirrored onto the negative side.
    """
    slicer = [slice(None)] * half_kernel.ndim
    slicer[axis] = slice(1, None)
    positive_side = half_kernel[tuple(slicer)]
    mirrored_side = numpy.flip(positive_side, axis=axis)
    return numpy.concatenate((mirrored_side, half_kernel), axis=axis)


def _build_full_kernel_from_octant(octant: numpy.ndarray) -> numpy.ndarray:
    """Expand a stored positive octant into a full three-dimensional kernel.

    Parameters
    ----------
    octant : numpy.ndarray
        Positive octant whose index 0 corresponds to the kernel center along each
        axis.

    Returns
    -------
    numpy.ndarray
        Full symmetric three-dimensional kernel.
    """
    kernel = octant
    for axis in range(3):
        kernel = _mirror_axis_from_center(kernel, axis=axis)
    return kernel


def _odd_voxel_count_for_physical_size(size_mm: float, voxel_size_mm: float) -> int:
    """Return the nearest odd voxel count for a requested physical size.

    Parameters
    ----------
    size_mm : float
        Requested physical size in millimeters.
    voxel_size_mm : float
        Isotropic voxel size in millimeters.

    Returns
    -------
    int
        Odd voxel count closest to the requested physical extent.

    Raises
    ------
    ValueError
        If ``size_mm`` is not strictly positive.
    """
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
    """Crop a kernel symmetrically around its center voxel.

    Parameters
    ----------
    kernel : numpy.ndarray
        Full three-dimensional dose kernel.
    voxel_size_mm : float
        Isotropic voxel size represented by ``kernel``.
    crop_size_mm : float, optional
        Requested physical crop size in millimeters. If ``None``, no cropping is
        applied.

    Returns
    -------
    numpy.ndarray
        Cropped kernel, or the original kernel when no crop is needed.
    """
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
    """Dose voxel kernel used for convolution-based dosimetry calculations.

    Parameters
    ----------
    isotope : str
        Requested isotope label.
    voxel_size_mm : float
        Requested isotropic voxel size in millimeters.
    crop_kernel_size_mm : float, optional
        Physical crop size of the cubic kernel in millimeters. Pass ``None`` to
        keep the full kernel.

    Attributes
    ----------
    kernel : numpy.ndarray
        Loaded full or cropped dose kernel.
    voxel_size_mm : float
        Voxel size of the selected kernel in millimeters.
    matrix_size : int
        Matrix size of the selected full kernel before optional cropping.
    isotope : str
        Canonical isotope label of the selected kernel.
    """

    def __init__(
        self,
        isotope: str,
        voxel_size_mm: float,
        crop_kernel_size_mm: Optional[float] = _DEFAULT_CROP_SIZE_MM,
    ) -> None:
        """Initialize a dose voxel kernel from local or downloaded CSV files.

        Parameters
        ----------
        isotope : str
            Requested isotope label, for example ``"Lu177"``.
        voxel_size_mm : float
            Requested isotropic voxel size in millimeters.
        crop_kernel_size_mm : float, optional
            Physical cubic crop size in millimeters. Pass ``None`` to disable
            cropping.

        Raises
        ------
        ValueError
            If the reconstructed kernel shape does not match the filename metadata.
        NotImplementedError
            If no kernel files can be found or downloaded for ``isotope``.
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
                "Applying local mass-density reweighting to a homogeneous-medium dose "
                "kernel result. This is not a full heterogeneity correction and is "
                "only a rough approximation in soft tissue."
            )
            dose_mGy = self.weight_dose_by_density(dose_map=dose_mGy, ct=ct)

        return dose_mGy

    def weight_dose_by_density(
        self, dose_map: numpy.ndarray, ct: numpy.ndarray
    ) -> numpy.ndarray:
        """Apply local mass-density reweighting to a dose map.

        This method does not perform a true heterogeneity correction. It assumes the
        dose kernel was computed in a homogeneous medium close to soft tissue and
        rescales the resulting dose voxel-by-voxel by ``1 / rho`` using CT-derived
        density. As a result, it should be treated only as a rough local mass
        correction in near-water soft tissues.

        Methodological limitations:
        - It uses only the local density of the target voxel and ignores transport
          effects from surrounding tissues.
        - It does not account for material composition differences beyond the
          HU-to-density mapping.
        - Negative HU values are clipped to 0 HU before conversion so air and other
          low-density voxels are effectively treated as water-equivalent for this
          scaling step. This avoids unphysical dose blow-up but is not physically
          rigorous.

        Parameters
        ----------
        dose_map : numpy.ndarray
            Dose map obtained from convolution of the TIA map with a homogeneous-
            medium dose kernel.
        ct : numpy.ndarray
            CT image in HU, sampled on the same grid as ``dose_map``.

        Returns
        -------
        numpy.ndarray
            Dose map after local density reweighting.

        Raises
        ------
        ValueError
            If ``dose_map`` and ``ct`` do not share the same shape.
        """
        if dose_map.shape != ct.shape:
            raise ValueError(
                "Dose-map and CT array must have the same shape for density weighting. "
                f"Got {dose_map.shape} and {ct.shape}."
            )

        ct_clipped = numpy.clip(ct.astype(numpy.float64), 0.0, None)
        rho = hu_to_rho(hu=ct_clipped)

        logger.info(
            "Applying local density reweighting with HU clipped to [0, inf) before "
            "HU-to-density conversion."
        )

        return dose_map / rho
