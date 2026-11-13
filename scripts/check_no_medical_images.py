#!/usr/bin/env python3
"""Block medical image data from entering the repository.

Patient imaging must never be committed to this repository. Sample data
belongs in an external, de-identified dataset (for example on Zenodo) that the
tests and tutorials download at run time.

This guard is deliberately content-based as well as name-based: renaming a
DICOM file to ``.bin`` does not get it past the magic-byte check.

Usage::

    check_no_medical_images.py FILE [FILE ...]   # pre-commit passes staged files
    check_no_medical_images.py --all             # scan every tracked file

Exits non-zero and lists the offending paths if anything is found.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Extensions that are always medical image data.
DICOM_SUFFIXES = {".dcm", ".dicom", ".ima"}

# Volumetric formats that commonly carry patient data. Allowed only under
# ALLOWED_PREFIXES, which is for non-patient reference data such as the ICRP
# computational phantom masks.
VOLUME_SUFFIXES = {".nii", ".nrrd", ".nhdr", ".mha", ".mhd"}

# Paths permitted to contain volumetric reference data (NOT DICOM, which is
# never allowed anywhere). Keep this list short and justified.
ALLOWED_PREFIXES = ("pytheranostics/data/phantom/",)

LFS_POINTER_MAGIC = b"version https://git-lfs"


def _suffixes(path: Path) -> set[str]:
    """Return lowercase suffixes, treating ``.nii.gz`` as ``.nii``."""
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[: -len(".gz")]
    return {Path(name).suffix}


def is_allowed(rel_path: str) -> bool:
    """Whether this path is explicitly permitted to hold volumetric data."""
    posix = Path(rel_path).as_posix()
    return any(posix.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def looks_like_dicom(path: Path) -> bool:
    """Detect DICOM by content, regardless of the file's name."""
    try:
        with path.open("rb") as handle:
            head = handle.read(132)
    except OSError:
        return False

    # Standard DICOM: 128-byte preamble followed by the 'DICM' magic.
    if len(head) >= 132 and head[128:132] == b"DICM":
        return True

    # Preamble-less DICOM stream: tag (0002,0000) with explicit VR 'UL'.
    if head[:4] == b"\x02\x00\x00\x00" and head[4:6] == b"UL":
        return True

    return False


def is_lfs_pointer(path: Path) -> bool:
    """Whether the file is a Git LFS pointer rather than real content."""
    try:
        with path.open("rb") as handle:
            return handle.read(len(LFS_POINTER_MAGIC)) == LFS_POINTER_MAGIC
    except OSError:
        return False


def tracked_files() -> list[str]:
    """Every file tracked by git."""
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    return [p for p in out.decode().split("\0") if p]


def check(paths: list[str]) -> list[tuple[str, str]]:
    """Return a list of (path, reason) for everything that must not be here."""
    problems: list[tuple[str, str]] = []

    for rel in paths:
        path = Path(rel)
        if not path.is_file():
            continue

        suffixes = _suffixes(path)
        pointer = is_lfs_pointer(path)

        if suffixes & DICOM_SUFFIXES:
            reason = "DICOM file extension"
            if pointer:
                reason += " (Git LFS pointer — the image data is in LFS storage)"
            problems.append((rel, reason))
            continue

        if suffixes & VOLUME_SUFFIXES and not is_allowed(rel):
            reason = f"medical volume format ({sorted(suffixes)[0]})"
            if pointer:
                reason += " (Git LFS pointer)"
            problems.append((rel, reason))
            continue

        # Content check last: catches a DICOM renamed to something innocuous.
        if not pointer and looks_like_dicom(path):
            problems.append((rel, "DICOM content (magic bytes), despite its name"))

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="files to check")
    parser.add_argument(
        "--all", action="store_true", help="check every tracked file instead"
    )
    args = parser.parse_args()

    paths = tracked_files() if args.all else args.files
    if not paths:
        return 0

    problems = check(paths)
    if not problems:
        return 0

    print("ERROR: medical image data must not be committed to this repository.\n")
    for rel, reason in problems:
        print(f"  {rel}\n      -> {reason}")
    print(
        "\nPatient imaging must never enter git history: it cannot be fully\n"
        "removed afterwards, because GitHub keeps pull-request refs and LFS\n"
        "objects that a history rewrite does not touch.\n\n"
        "Put sample data in an external, de-identified dataset (e.g. Zenodo)\n"
        "and download it at run time, as tests/test_voxel_dosimetry_validation.py\n"
        "already does.\n\n"
        "If a file is genuinely non-patient reference data, add its directory to\n"
        "ALLOWED_PREFIXES in scripts/check_no_medical_images.py, with a comment\n"
        "saying why."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
