"""Module for longitudinal medical imaging studies."""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy
import SimpleITK
from numpy.typing import NDArray

from pytheranostics.imaging_ds.metadata import ImagingMetadata
from pytheranostics.imaging_tools.tools import (
    itk_image_from_array,
    jaccard_index,
    load_from_dicom_dir,
    resample_mask_to_target,
)
from pytheranostics.registration.phantom_to_ct import PhantomToCTBoneReg

logger = logging.getLogger(__name__)


class LongitudinalStudy:
    """Longitudinal Study Data Class.

    Holds multiple medical imaging datasets, alongside with masks for organs/regions
    of interest and meta-data.
    """

    # Cached valid organ names loaded from config
    _VALID_ORGAN_NAMES = None

    @classmethod
    def _get_valid_organ_names(cls) -> List[str]:
        """Get valid organ names from config file.

        Searches for voi_mappings_config.json in order:
        1. Current directory (project-specific config)
        2. One level up (project root)
        3. Package template (OLINDA-compatible defaults)

        Returns
        -------
        List[str]
            List of valid organ names.

        Raises
        ------
        FileNotFoundError
            If no config file can be found.
        ValueError
            If config file doesn't contain valid_organ_names.
        """
        if cls._VALID_ORGAN_NAMES is not None:
            return cls._VALID_ORGAN_NAMES

        # Try project-specific configs first
        search_paths = [
            Path.cwd() / "voi_mappings_config.json",
            Path.cwd().parent / "voi_mappings_config.json",
        ]

        for config_path in search_paths:
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        if "valid_organ_names" in config:
                            # Handle both old format (list) and new format (dict with names key)
                            organ_names = config["valid_organ_names"]
                            if isinstance(organ_names, dict):
                                cls._VALID_ORGAN_NAMES = organ_names.get("names", [])
                            else:
                                cls._VALID_ORGAN_NAMES = organ_names
                            return cls._VALID_ORGAN_NAMES
                except Exception:
                    continue

        # Load from package template (OLINDA defaults)
        try:
            import importlib.resources as pkg_resources

            template_path = pkg_resources.files("pytheranostics.data").joinpath(
                "configuration_templates/voi_mappings_config.json"
            )
            with open(template_path, "r") as f:
                config = json.load(f)
                if "valid_organ_names" in config:
                    organ_names = config["valid_organ_names"]
                    if isinstance(organ_names, dict):
                        cls._VALID_ORGAN_NAMES = organ_names.get("names", [])
                    else:
                        cls._VALID_ORGAN_NAMES = organ_names
                    return cls._VALID_ORGAN_NAMES
        except Exception as e:
            raise FileNotFoundError(
                "Could not load valid_organ_names from any config file. "
                "Please ensure voi_mappings_config.json exists in your project or package."
            ) from e

        raise ValueError(
            "Config file found but does not contain 'valid_organ_names' section."
        )

    def __init__(
        self,
        images: Dict[int, SimpleITK.Image],
        meta: Dict[int, ImagingMetadata],
        modality: str = "NM",
    ) -> None:
        """Initialize a LongitudinalStudy instance.

        Args:
            images (Dict[int, SimpleITK.Image]): Dictionary of (time-point ID, SimpleITK.Image)
                representing CT or quantitative nuclear medicine images for each time point
                in the longitudinal study.
            meta (Dict[int, ImagingMetadata]): Dictionary of (time-point ID, ImagingMetadata)
                representing metadata for each time point, containing acquisition details
                and radionuclide information.
            modality (str, optional): The imaging modality type. Supported values are "NM"
                (Nuclear Medicine), "PT" (PET), "CT", or "DOSE". Defaults to "NM".

        Raises
        ------
        ValueError
            If the specified modality is not one of the supported values:
                "NM", "PT", "CT", or "DOSE".

        Note
        ----
            The constructor initializes an empty masks dictionary that can be populated later
            using the `add_masks_to_time_point` method. It also defines a comprehensive list
            of valid mask names for regions of interest including organs, glands, and lesions.
        """
        if images.keys() != meta.keys():
            raise ValueError(
                "Not all time points have corresponding images and metadata."
            )

        # TODO Consistency checks: verify that there are no missing masks across time points.
        # NOTE: Such consistency would involve running add_mask_to_time_point() in __init__

        if modality not in ["NM", "PT", "CT", "DOSE"]:
            raise ValueError(f"Modality {modality} is not supported.")

        self.modality = modality
        self.images = images
        self.meta = meta
        self.masks: Dict[int, Dict[str, NDArray[numpy.bool_]]] = (
            {}
        )  # {time_id: {mask_name: array}}

        return None

    @classmethod
    def from_dicom(
        cls,
        dicom_dirs: List[str],
        modality: str = "CT",
        calibration_factor: Optional[float] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> "LongitudinalStudy":
        """Create a LongitudinalStudy object from a list of DICOM directories.

        Currently assumes the order of the list corresponds to the order of the time points.

        Args:
            dicom_dirs (List[str]): List of paths to DICOM directories, each containing
                images for one time point in the longitudinal study.
            modality (str, optional): The imaging modality. Supported values are "CT"
                and "Lu177_SPECT". Defaults to "CT".
            calibration_factor (float, optional): Converts reconstructed SPECT image
                (raw counts * num_proj) to units of Bq/mL. Defaults to None.
            parallel (bool, optional): Whether to load DICOM directories in parallel.
                Defaults to True for faster loading of multiple timepoints.
            max_workers (int, optional): Maximum number of parallel workers. If None,
                defaults to min(number of CPUs, number of directories).

        Returns
        -------
        LongitudinalStudy
            A new LongitudinalStudy instance containing the loaded
                DICOM data organized by time points.

        Raises
        ------
        ValueError
            If the specified modality is not supported.
        """
        # TODO: should fix this to make it robust and look at dicom header info for sorting time-points.
        supported_modalities = {
            "CT": "CT",
            "Lu177_SPECT": "NM",
        }
        if modality not in supported_modalities.keys():
            raise ValueError(
                f"Modality '{modality}' not supported. Currently, the following modalities are supported: {list(supported_modalities.keys())}"
            )
        internal_modality = supported_modalities[modality]

        images: Dict[int, SimpleITK.Image] = {}
        metadata: Dict[int, ImagingMetadata] = {}

        if parallel and len(dicom_dirs) > 1:
            # Parallel loading for multiple timepoints
            logger.info(
                f"Loading {len(dicom_dirs)} {modality} timepoints in parallel..."
            )

            # Helper function for parallel execution
            def load_single_timepoint(args):
                time_id, dicom_dir = args
                logger.debug(
                    f"  Loading timepoint {time_id} from {Path(dicom_dir).name}..."
                )
                return time_id, load_from_dicom_dir(
                    dir=dicom_dir,
                    modality=modality,
                    calibration_factor=calibration_factor,
                )

            # Use ThreadPoolExecutor for I/O-bound DICOM loading
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        load_single_timepoint, (time_id, dicom_dir)
                    ): time_id
                    for time_id, dicom_dir in enumerate(dicom_dirs)
                }

                for future in as_completed(futures):
                    time_id, (image, meta) = future.result()
                    images[time_id] = image
                    metadata[time_id] = meta
                    logger.debug(f"  ✓ Timepoint {time_id} loaded")
        else:
            # Sequential loading
            for time_id, dicom_dir in enumerate(dicom_dirs):
                logger.info(
                    f"Loading timepoint {time_id} from {Path(dicom_dir).name}..."
                )
                image, meta = load_from_dicom_dir(
                    dir=dicom_dir,
                    modality=modality,
                    calibration_factor=calibration_factor,
                )
                images[time_id] = image
                metadata[time_id] = meta

        return cls(
            images=images,
            meta=metadata,
            modality=internal_modality,
        )

    @staticmethod
    def _is_valid_mask_name(mask_name: str) -> bool:
        """Check if a mask name is valid.

        Valid names are either:
        - Standard organ names from config or default list
        - Lesion names in format 'Lesion_N' where N is a positive integer
        """
        valid_names = LongitudinalStudy._get_valid_organ_names()
        if mask_name in valid_names:
            return True
        lesion_pattern = r"^Lesion_([1-9]\d*)$"
        return bool(re.match(lesion_pattern, mask_name))

    # --- ROI name normalization & mapping helpers -----------------------------------------

    @staticmethod
    def canonical_roi_name(name: str) -> str:
        """Return a best-effort canonical ROI name for pyTheranostics/Olinda.

        This performs lightweight normalization of common suffixes and synonyms, keeping
        unknown names as-is so users can decide later.

        Rules applied:
        - Drop training suffixes like "_m" (morphology/CT) and "_a" (activity/NM)
        - Map frequent abbreviations to long-form organ names used across the codebase
        """
        base = name
        if base.endswith("_m") or base.endswith("_a"):
            base = base[:-2]

        synonyms = {
            "Kidney_L": "Kidney_Left",
            "Kidney_R": "Kidney_Right",
            "Parotid_L": "ParotidGland_Left",
            "Parotid_R": "ParotidGland_Right",
            "Submandibular_L": "SubmandibularGland_Left",
            "Submandibular_R": "SubmandibularGland_Right",
            "WBCT": "WholeBody",
            "WB": "WholeBody",
        }
        return synonyms.get(base, base)

    @classmethod
    def propose_mapping_from_names(cls, names: Iterable[str]) -> Dict[str, str]:
        """Propose a mapping from raw ROI names to canonical targets.

        Parameters
        ----------
        names : Iterable[str]
            Collection of raw ROI names (e.g., as found in RTSTRUCT files).

        Returns
        -------
        Dict[str, str]
            Proposed mapping {raw_name: canonical_name} using lightweight rules.
        """
        return {n: cls.canonical_roi_name(n) for n in set(names)}

    @classmethod
    def propose_mapping_from_studies(
        cls, studies: Iterable["LongitudinalStudy"]
    ) -> Dict[str, str]:
        """Propose a mapping from all mask names found across multiple studies.

        Parameters
        ----------
        studies : Iterable[LongitudinalStudy]
            One or more LongitudinalStudy instances (e.g., SPECT and CT).

        Returns
        -------
        Dict[str, str]
            Proposed mapping {raw_name: canonical_name} across all timepoints.
        """
        raw: set[str] = set()
        for study in studies:
            for _, masks in study.masks.items():
                raw.update(masks.keys())
        return cls.propose_mapping_from_names(raw)

    def rename_masks(
        self, mapping: Dict[str, str], *, validate_targets: bool = True
    ) -> None:
        """Rename masks in-place according to a mapping.

        Parameters
        ----------
        mapping : Dict[str, str]
            Dictionary mapping source names to destination names.
        validate_targets : bool, optional
            If True, only apply renames where the destination is a valid mask name.
        """
        for time_id, masks in self.masks.items():
            for src, dst in mapping.items():
                if src in masks:
                    if validate_targets and not self._is_valid_mask_name(dst):
                        # Skip invalid targets to avoid breaking downstream
                        continue
                    masks[dst] = masks[src]
                    if dst != src:
                        try:
                            del masks[src]
                        except KeyError:
                            pass
        return None

    def missing_targets(self, required: Iterable[str]) -> Dict[int, List[str]]:
        """Report which required mask names are missing at each timepoint.

        Parameters
        ----------
        required : Iterable[str]
            Canonical ROI names expected to be present (e.g., from config).

        Returns
        -------
        Dict[int, List[str]]
            Per-timepoint list of missing ROI names (empty dict means all present).
        """
        req_set = set(required)
        missing: Dict[int, List[str]] = {}
        for tp in sorted(self.images.keys()):
            have = set(self.masks.get(tp, {}).keys())
            miss = sorted(list(req_set - have))
            if miss:
                missing[tp] = miss
        return missing

    @staticmethod
    def apply_per_modality_mappings(
        ct_study: "LongitudinalStudy",
        spect_study: "LongitudinalStudy",
        ct_mask_mapping: Optional[Dict[str, str]] = None,
        spect_mask_mapping: Optional[Dict[str, str]] = None,
        manual_overrides: Optional[Dict[str, str]] = None,
        validate_targets: bool = True,
    ) -> Dict[str, Any]:
        """Apply modality-specific mask mappings to CT and SPECT studies.

        This helper automates the workflow:
        1. Merge user-provided mappings with manual overrides.
        2. Filter each mapping to keys actually present in each study.
        3. Check for conflicts (multiple sources mapping to the same target within a study).
        4. Apply renames in-place to both studies.
        5. Return diagnostic info (applied mappings, absent keys, conflicts).

        Parameters
        ----------
        ct_study : LongitudinalStudy
            The CT longitudinal study (used for volume/morphology).
        spect_study : LongitudinalStudy
            The SPECT/NM longitudinal study (used for activity).
        ct_mask_mapping : Optional[Dict[str, str]], optional
            User-provided mapping {raw_name: canonical_name} for CT masks.
            If None, proposes a mapping automatically from CT mask names.
        spect_mask_mapping : Optional[Dict[str, str]], optional
            User-provided mapping {raw_name: canonical_name} for SPECT masks.
            If None, proposes a mapping automatically from SPECT mask names.
        manual_overrides : Optional[Dict[str, str]], optional
            Additional mappings to override both CT and SPECT proposals (e.g., lesion mappings).
        validate_targets : bool, optional
            If True, only apply renames where the destination is a valid mask name (default: True).

        Returns
        -------
        Dict[str, Any]
            Diagnostic dictionary with keys:
            - 'ct_applied': Dict[str, str] - mappings applied to CT
            - 'spect_applied': Dict[str, str] - mappings applied to SPECT
            - 'ct_absent': List[str] - CT mapping keys not found in CT masks
            - 'spect_absent': List[str] - SPECT mapping keys not found in SPECT masks
            - 'ct_conflicts': Dict[str, List[str]] - CT conflicts (target: [sources])
            - 'spect_conflicts': Dict[str, List[str]] - SPECT conflicts (target: [sources])

        Example
        -------
        >>> ct_mapping = {
        ...     "Kidney_L_m": "Kidney_Left",
        ...     "Kidney_R_m": "Kidney_Right",
        ...     "Liver": "Liver",
        ... }
        >>> spect_mapping = {
        ...     "Kidney_L_a": "Kidney_Left",
        ...     "Kidney_R_a": "Kidney_Right",
        ...     "Liver": "Liver",
        ... }
        >>> result = LongitudinalStudy.apply_per_modality_mappings(
        ...     ct_study=longCT,
        ...     spect_study=longSPECT,
        ...     ct_mask_mapping=ct_mapping,
        ...     spect_mask_mapping=spect_mapping,
        ... )
        >>> print(result['ct_applied'])
        """
        manual = manual_overrides or {}

        # Default to auto-proposal if no explicit mapping provided
        if ct_mask_mapping is None:
            ct_names = set()
            for masks in ct_study.masks.values():
                ct_names.update(masks.keys())
            ct_mask_mapping = LongitudinalStudy.propose_mapping_from_names(ct_names)

        if spect_mask_mapping is None:
            spect_names = set()
            for masks in spect_study.masks.values():
                spect_names.update(masks.keys())
            spect_mask_mapping = LongitudinalStudy.propose_mapping_from_names(
                spect_names
            )

        # Merge manual overrides (take precedence)
        mapping_ct = dict(ct_mask_mapping)
        mapping_ct.update(manual)

        mapping_spect = dict(spect_mask_mapping)
        mapping_spect.update(manual)

        # Gather which source keys actually exist in each study
        def _keys_in_study(study: "LongitudinalStudy") -> set:
            present = set()
            for masks in study.masks.values():
                present.update(masks.keys())
            return present

        ct_present_keys = _keys_in_study(ct_study)
        spect_present_keys = _keys_in_study(spect_study)

        # Filter mappings to present keys
        filtered_ct = {k: v for k, v in mapping_ct.items() if k in ct_present_keys}
        filtered_spect = {
            k: v for k, v in mapping_spect.items() if k in spect_present_keys
        }

        # Track absent keys (provided but not present)
        absent_ct = sorted([k for k in mapping_ct.keys() if k not in ct_present_keys])
        absent_spect = sorted(
            [k for k in mapping_spect.keys() if k not in spect_present_keys]
        )

        # Check for conflicts (multiple sources -> same target within a study)
        def _check_conflicts(mapping: Dict[str, str]) -> Dict[str, List[str]]:
            inv: Dict[str, List[str]] = {}
            for src, dst in mapping.items():
                inv.setdefault(dst, []).append(src)
            return {dst: srcs for dst, srcs in inv.items() if len(srcs) > 1}

        conflicts_ct = _check_conflicts(filtered_ct)
        conflicts_spect = _check_conflicts(filtered_spect)

        # Apply renames in-place
        ct_study.rename_masks(filtered_ct, validate_targets=validate_targets)
        spect_study.rename_masks(filtered_spect, validate_targets=validate_targets)

        return {
            "ct_applied": filtered_ct,
            "spect_applied": filtered_spect,
            "ct_absent": absent_ct,
            "spect_absent": absent_spect,
            "ct_conflicts": conflicts_ct,
            "spect_conflicts": conflicts_spect,
        }

    @staticmethod
    def load_mappings_from_json(
        json_path: Union[str, Path],
    ) -> Dict[str, Dict[str, str]]:
        """Load CT and SPECT mask mappings from a JSON configuration file.

        The JSON file should contain a dictionary with keys 'ct_mappings' and/or
        'spect_mappings', each mapping to a dictionary of {raw_name: canonical_name}.

        Parameters
        ----------
        json_path : Union[str, Path]
            Path to the JSON configuration file.

        Returns
        -------
        Dict[str, Dict[str, str]]
            Dictionary with keys:
            - 'ct_mappings': Dict[str, str] - CT mask name mappings
            - 'spect_mappings': Dict[str, str] - SPECT mask name mappings

        Raises
        ------
        FileNotFoundError
            If the JSON file does not exist.
        json.JSONDecodeError
            If the JSON file is malformed.
        KeyError
            If the JSON file does not contain expected keys.

        Example
        -------
        >>> mappings = LongitudinalStudy.load_mappings_from_json("roi_mappings.json")
        >>> result = LongitudinalStudy.apply_per_modality_mappings(
        ...     ct_study=longCT,
        ...     spect_study=longSPECT,
        ...     ct_mask_mapping=mappings['ct_mappings'],
        ...     spect_mask_mapping=mappings['spect_mappings'],
        ... )

        Example JSON format
        -------------------
        {
            "ct_mappings": {
                "Kidney_L_m": "Kidney_Left",
                "Kidney_R_m": "Kidney_Right",
                "Liver": "Liver"
            },
            "spect_mappings": {
                "Kidney_L_a": "Kidney_Left",
                "Kidney_R_a": "Kidney_Right",
                "Liver": "Liver"
            }
        }
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Mapping config file not found: {json_path}")

        with open(path, "r") as f:
            config = json.load(f)

        # Validate structure
        if not isinstance(config, dict):
            raise ValueError(
                f"Expected JSON to contain a dictionary, got {type(config).__name__}"
            )

        result = {
            "ct_mappings": config.get("ct_mappings", {}),
            "spect_mappings": config.get("spect_mappings", {}),
        }

        # Validate that each mapping is a dict
        for key in ["ct_mappings", "spect_mappings"]:
            if not isinstance(result[key], dict):
                raise ValueError(
                    f"Expected '{key}' to be a dictionary, got {type(result[key]).__name__}"
                )

        return result

    def array_at(self, time_id: int) -> NDArray[Any]:
        """Access Array Data.

        Parameters
        ----------
        time_id : int
            The time point ID.

        Returns
        -------
        NDArray[Any]
            The array data at the specified time point.
        """
        return numpy.transpose(
            numpy.squeeze(SimpleITK.GetArrayFromImage(self.images[time_id])),
            axes=(1, 2, 0),
        )

    def array_of_activity_at(
        self, time_id: int, region: Optional[str] = None
    ) -> NDArray[Any]:
        """Return the array in units of activity in Bq.

        With the posibility of masking out for one specific region.
        """
        if self.modality not in ["NM", "PT"]:
            raise ValueError(f"Activity can't be calculated from {self.modality} data.")

        if time_id not in self.images:
            raise ValueError(f"Time ID {time_id} not found in dataset.")

        array = self.array_at(time_id=time_id)

        if region is None:
            mask = numpy.ones(shape=array.shape, dtype=numpy.bool_)
        else:
            if time_id not in self.masks:
                raise ValueError(
                    f"Time ID {time_id} does not include mask data. Did you run "
                    "add_masks_to_time_point()?"
                )
            if region not in self.masks[time_id]:
                available_regions = list(self.masks[time_id].keys())
                raise ValueError(
                    f"Region {region} not found in masks for time ID {time_id}. "
                    f"Available regions: {available_regions}"
                )
            if self.masks[time_id][region].shape != array.shape:
                raise ValueError(
                    f"Mask shape {self.masks[time_id][region].shape} doesn't match "
                    f"array shape {array.shape} for time ID {time_id}"
                )
            mask = self.masks[time_id][region]

        return array * mask * self.voxel_volume(time_id=time_id)

    def add_masks_to_time_point(
        self,
        time_id: int,
        masks: Dict[str, SimpleITK.Image],
        mask_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add Masks to time point.

        Args:
            time_id (int): Index of time-point ID.
            masks (Dict[str, SimpleITK.Image]): Dictionary containing masks for time point time_id, in the format {mask_name: mask_image (simpleITK)}
            mask_mapping (Optional[Dict[str, str]], optional): Mapping between masks names in input masks dictionary, and standard mask names in pyTheranostics. Defaults to None. If None, takes each name as is.

        Raises
        ------
        ValueError
            If mapping between user input masks and pyTheranostics standard mask names is invalid.

        """
        # If mask mapping is not specified, utilize user defined names in masks Dictionary.
        if mask_mapping is None:
            mask_mapping = {mask_name: mask_name for mask_name in masks.keys()}

        if time_id not in self.masks:
            self.masks[time_id] = {}

        for mask_source, mask_target in mask_mapping.items():

            if mask_source not in masks:
                raise ValueError(
                    f"{mask_source} is not part of the available masks: {masks.keys()}"
                )

            if not self._is_valid_mask_name(mask_target):
                raise ValueError(
                    f"{mask_target} is not a valid mask name. Please use one of: "
                    f"\n{self._VALID_ORGAN_NAMES}\nor 'Lesion_N' where N is a positive integer."
                )

            if mask_target in self.masks[time_id]:
                logger.warning(
                    f"{mask_target} found at Time = {time_id}. It will be over-written!"
                )

            # Masks are in the right orientation and spacing, however there could be discrepancies
            # in array shapes (reason, unknown). We resample to ensure shapes between image and
            # masks are consistent.
            # TODO: Fix.
            mask_ = resample_mask_to_target(
                mask_img=masks[mask_source], target_img=self.images[time_id]
            )

            mask_array = numpy.transpose(
                SimpleITK.GetArrayFromImage(mask_), axes=(1, 2, 0)
            )
            self.masks[time_id][mask_target] = mask_array.astype(numpy.bool_)

        return None

    def add_raw_masks_to_time_point(
        self,
        time_id: int,
        masks: Dict[str, SimpleITK.Image],
        *,
        resample_to_image_geometry: bool = True,
    ) -> None:
        """Add masks using their incoming names without validating or remapping.

        This is a permissive import method intended for early data ingestion.
        It stores masks under their original ROI names as found in RTSTRUCT or
        other sources. Downstream workflows can later inspect available names
        and explicitly normalize or remap to canonical labels.

        Args
        ----
        time_id : int
            Index of the time point to which masks will be added.
        masks : Dict[str, SimpleITK.Image]
            Dictionary of incoming masks {roi_name: sitk.Image}.
        resample_to_image_geometry : bool
            If True, resample each mask to match the study image geometry at
            this time point to ensure consistent array shapes. Defaults to True.

        Notes
        -----
        - No validation is performed on the ROI names.
        - Existing masks with the same name at this time_id will be overwritten.
        """
        if time_id not in self.masks:
            self.masks[time_id] = {}

        for mask_name, mask_img in masks.items():
            # Optionally enforce geometry consistency
            mask_itk = (
                resample_mask_to_target(
                    mask_img=mask_img, target_img=self.images[time_id]
                )
                if resample_to_image_geometry
                else mask_img
            )

            mask_array = numpy.transpose(
                SimpleITK.GetArrayFromImage(mask_itk), axes=(1, 2, 0)
            )
            if mask_name in self.masks[time_id]:
                logger.warning(
                    f"{mask_name} found at Time = {time_id}. It will be over-written!"
                )
            self.masks[time_id][mask_name] = mask_array.astype(numpy.bool_)

        return None

    def volume_of(self, region: str, time_id: int) -> float:
        """Return the volume of a region of interest, in mL.

        Parameters
        ----------
        region : str
            The region name.
        time_id : int
            The time point ID.

        Returns
        -------
        float
            Volume in mL.
        """
        return numpy.sum(self.masks[time_id][region]) * self.voxel_volume(
            time_id=time_id
        )

    def activity_in(self, region: str, time_id: int) -> float:
        """Return the activity within a region of interest.

        The units of the nuclear medicine data should be Bq/mL.
        """
        if self.meta[time_id].Radionuclide is None or self.modality not in ["NM", "PT"]:
            raise AssertionError(
                "Can't compute activity if the image data does not represent the distribution of a radionuclide"
            )
        return numpy.sum(
            self.masks[time_id][region]
            * self.array_at(time_id=time_id)
            * self.voxel_volume(time_id=time_id)
        )

    def density_of(self, region: str, time_id: int) -> float:
        """Return the mean density of region of interest, in HU.

        Parameters
        ----------
        region : str
            The region name.
        time_id : int
            The time point ID.

        Returns
        -------
        float
            Mean density in HU.
        """
        return float(
            numpy.mean(self.array_at(time_id=time_id)[self.masks[time_id][region] > 0])
        )

    def voxel_volume(self, time_id: int) -> float:
        """Return the volume of a voxel in mL.

        Parameters
        ----------
        time_id : int
            The time point ID.

        Returns
        -------
        float
            Voxel volume in mL.
        """
        spacing = self.images[time_id].GetSpacing()
        return float(spacing[0] / 10 * spacing[1] / 10 * spacing[2] / 10)

    def average_of(self, region: str, time_id: int) -> float:
        """Compute average value in a region.

        Args:
            region (str): The region name.
            time_id (int): The time point ID.

        Returns
        -------
        float
            Average value in the region.
        """
        return float(
            numpy.average(self.array_at(time_id=time_id)[self.masks[time_id][region]])
        )

    def add_bone_marrow_mask_from_phantom(
        self,
        phantom_skeleton_path: Path,
        phantom_bone_marrow_path: Path,
        num_iterations: int = 3,
    ) -> None:
        """Generate Bone Marrow mask on each time point.

        Registers a generic skeleton derived from an XCAT phantom into the patient's Skeleton
        CT and subsequently applying this spatial transformation to register the phantom's bone
        marrow into the patient's anatomy.

        Args
        ----
            phantom_skeleton_path (Path): Path to phantom Skeleton .nii file.
            phantom_bone_marrow_path (Path): Path to phantom Bone Marrow .nii file.
        """
        logger.info(
            "Running Personalized Bone Marrow generation from XCAT Phantom. This feature is unstable. Please review the generated BoneMarrow masks."
        )

        if self.modality != "CT":
            raise AssertionError(
                f"Phantom skeleton can only be registered to CT data. This is modality = {self.modality}"
            )

        if "Skeleton" not in self.masks[0]:
            raise AssertionError("Skeleton mask not found. Can't continue.")

        # Since algorithm is not very stable (sometimes registration fails), we perform multiple iterations (aka repetitions) and keep best
        # results according to jaccard index.
        best_index = {time_id: 0 for time_id in self.images.keys()}

        for i in range(num_iterations):
            logger.info(f"Registration :: Iteration {i+1}")
            # Loop through each time point:
            for time_id, ct in self.images.items():
                # Register Skeleton
                logger.debug(
                    f" >> Registering Phantom Skeleton to CT at time point {time_id} ..."
                )
                RegManager = PhantomToCTBoneReg(
                    CT=ct, phantom_skeleton_path=phantom_skeleton_path
                )
                _ = RegManager.register(
                    fixed_image=RegManager.CT, moving_image=RegManager.Phantom
                )

                # Register Bone Marrow
                marrow_mask = numpy.transpose(
                    SimpleITK.GetArrayFromImage(
                        RegManager.register_mask(
                            fixed_image=RegManager.CT,
                            mask_path=phantom_bone_marrow_path,
                        )
                    ),
                    axes=(1, 2, 0),
                )

                # Threshold:
                marrow_mask = marrow_mask >= 1

                # Exclude voxels outside of the patient's skeleton.
                marrow_mask *= self.masks[time_id]["Skeleton"]

                # Compute Index:
                jaccard = jaccard_index(self.masks[time_id]["Skeleton"], marrow_mask)

                if jaccard > best_index[time_id]:
                    self.masks[time_id]["BoneMarrow"] = marrow_mask  # Threshold.
                    best_index[time_id] = jaccard

                # Calculate Index
                logger.debug(
                    f" >>> Jaccard Index between Skeleton and Segmented Bone Marrow: {jaccard: 1.2f}"
                )

        # Final Results:
        logger.info(" >>> Final Jaccard Indices:")
        for time_id in self.masks.keys():
            logger.info(f" >>> Time point {time_id}: {best_index[time_id]}")

        return None

    def check_masks_consistency(self) -> None:
        """Check that we have the same masks in all time points.

        Raises
        ------
        AssertionError
            If masks are inconsistent across time points.
        """
        masks_list = [sorted(list(masks.keys())) for _, masks in self.masks.items()]

        sample = masks_list[0]

        for masks in masks_list:
            if masks != sample:
                raise AssertionError(f"Incosistent Masks! -> {masks_list}")

        return None

    def save_image_to_nii_at(
        self, time_id: int, out_path: Path, name: str = ""
    ) -> None:
        """Save Image from a particular time-point as a nifty file.

        Args
        ----
            time_id (int): The time ID representing the time point to be saved.
            out_path (Path): The path to the folder where images will be written.
        """
        logger.info(f"Writing Image ({name}) into nifty file.")
        SimpleITK.WriteImage(
            image=SimpleITK.Cast(self.images[time_id], SimpleITK.sitkInt32),
            fileName=out_path / f"Image_{time_id}{name}.nii.gz",
        )
        return None

    def save_image_to_mhd_at(
        self, time_id: int, out_path: Path, name: str = ""
    ) -> None:
        """Save Image from a particular time-point as a nifty file.

        Args
        ----
            time_id (int): The time ID representing the time point to be saved.
            out_path (Path): The path to the folder where images will be written.
        """
        logger.info(f"Writing Image ({name}) into mhd file.")
        SimpleITK.WriteImage(
            image=SimpleITK.Cast(self.images[time_id], SimpleITK.sitkInt32),
            fileName=os.path.join(out_path, f"{name}.mhd"),
        )
        return None

    def save_masks_to_nii_at(
        self, time_id: int, out_path: Path, regions: List[str]
    ) -> None:
        """Save Masks from a particular time-point as a nifty file.

        Args
        ----
            time_id (int): The time ID representing  the time point to be saved.
            out_path (Path): The path to the folder where images will be written.
            regions (List[str]): A list of regions (masks) to be saved. If empty, save all masks.
        """
        mask_names = list(self.masks[time_id].keys())
        all_masks = numpy.zeros_like(self.masks[time_id][mask_names[0]]).astype(
            numpy.int16
        )  # Get the shape of the first mask available.

        if len(regions) > 0:
            mask_names = [
                region for region in regions if region in self.masks[time_id].keys()
            ]

        for mask_id, region_name in enumerate(mask_names):
            all_masks += (mask_id + 1) * (self.masks[time_id][region_name]).astype(
                numpy.int16
            )

        mask_image = itk_image_from_array(
            array=numpy.transpose(all_masks, axes=(2, 0, 1)),
            ref_image=self.images[time_id],
        )

        logger.info(f"Writing Masks ({mask_names}) into nifty file.")

        SimpleITK.WriteImage(
            image=mask_image, fileName=out_path / f"Masks_{time_id}.nii.gz"
        )

        return None
