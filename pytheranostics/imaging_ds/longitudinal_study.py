"""Module for longitudinal medical imaging studies."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy
import SimpleITK
from numpy.typing import NDArray

from pytheranostics.imaging_ds.metadata import ImagingMetadata
from pytheranostics.imaging_tools.Tools import (
    itk_image_from_array,
    jaccard_index,
    load_from_dicom_dir,
    resample_mask_to_target,
)
from pytheranostics.registration.PhantomToCT import PhantomToCTBoneReg


class LongitudinalStudy:
    """Longitudinal Study Data Class.

    Holds multiple medical imaging datasets, alongside with masks for organs/regions
    of interest and meta-data.
    """

    _VALID_ORGAN_NAMES = [
        "Kidney_Left",
        "Kidney_Right",
        "Liver",
        "Spleen",
        "Bladder",
        "SubmandibularGland_Left",
        "SubmandibularGland_Right",
        "ParotidGland_Left",
        "ParotidGland_Right",
        "BoneMarrow",
        "Skeleton",
        "WholeBody",
        "RemainderOfBody",
        "TotalTumorBurden",
    ]

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

        for time_id, dicom_dir in enumerate(dicom_dirs):
            image, meta = load_from_dicom_dir(
                dir=dicom_dir, modality=modality, calibration_factor=calibration_factor
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
        - Standard organ names from _VALID_ORGAN_NAMES
        - Lesion names in format 'Lesion_N' where N is a positive integer
        """
        if mask_name in LongitudinalStudy._VALID_ORGAN_NAMES:
            return True
        lesion_pattern = r"^Lesion_([1-9]\d*)$"
        return bool(re.match(lesion_pattern, mask_name))

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
                print(
                    f"Warning: {mask_target} found at Time = {time_id}. It will be over-written!"
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
        print(
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
            print(f"Registration :: Iteration {i+1}")
            # Loop through each time point:
            for time_id, ct in self.images.items():
                # Register Skeleton
                print(
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
                print(
                    f" >>> Jaccard Index between Skeleton and Segmented Bone Marrow: {jaccard: 1.2f}"
                )

        # Final Results:
        print(" >>> Final Jaccard Indices:")
        for time_id in self.masks.keys():
            print(f" >>> Time point {time_id}: {best_index[time_id]}")

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
        print(f"Writing Image ({name}) into nifty file.")
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
        print(f"Writing Image ({name}) into mhd file.")
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

        print(f"Writing Masks ({mask_names}) into nifty file.")

        SimpleITK.WriteImage(
            image=mask_image, fileName=out_path / f"Masks_{time_id}.nii.gz"
        )

        return None
