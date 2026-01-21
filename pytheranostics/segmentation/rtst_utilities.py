"""Helpers for working with RT structure sets."""

import csv
import os
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import pydicom
from rt_utils import RTStructBuilder


class RTStructConverter:
    """Convert NIfTI segmentation masks to DICOM RT-STRUCT."""

    def __init__(self, ct_dicom_folder: str):
        """Initialize the RT-STRUCT converter.

        Parameters
        ----------
        ct_dicom_folder : str
            Path to CT DICOM series folder.
        """
        self.ct_dicom_folder = ct_dicom_folder
        self.rtstruct = RTStructBuilder.create_new(dicom_series_path=ct_dicom_folder)

    def add_mask_from_nifti(
        self,
        mask_path: str,
        roi_name: Optional[str] = None,
        permute_axes: bool = True,
        flip_x: bool = True,
    ):
        """Add a NIfTI mask as an ROI to the RT-STRUCT.

        Parameters
        ----------
        mask_path : str
            Path to the NIfTI mask file.
        roi_name : str, optional
            Name for the ROI (defaults to filename).
        permute_axes : bool, optional
            Whether to swap X and Y axes, by default True.
        flip_x : bool, optional
            Whether to flip the X axis, by default True.
        """
        mask_path = Path(mask_path)

        # Use filename as ROI name if not provided
        if roi_name is None:
            roi_name = mask_path.stem.replace(".nii", "")

        # Load the NIfTI mask
        mask_nii = nib.load(str(mask_path))
        mask_array = mask_nii.get_fdata().astype(bool)

        # Apply transformations if needed
        if permute_axes:
            mask_array = mask_array.transpose(1, 0, 2)
        if flip_x:
            mask_array = mask_array[::-1, :, :]

        # Add to RT-STRUCT
        self.rtstruct.add_roi(mask=mask_array, name=roi_name)
        print(f"Added ROI: {roi_name}")

    def add_masks_from_folder(
        self, nifti_folder: str, permute_axes: bool = True, flip_x: bool = True
    ):
        """Add all NIfTI masks from a folder to the RT-STRUCT.

        Parameters
        ----------
        nifti_folder : str
            Path to folder containing NIfTI masks.
        permute_axes : bool, optional
            Whether to swap X and Y axes, by default True.
        flip_x : bool, optional
            Whether to flip the X axis, by default True.
        """
        nifti_folder = Path(nifti_folder)

        for fname in os.listdir(nifti_folder):
            if fname.endswith(".nii") or fname.endswith(".nii.gz"):
                mask_path = nifti_folder / fname
                self.add_mask_from_nifti(
                    str(mask_path), permute_axes=permute_axes, flip_x=flip_x
                )

    def save(self, output_path: str):
        """Save the RT-STRUCT to a DICOM file.

        Parameters
        ----------
        output_path : str
            Path for the output RT-STRUCT file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.rtstruct.save(str(output_path))
        print(f"RT-STRUCT saved to: {output_path}")


def get_rtstruct_roi_names(rtstruct_path: str) -> List[str]:
    """Get list of ROI names from an RT-STRUCT file.

    Parameters
    ----------
    rtstruct_path : str
        Path to the RT-STRUCT DICOM file.

    Returns
    -------
    List[str]
        List of ROI names.
    """
    ds = pydicom.dcmread(rtstruct_path)
    roi_names = []

    if hasattr(ds, "StructureSetROISequence"):
        for roi in ds.StructureSetROISequence:
            roi_names.append(roi.ROIName)

    return roi_names


def print_rtstruct_info(rtstruct_path: str):
    """Print detailed information about an RT-STRUCT file.

    Parameters
    ----------
    rtstruct_path : str
        Path to the RT-STRUCT DICOM file.
    """
    ds = pydicom.dcmread(rtstruct_path)

    print(f"\n{'='*60}")
    print(f"RT-STRUCT: {Path(rtstruct_path).name}")
    print(f"{'='*60}")

    # Basic info
    if hasattr(ds, "PatientName"):
        print(f"Patient Name: {ds.PatientName}")
    if hasattr(ds, "PatientID"):
        print(f"Patient ID: {ds.PatientID}")
    if hasattr(ds, "StudyDate"):
        print(f"Study Date: {ds.StudyDate}")
    if hasattr(ds, "StructureSetLabel"):
        print(f"Structure Set Label: {ds.StructureSetLabel}")

    # ROI information
    if hasattr(ds, "StructureSetROISequence"):
        print(f"\nNumber of ROIs: {len(ds.StructureSetROISequence)}")
        print("\nROI List:")
        for i, roi in enumerate(ds.StructureSetROISequence, 1):
            roi_number = roi.ROINumber
            roi_name = roi.ROIName
            print(f"  {i}. [{roi_number}] {roi_name}")
    else:
        print("\nNo ROIs found in this RT-STRUCT")

    print(f"{'='*60}\n")


def export_rtstruct_rois_to_csv(rtstruct_path: str, output_csv: str):
    """Export RT-STRUCT ROI information to a CSV file.

    Parameters
    ----------
    rtstruct_path : str
        Path to the RT-STRUCT DICOM file.
    output_csv : str
        Path for the output CSV file.
    """
    ds = pydicom.dcmread(rtstruct_path)

    # Prepare data for CSV
    rows = []

    if hasattr(ds, "StructureSetROISequence"):
        for roi in ds.StructureSetROISequence:
            roi_number = roi.ROINumber
            roi_name = roi.ROIName
            rows.append(
                {
                    "ROI_Number": roi_number,
                    "ROI_Name": roi_name,
                    "RT_STRUCT_File": Path(rtstruct_path).name,
                }
            )

    # Write to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as csvfile:
        if rows:
            fieldnames = ["ROI_Number", "ROI_Name", "RT_STRUCT_File"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            print(f"✓ Exported {len(rows)} ROIs to: {output_path}")
        else:
            print(f"⚠️  No ROIs found in {rtstruct_path}")


def export_multiple_rtstructs_to_csv(rtstruct_dir: str, output_csv: str):
    """Export ROI information from multiple RT-STRUCT files to a single CSV.

    Parameters
    ----------
    rtstruct_dir : str
        Directory containing RT-STRUCT files.
    output_csv : str
        Path for the output CSV file.
    """
    rtstruct_dir = Path(rtstruct_dir)
    all_rows = []

    # Process all DICOM files in the directory (recursively)
    for dcm_file in sorted(rtstruct_dir.rglob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(dcm_file))

            # Extract timepoint from filename (e.g., "rtstruct_0p5h.dcm" -> "0p5h")
            filename = dcm_file.stem  # rtstruct_0p5h
            timepoint = filename.replace("rtstruct_", "")
            # Patient folder assumed to be parent directory name
            patient_id = dcm_file.parent.name

            if hasattr(ds, "StructureSetROISequence"):
                for roi in ds.StructureSetROISequence:
                    all_rows.append(
                        {
                            "PatientID": patient_id,
                            "Timepoint": timepoint,
                            "ROI_Number": roi.ROINumber,
                            "ROI_Name": roi.ROIName,
                            "RT_STRUCT_File": dcm_file.name,
                        }
                    )
        except Exception as e:
            print(f"⚠️  Error processing {dcm_file.name}: {e}")

    # Write to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if all_rows:
        with open(output_path, "w", newline="") as csvfile:
            fieldnames = [
                "PatientID",
                "Timepoint",
                "ROI_Number",
                "ROI_Name",
                "RT_STRUCT_File",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
            print(
                f"✓ Exported {len(all_rows)} ROIs from {len(set((r['PatientID'], r['Timepoint']) for r in all_rows))} patient-timepoints to: {output_path}"
            )
    else:
        print(f"⚠️  No ROIs found in {rtstruct_dir}")


def rtst_to_mask(dicom_series_path, rt_struct_path):
    """Load an RTSTRUCT and return a dict of ROI masks keyed by ROI name."""
    # Load existing RT Struct. Requires the series path and existing RT Struct path
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=dicom_series_path, rt_struct_path=rt_struct_path
    )

    # View all of the ROI names from within the image
    print(rtstruct.get_roi_names())
    rois = rtstruct.get_roi_names()

    # Loading the 3D Mask from within the RT Struct
    mask_3d = {}

    for voi in rois:
        mask_3d[voi] = rtstruct.get_roi_mask_by_name(voi)

    return mask_3d
    # # Display one slice of the region
    # first_mask_slice = mask_3d[voi][:, :, 0]
    # plt.imshow(first_mask_slice)
    # plt.show()
