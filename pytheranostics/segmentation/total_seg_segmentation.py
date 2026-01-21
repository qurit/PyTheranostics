"""Segmentation processing module for TotalSegmentator workflows."""

import re
from pathlib import Path
from typing import List

from totalsegmentator.python_api import totalsegmentator


class SegmentationProcessor:
    """Handler for batch processing CT scans with TotalSegmentator."""

    def __init__(self, base_output_dir: str, device: str = "mps"):
        """Initialize the segmentation processor.

        Parameters
        ----------
        base_output_dir : str
            Base directory for all segmentation outputs.
        device : str, optional
            Computing device ('mps', 'cuda', or 'cpu'), by default "mps".
        """
        self.base_output_dir = Path(base_output_dir)
        self.device = device

    def extract_timepoint(self, folder_path: Path) -> str:
        """Extract timepoint from folder name using regex.

        Parameters
        ----------
        folder_path : Path
            Path to the input folder.

        Returns
        -------
        str
            Timepoint string (e.g., '0p5h', '6h', '24h').
        """
        match = re.search(r"CT\.(\d+(?:\.\d+)?h)", folder_path.name)
        if match:
            timepoint = match.group(1).replace(".", "p")
            return timepoint
        return "unknown"

    def process_folder(self, input_folder: str) -> str:
        """Process a single input folder with TotalSegmentator.

        Parameters
        ----------
        input_folder : str
            Path to input DICOM folder.

        Returns
        -------
        str
            Timepoint identifier.
        """
        input_path = Path(input_folder)
        timepoint = self.extract_timepoint(input_path)
        output_subfolder = self.base_output_dir / timepoint

        print(f"Processing {timepoint}: {input_path}")
        print(f"Output to: {output_subfolder}")

        totalsegmentator(str(input_path), str(output_subfolder), device=self.device)

        print(f"✓ Completed {timepoint}")
        return timepoint

    def process_batch(
        self, input_folders: List[str], parallel: bool = False, max_workers: int = 2
    ):
        """Process multiple input folders.

        Parameters
        ----------
        input_folders : List[str]
            List of input folder paths.
        parallel : bool, optional
            Whether to process in parallel, by default False.
        max_workers : int, optional
            Number of parallel workers (if parallel=True), by default 2.
        """
        if parallel:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                executor.map(self.process_folder, input_folders)
        else:
            for folder in input_folders:
                self.process_folder(folder)
