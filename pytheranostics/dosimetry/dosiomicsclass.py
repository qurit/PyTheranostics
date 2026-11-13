"""Radiomics feature extraction utilities."""

from __future__ import print_function

from pathlib import Path

import pandas as pd
import SimpleITK as sitk
import six
from radiomics import featureextractor

# PyRadiomics parameter file shipped with the package.
DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[1] / "data" / "Params.yaml"


class Radiomics:
    """Generate radiomics features for longitudinal studies."""

    def __init__(
        self,
        imagemodality,
        patient_id,
        cycle,
        image,
        mask,
        organslist,
        *,
        output_dir,
        params_path=None,
    ):
        """Store study metadata, image arrays, and ROI masks.

        Parameters
        ----------
        imagemodality : str
            Modality label used in the generated file names.
        patient_id : str
            Identifier used to lay out the per-study output directories.
        cycle : int
            Treatment cycle number.
        image : numpy.ndarray
            Image volume to extract features from.
        mask : dict of str to numpy.ndarray
            ROI masks keyed by organ name.
        organslist : list of str
            Organs to extract features for.
        output_dir : str or pathlib.Path
            Directory that the NRRD files and the feature table are written
            to. Required: this must never point at a location holding
            identifiable patient data that is shared or version controlled.
        params_path : str or pathlib.Path, optional
            PyRadiomics parameter file. Defaults to the one shipped in
            ``pytheranostics/data/Params.yaml``.
        """
        self.imagemodality = imagemodality
        self.patient_id = patient_id
        self.cycle = cycle
        self.image = image
        self.mask = mask
        self.organslist = organslist
        self.output_dir = Path(output_dir)
        self.params_path = Path(params_path) if params_path else DEFAULT_PARAMS_PATH

    def _study_dir(self, kind):
        """Return the directory for one study's ``kind`` files, creating it."""
        path = self.output_dir / self.patient_id / f"cycle0{self.cycle}" / kind
        path.mkdir(parents=True, exist_ok=True)
        return path

    def prepareimages(self):
        """Export the image and ROI masks to NRRD files for PyRadiomics."""
        destination = self._study_dir("radiomics")

        img = sitk.GetImageFromArray(self.image)
        sitk.WriteImage(img, str(destination / f"{self.imagemodality}.nrrd"))

        for organ in self.organslist:
            self.mask[organ] = self.mask[organ].astype(int)
            img = sitk.GetImageFromArray(self.mask[organ])
            sitk.WriteImage(img, str(destination / f"{organ}.nrrd"))

    def featureextractor(self):
        """Run PyRadiomics using the configured parameter set."""
        extractor = featureextractor.RadiomicsFeatureExtractor(str(self.params_path))

        source = self._study_dir("dosiomics")
        imagepath = source / f"{self.imagemodality}.nrrd"

        radiomics_list = []
        for organ in self.organslist:
            maskpath = source / f"{organ}.nrrd"
            result = extractor.execute(str(imagepath), str(maskpath))

            data = {"organ": organ}
            for key, value in six.iteritems(result):
                data[key] = value

            radiomics_list.append(data)

        radiomics_df = pd.DataFrame(radiomics_list)

        output_root = self.output_dir / "output"
        output_root.mkdir(parents=True, exist_ok=True)
        radiomics_df.to_csv(
            output_root
            / (
                f"{self.patient_id}_cycle0{self.cycle}"
                f"_radiomics_{self.imagemodality}_output.csv"
            )
        )

        return radiomics_df
