"""Radiomics feature extraction utilities."""

from __future__ import print_function

import os

import pandas as pd
import SimpleITK as sitk
import six
from radiomics import featureextractor


class Radiomics:
    """Generate radiomics features for longitudinal studies."""

    def __init__(self, imagemodality, patient_id, cycle, image, mask, organslist):
        """Store study metadata, image arrays, and ROI masks."""
        self.imagemodality = imagemodality
        self.patient_id = patient_id
        self.cycle = cycle
        self.image = image
        self.mask = mask
        self.organslist = organslist

    def prepareimages(self):
        """Export the image and ROI masks to NRRD files for PyRadiomics."""
        img = sitk.GetImageFromArray(self.image)

        sitk.WriteImage(
            img,
            f"/mnt/y/Sara/PR21_dosimetry/{self.patient_id}/cycle0{self.cycle}/radiomics/{self.imagemodality}.nrrd",
        )
        for organ in self.organslist:
            self.mask[organ] = self.mask[organ].astype(int)
            img = sitk.GetImageFromArray(self.mask[organ])
            sitk.WriteImage(
                img,
                f"/mnt/y/Sara/PR21_dosimetry/{self.patient_id}/cycle0{self.cycle}/radiomics/{organ}.nrrd",
            )

    def featureextractor(self):
        """Run PyRadiomics using the configured parameter set."""
        paramPath = os.path.join("..", "data", "Params.yaml")

        extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)

        radiomics_list = []
        for organ in self.organslist:
            imagepath = f"/mnt/y/Sara/PR21_dosimetry/{self.patient_id}/cycle0{self.cycle}/dosiomics/{self.imagemodality}.nrrd"
            maskpath = f"/mnt/y/Sara/PR21_dosimetry/{self.patient_id}/cycle0{self.cycle}/dosiomics/{organ}.nrrd"
            result = extractor.execute(imagepath, maskpath)

            data = {"organ": organ}
            for key, value in six.iteritems(result):
                data[key] = value

            radiomics_list.append(data)

        radiomics_df = pd.DataFrame(radiomics_list)
        radiomics_df.to_csv(
            f"/mnt/y/Sara/PR21_dosimetry/output/{self.patient_id}_cycle0{self.cycle}_radiomics_{self.imagemodality}_output.csv"
        )

        return radiomics_df
