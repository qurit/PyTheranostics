"""QC checks for planar acquisitions."""

import pydicom

from pytheranostics.qc.qc import QC


class PlanarQC(QC):
    """QC checks specific to planar acquisitions."""

    def __init__(self, isotope, dicomfile, db_dic, cal_type="planar"):
        """Load the planar DICOM file and associated calibration forms."""
        super().__init__(isotope, db_dic=db_dic, cal_type=cal_type)
        self.ds = pydicom.dcmread(dicomfile)

    def check_windows_energy(self):
        """Run the planar QC workflow and populate the summary."""
        self.append_to_summary(f"QC for planar scan of {self.isotope}:\n\n")

        self.check_camera_parameters()

        self.print_summary()

    def check_camera_parameters(self):
        """Verify DICOM acquisition parameters match the expected protocol."""
        camera_manufacturer = self.ds.Manufacturer
        camera_model = self.ds.ManufacturerModelName
        acquisition_date = self.ds.AcquisitionDate
        acquisition_time = self.ds.AcquisitionTime
        modality = self.ds.Modality
        duration = (self.ds.ActualFrameDuration / 1000) / 60
        rows = self.ds.Rows
        cols = self.ds.Columns
        zoom = self.ds.DetectorInformationSequence[0].ZoomFactor

        self.append_to_summary(f"CAMERA: {camera_manufacturer} {camera_model}\t \n")
        self.append_to_summary(f"MODALITY: {modality}\t \n")
        self.append_to_summary(
            f"Scan performed on: {acquisition_date} at {acquisition_time}\t \n\n"
        )

        # check duration
        if round(duration) == self.isotope_dic["planar"]["duration"]:
            self.append_to_summary(f"SCAN DURATION: {round(duration)} minutes\tOK\n\n")
        else:
            self.append_to_summary(
                f'SCAN DURATION: {duration} minutes (should be {self.isotope_dic["planar"]["duration"]} mins)\tMISMATCH\n\n'
            )

        # check windows
        self.window_check_df = self.window_check(type="planar")

        # check collimator
        accepted_collimators = self.isotope_dic["accepted_collimators"]
        if (len(self.db_df["cal_data"]["collimator"].unique()) == 1) & (
            self.db_df["cal_data"]["collimator"].unique()[0] in accepted_collimators
        ):
            self.append_to_summary(
                f"COLLIMATOR: {self.db_df['cal_data']['collimator'].unique()[0]}\t         OK\n\n"
            )
        else:
            self.append_to_summary(
                f"COLLIMATOR: {self.db_df['cal_data']['collimator'].unique()[0]} is not in the accepted collimator list.\t         VERIFY\n\n"
            )

        # check matrix size
        if [rows, cols] == self.isotope_dic["planar"]["matrix"]:
            self.append_to_summary(f"MATRIX SIZE: {rows} x {cols}\t         OK\n")
        else:
            self.append_to_summary(
                f'MATRIX SIZE: {rows} x {cols} (should be {self.isotope_dic["planar"]["matrix"][0]} x {self.isotope_dic["planar"]["matrix"][1]}\t MISMATCH\n'
            )

        # check zoom
        if zoom == self.isotope_dic["planar"]["zoom"]:
            self.append_to_summary(f"ZOOM: {zoom}\t                   OK\n\n")
        else:
            self.append_to_summary(
                f'ZOOM: {zoom} (should be {self.isotope_dic["planar"]["zoom"]}\tMISMATCH\n\n'
            )
