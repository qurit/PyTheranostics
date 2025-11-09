"""Organ S-value dosimetry class.

Perform organ-level, patient-specific dosimetry using organ S-values.
Currently supports export to Olinda/EXM.
"""

import datetime
import re
from os import makedirs, path
from typing import Any, Dict, Optional, Tuple

import numpy
import pandas
from scipy.interpolate import PchipInterpolator

from pytheranostics.dosimetry.base_dosimetry import BaseDosimetry
from pytheranostics.imaging_ds.longitudinal_study import LongitudinalStudy

parent_dir = path.dirname(path.dirname(__file__))
SVALUES_PATH = path.join(parent_dir, "data", "s-values")
MASSES_PATH = path.join(parent_dir, "data", "ICRP_phantom_masses")


class OrganSDosimetry(BaseDosimetry):
    """Organ S-value dosimetry class.

    Perform organ-level, patient-specific dosimetry using organ S-values.
    Currently supports export to Olinda/EXM.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        nm_data: LongitudinalStudy,
        ct_data: Optional[LongitudinalStudy],
        clinical_data: Optional[pandas.DataFrame] = None,
    ) -> None:
        super().__init__(config, nm_data, ct_data, clinical_data)
        self.check_mandatory_fields_organ()

        """Inputs:
            config: Configuration parameters for dosimetry calculations, a Dict.
                    Note: defined VOIs should have the same naming convention as source organs in Olinda.
                    We included method prepare_df() that combines kidneys and salivary glands into one VOI.
            nm_data: longitudinal, quantitative, nuclear-medicine imaging data, type LongitudinalStudy.
                     Note: voxel values should be in units of Bq/mL.
            ct_data: longitudinal CT imaging data, type LongitudinalStudy,
                     Note: voxel values should be in HU units.
            clinical_data: clinical data such as blood sampling, an optional pandas DataFrame.
                     Note: blood counting should be in units of Bq/mL.
        """

    def check_mandatory_fields_organ(self) -> None:
        """Check for mandatory fields in the configuration for organ-level dosimetry."""
        if "Organ" not in self.config["Level"]:
            print("Verify the level on which dosimetry should be performed.")

        return None

    def composition_and_density_from_HU(self, density: float) -> Tuple[str, float]:
        """Determine composition and density for a given CT HU value."""
        if density <= 100:
            return "100%/0%", 1.03
        elif density <= 250:
            return "75%/25%", 1.255
        elif density <= 500:
            return "50%/50%", 1.48
        elif density <= 750:
            return "25%/75%", 1.7
        else:
            return "0%/100%", 1.92

    def s_value_from_mass(self, mass: float, composition: str) -> float:
        """Return interpolated total S value (mGy MBq^-1 h^-1) for mass and composition."""
        # Select the appropriate tumor mass and s_value arrays based on composition
        mass_data = self.mass_and_s_values[composition]["tumor_mass"]
        s_value_data = self.mass_and_s_values[composition]["total_s_value"]

        # Perform PCHIP interpolation
        pchip_interpolator = PchipInterpolator(mass_data, s_value_data)

        # Interpolate and return the result in mGy MBq^-1 h^-1
        return pchip_interpolator(mass) * 3.6 * 10**12

    def apply_sphere_method(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """Compute absorbed dose using the sphere method."""
        df = df.copy()

        # Compute mean volume and density
        df["Volume_CT_mL"] = df["Volume_CT_mL"].apply(lambda x: numpy.mean(x))
        df["Density_HU"] = df["Density_HU"].apply(lambda x: numpy.mean(x))

        # Compute composition and density from HU
        df[["Composition", "Density_g_per_mL"]] = df["Density_HU"].apply(
            lambda x: pandas.Series(self.composition_and_density_from_HU(x))
        )

        # Calculate mass
        df["Mass_g"] = df["Density_g_per_mL"] * df["Volume_CT_mL"]

        # Calculate total S-value
        df["Total_S_Value"] = df.apply(
            lambda row: self.s_value_from_mass(row["Mass_g"], row["Composition"]),
            axis=1,
        )

        # Calculate absorbed dose in Gy
        injected_activity = float(self.config["InjectedActivity"])
        df["AD_Gy"] = (  # Gy
            df["TIA_h"]  # h
            * df["Total_S_Value"]  # mGy MBq^-1 h^-1
            * injected_activity  # MBq
            / 1000
        )
        return df

    def calculate_ttb(self):
        """Compute Total Tumor Burden (TTB) metrics and append to results_lesions."""
        metrics = {
            "Mass_g": self.results_lesions["Mass_g"].sum(),
            "Volume_CT_mL": self.results_lesions["Volume_CT_mL"].sum(),
            "TIA_h": self.results_lesions["TIA_h"].sum(),
            "AD_Gy": (
                (self.results_lesions["Mass_g"] * self.results_lesions["AD_Gy"]).sum()
            )
            / (
                self.results_lesions["Mass_g"].sum()
                if self.results_lesions["Mass_g"].sum() > 0
                else 0
            ),
        }

        TTB = pandas.DataFrame(metrics, index=["TTB"])
        self.results_lesions = pandas.concat([self.results_lesions, TTB], axis=0)
        return self.results_lesions

    def prepare_data(self) -> None:
        """
        Prepare data for dosimetry calculations or export based on the configuration.

        For organ-level workflows the method either exports data compatible with
        Olinda/MIRDcalc or performs the configured calculation, sourcing S-values
        from the selected tables and honoring options such as ROB, lesions, or
        salivary gland handling. For voxel-level workflows it assembles the inputs
        for kernel-based calculations and writes the data using the requested
        voxel-level format (for example, NIfTI).
        """
        self.results_fitting = self.results[["Volume_CT_mL", "TIA_h"]].copy()
        # Average Volume over time points.
        self.results_fitting["Volume_CT_mL"] = self.results_fitting[
            "Volume_CT_mL"
        ].apply(lambda x: numpy.mean(x))
        if "Organ" in self.config["Level"]:
            organ_conf = self.config["OrganLevel"]
            output_type = organ_conf["Output"]["Type"]
            print(output_type)

            # Average Volume over time points.
            self.results_fitting["Volume_CT_mL"] = self.results_fitting[
                "Volume_CT_mL"
            ].apply(lambda x: numpy.mean(x))

            # Combine Kidneys.
            kidneys = ["Kidney_Left", "Kidney_Right"]
            self.results_fitting.loc["Kidneys"] = self.results_fitting.loc[
                kidneys
            ].sum()
            self.results_fitting = self.results_fitting.drop(kidneys)

            # Combine Salivary Glands.
            sal_glands = [
                "ParotidGland_Left",
                "ParotidGland_Right",
                "SubmandibularGland_Left",
                "SubmandibularGland_Right",
            ]
            if "Yes" in self.config["OrganLevel"]["AdditionalOptions"].get(
                "SalivaryGlandsSeparately"
            ):
                self.results_salivaryglands = self.results[
                    self.results.index.str.contains("Gland")
                ]
            else:
                pass
            self.results_fitting.loc["Salivary Glands"] = self.results_fitting.loc[
                sal_glands
            ].sum()
            self.results_fitting = self.results_fitting.drop(sal_glands)

            if "Skeleton" in self.results_fitting.index:
                skeleton_row = self.results_fitting.loc["Skeleton"]

                # Based on ICRP publication 70; need to be verified for specific cases (different bones have different proportions)
                trabecular = skeleton_row * 0.62
                cortical = skeleton_row * 0.38

                self.results_fitting.loc["Trabecular Bone"] = trabecular
                self.results_fitting.loc["Cortical Bone"] = cortical

                # Drop the original "Skeleton" entry
                self.results_fitting = self.results_fitting.drop("Skeleton")

            self.results_fitting = self.results_fitting.drop(["WholeBody"])

            # Rename
            self.results_fitting = self.results_fitting.rename(
                index={
                    "Bladder": "Urinary Bladder Contents",
                    "BoneMarrow": "Red Marrow",
                }
            )

            self.results_fitting.loc["Red Marrow"][
                "Volume_CT_mL"
            ] = 1170  # TODO volume hardcoded, think about alternatives
            self.results_fitting.loc["RemainderOfBody"]["Volume_CT_mL"] = (
                self.config["PatientWeight_g"]
                - self.results_fitting.loc[
                    ~self.results_fitting.index.isin(["Total Body", "RemainderOfBody"]),
                    "Volume_CT_mL",
                ].sum()
            )

            if "Yes" in self.config["OrganLevel"]["AdditionalOptions"].get(
                "LesionDosimetry"
            ):
                # Separate dosimetry results into lesions and non-lesions
                lesion_mask = self.results_fitting.index.str.contains(
                    "Lesion", case=False, na=False
                )
                self.results_fitting_organs = self.results_fitting[
                    ~lesion_mask
                ].copy()  # all non-lesion entries
                self.results_fitting_lesions = self.results_fitting[
                    lesion_mask
                ].copy()  # only lesion entries

            if "TotalTumorBurden" in self.results_fitting.index:
                self.results_fitting.drop("TotalTumorBurden", axis=0, inplace=True)

            if output_type == "Export":
                fmt = organ_conf["Output"]["ExportFormat"]
                if fmt.lower() == "olinda":
                    self.results_fitting = self.results_fitting.rename(
                        index={"RemainderOfBody": "Total Body"}
                    )
                    self.results_fitting.loc["Total Body"]["Volume_CT_mL"] = (
                        self.config["PatientWeight_g"]
                        - self.results_fitting.loc[
                            ~self.results_fitting.index.isin(
                                ["Total Body", "RemainderOfBody"]
                            ),
                            "Volume_CT_mL",
                        ].sum()
                    )
                elif fmt.lower() == "mirdcalc":
                    print("Not Implemented yet.")
                else:
                    print(f"Export format {fmt} not recognized.")

        return None

    def create_output_file(self, dirname: str, savefile: bool = False) -> None:
        """Create output file(s) for the selected organ-level output format."""
        if self.config["OutputFormat"] == "Olinda":
            self.create_Olinda_file(dirname, savefile)
        else:
            print(
                "In the current version, we only support Olinda as external organ S-value software."
            )  # TODO: other software case files

    def load_svalues(self, filepath):
        """Load S-values CSV and rename Olinda column names to human-friendly organ names."""
        svalues_df = pandas.read_csv(filepath, index_col=0)
        olinda_to_human_source_organ_map = {
            "GB Cont": "Gallbladder Contents",
            "StomCont": "Stomach Contents",
            "Salivary": "Salivary Glands",
            "Red Mar.": "Red Marrow",
            "CortBone": "Cortical Bone",
            "Hrt Wall": "Heart Wall",
            "TrabBone": "Trabecular Bone",
            "HeartCon": "Heart Contents",
            "SI Cont": "Small Intestine",
            "UB Cont": "Urinary Bladder Contents",
            "Tot Body": "Total Body",
        }
        return svalues_df.rename(columns=olinda_to_human_source_organ_map)

    def process_dosimetry(self) -> None:
        """Execute the main dosimetry workflow (prepare data, export or calculate)."""
        organ_conf = self.config["OrganLevel"]
        output_type = organ_conf["Output"]["Type"]  # 'Export' or 'Calculate'

        self.prepare_data()

        print("Processing dosimetry at the organ level of OARs")
        if output_type == "Export":
            fmt = organ_conf["Output"]["ExportFormat"]
            if fmt.lower() == "olinda":
                print("Creating .cas file for Olinda/EXM export.")
                self.create_Olinda_file(
                    dirname=organ_conf["Output"]["ExportDirectory"], savefile=True
                )
            elif fmt.lower() == "mirdcalc":
                print("Not Implemented yet.")
            else:
                print(f"Export format {fmt} not recognized.")

        elif output_type == "Calculate":
            self.calculate_absorbed_dose()

        else:
            raise ValueError(f"Unknown output type: {output_type}")

        if "Yes" in self.config["OrganLevel"]["AdditionalOptions"].get(
            "LesionDosimetry"
        ):
            self.results_dosimetry_lesions = self.apply_sphere_method(
                self.results_fitting_lesions.index.str.contains("Lesion")
            )
            self.results_dosimetry_lesions = self.calculate_ttb()
        if "Yes" in self.config["OrganLevel"]["AdditionalOptions"].get(
            "SalivaryGlandsSeparately"
        ):
            print("Processing dosimetry of salivary glands")
            self.results_dosimetry_salivaryglands = self.apply_sphere_method(
                self.results_salivaryglands
            )

    def calculate_absorbed_dose(self) -> pandas.DataFrame:
        """Calculate absorbed dose per target organ based on model and disintegration data."""
        model_files = {
            "Female": {
                "beta": f'177Lu_S_values_female_{self.config["OrganLevel"]["Calculation"]["SValueSource"].lower()}_BETA.csv',
                "gamma": f'177Lu_S_values_female_{self.config["OrganLevel"]["Calculation"]["SValueSource"].lower()}_GAMMA.csv',
            },
            "Male": {
                "beta": f'177Lu_S_values_male_{self.config["OrganLevel"]["Calculation"]["SValueSource"].lower()}_BETA.csv',
                "gamma": f'177Lu_S_values_male_{self.config["OrganLevel"]["Calculation"]["SValueSource"].lower()}_GAMMA.csv',
            },
        }

        svalues_beta = self.load_svalues(
            path.join(SVALUES_PATH, model_files[self.config["Gender"]]["beta"])
        )
        svalues_gamma = self.load_svalues(
            path.join(SVALUES_PATH, model_files[self.config["Gender"]]["gamma"])
        )

        print("Source organs available in the model:", svalues_beta.columns.tolist())
        print("Source organs present :", self.results_fitting.index.tolist())

        self.source_organs_missing = set(svalues_beta.columns) - set(
            self.results_fitting.index
        )
        print(f"Source organs missing in DataFrame: {self.source_organs_missing}")

        self.results_fitting["TIA_s"] = (
            self.results_fitting["TIA_h"] * 3600
        )  # Time-integrated activity in s

        # Apply S-values and compute dose
        dose_matrix_beta = self.apply_s_value(
            self.results_fitting, svalues_beta, radiation_type="beta"
        )
        dose_matrix_gamma = self.apply_s_value(
            self.results_fitting, svalues_gamma, radiation_type="gamma"
        )

        # Sum doses over source organs to get total dose per target organ
        total_dose_beta = dose_matrix_beta.sum(axis=1)
        total_dose_gamma = dose_matrix_gamma.sum(axis=1)

        dose_df = pandas.DataFrame(
            {
                "Target organ": total_dose_beta.index,
                "AD_beta[Gy/GBq]": total_dose_beta.values,
                "AD_gamma[Gy/GBq]": total_dose_gamma.values,
            }
        )

        # Apply mass scaling
        dose_df = self.perform_mass_scaling(dose_df, self.config["Gender"])

        # Calculate absorbed dose in Gy for injected activity
        dose_df["AD_total[Gy/GBq]"] = (
            dose_df["AD_beta[Gy/GBq]"] + dose_df["AD_gamma[Gy/GBq]"]
        )
        injected_activity = float(self.config["InjectedActivity"])
        dose_df["AD_beta[Gy]"] = dose_df["AD_beta[Gy/GBq]"] / 1000 * injected_activity
        dose_df["AD_gamma[Gy]"] = dose_df["AD_gamma[Gy/GBq]"] / 1000 * injected_activity
        dose_df["AD_total[Gy]"] = dose_df["AD_total[Gy/GBq]"] / 1000 * injected_activity

        dose_df = dose_df.reset_index(drop=True)
        self.df_ad = dose_df.copy()
        return dose_df

    def hollow_organ_correction(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """Apply hollow organ correction to the dose calculations (electrons only)."""
        print("Applying hollow organ correction...")

        pairs = [
            ("Gallbladder Wall", "Gallbladder Contents"),
            ("Stomach Wall", "Stomach Contents"),
            ("Small Intestine", "Small Intestine"),
            ("ULI Wall", "ULI Cont"),
            ("LLI Wall", "LLI Cont"),
            ("Rectum", "Rectum"),
            ("Urinary Bladder Wall", "Urinary Bladder Contents"),
        ]

        for target, source in pairs:
            if target in df.index and source in df.columns:
                df.loc[target, source] *= 2

        return df

    def redistribute_ROB_into_source_organs_missing(
        self, tia_series: pandas.Series
    ) -> pandas.Series:
        """
        If only Total Body TIA is present, we leave it as is, because according to Olinda it represents the Total Body TIA.

        If organs other than Total Body are present, we redistribute the Total Body TIA into missing source organs as it represents the Remainder of the Body.

        This method redistributes the Remainder TIA into source organs that were not segmented and so represent Remainder of the Body.
        """
        if "RemainderOfBody" not in tia_series.index:
            return tia_series

        missing_organs_df = self.organ_masses.loc[
            self.organ_masses.index.isin(self.source_organs_missing)
        ]
        print(f"Missing organs DataFrame:\n{missing_organs_df.index.tolist()}")

        missing_organs_df = missing_organs_df.drop(index="Total Body")

        mass_source_organs = self.organ_masses.loc[
            [org for org in tia_series.index if org != "RemainderOfBody"], "Mass_g"
        ].sum()

        mass_total_body = self.config["PatientWeight_g"]

        tia_ROB = tia_series["TIA_s"]["RemainderOfBody"]

        missing_organs_df["TIA_s"] = (
            missing_organs_df["Mass_g"] / (mass_total_body - mass_source_organs)
        ) * tia_ROB

        missing_organs_df.loc["Heart Contents", "TIA_s"] = 0
        # print % masses
        print(
            f"Masses of missing organs (% of total body mass):\n{(missing_organs_df['Mass_g'] / mass_total_body) * 100}"
        )
        missing_organs_df = missing_organs_df.rename(columns={"Mass_g": "Volume_CT_mL"})

        print(f"Redistributed TIA values for missing organs: {missing_organs_df}")
        print(f"TIA series before redistribution:\n{tia_series}")
        tia_series = tia_series.drop(index="RemainderOfBody")
        tia_series = pandas.concat([tia_series, missing_organs_df])
        tia_series["h"] = tia_series["TIA_s"] / 3600  # Convert seconds to hours

        return tia_series

    def apply_s_value(self, tia_df, s_values, radiation_type) -> pandas.DataFrame:
        """Multiply S-values by TIA to compute dose matrix for radiation type."""
        # Path to organ masses
        masses_path = path.join(MASSES_PATH, "ICRP_mass_male.csv")
        self.organ_masses = pandas.read_csv(masses_path, index_col=0)

        # Handle remainder of the body
        # Redistribute ROB TIA into missing source organs if needed - approach consistent with MIRDcalc software
        if "RemainderOfBody" in tia_df.index:
            if "MirdCalc" in self.config["OrganLevel"]["Calculation"].get(
                "SValueSource", ""
            ):
                tia_df = self.redistribute_ROB_into_source_organs_missing(tia_df)

                common_source_organs = tia_df.index.intersection(s_values.columns)

                print(
                    f"{len(common_source_organs)} source organs: {common_source_organs}"
                )

                if common_source_organs.empty:
                    raise ValueError(
                        "No common source organs between TIA and S-value table."
                    )

                # Subset both dataframes
                tia_series = tia_df.loc[common_source_organs, "TIA_s"]
                s_values_subset = s_values[common_source_organs]

                print(
                    f"Selected source organs for dose calculation: {tia_series.index.tolist()}"
                )
                print(
                    f"Selected target organs for dose calculation: {s_values_subset.index.tolist()}"
                )

                # Multiply S-values by corresponding TIA
                dose_df = s_values_subset.multiply(tia_series, axis=1)

                # Apply hollow organ correction for beta dose
                dose_df = self.hollow_organ_correction(dose_df)

            # Handle find S-value for remainder of the body - approach consistent with Olinda software
            elif "Olinda" in self.config["OrganLevel"]["Calculation"].get(
                "SValueSource", ""
            ):
                common_source_organs = tia_df.index.intersection(s_values.columns)

                print(
                    f"{len(common_source_organs)} source organs: {common_source_organs}"
                )

                if common_source_organs.empty:
                    raise ValueError(
                        "No common source organs between TIA and S-value table."
                    )

                # Subset both dataframes
                tia_series = tia_df.loc[common_source_organs, "TIA_s"]
                s_values_subset = s_values[common_source_organs]

                print(
                    f"Selected source organs for dose calculation: {tia_series.index.tolist()}"
                )
                print(
                    f"Selected target organs for dose calculation: {s_values_subset.index.tolist()}"
                )

                # Multiply S-values by corresponding TIA
                dose_df = s_values_subset.multiply(tia_series, axis=1)

                # ROB
                dose_df["RemainderOfBody"] = 0
                total_body_mass = self.config["PatientWeight_g"]
                ROB_mass = tia_df.loc["RemainderOfBody", "Volume_CT_mL"]
                if radiation_type == "beta":
                    organs = s_values_subset.index.difference(
                        tia_series.index.difference(["Red Marrow", "Osteogenic Cells"])
                    )
                if radiation_type == "gamma":
                    organs = s_values_subset.index
                # adjust source organs so that they are different for gamma and beta radiation (in beta it is total - source organs + bone + skeleton)

                for target_organ in organs:
                    contribution_from_sources = 0
                    for source_organ in s_values.columns.difference(tia_series.index):

                        if target_organ.split()[0] == source_organ.split()[0]:
                            continue
                        if target_organ == "Osteogenic Cells" and source_organ in [
                            "Cortical Bone",
                            "Trabecular Bone",
                            "Red Marrow",
                        ]:
                            continue
                        if target_organ == "Red Marrow" and source_organ in [
                            "Trabecular Bone"
                        ]:
                            continue
                        if target_organ == "Total Body":
                            continue
                        if source_organ == "Total Body":
                            continue

                        source_organ_mass = self.organ_masses.loc[
                            source_organ, "Mass_g"
                        ]

                        s_value_source_to_target = s_values.loc[
                            target_organ, source_organ
                        ]
                        contribution_from_sources += s_value_source_to_target * (
                            source_organ_mass / ROB_mass
                        )

                    s_value_ROB_to_target = (
                        s_values.loc[target_organ, "Total Body"]
                        * (total_body_mass / ROB_mass)
                    ) - contribution_from_sources
                    dose_df.at[target_organ, "RemainderOfBody"] = (
                        s_value_ROB_to_target * tia_df.loc["RemainderOfBody", "TIA_s"]
                    )

            else:
                # TODO
                print("No ROB option selected. Proceeding without ROB handling.")

        return dose_df

    def perform_mass_scaling(
        self, df: pandas.DataFrame, gender: str
    ) -> pandas.DataFrame:
        """Apply mass scaling to absorbed dose calculations based on patient-specific organ masses."""
        masses_path = path.join(MASSES_PATH, f"ICRP_mass_{gender.lower()}_target.csv")
        model_masses_df = pandas.read_csv(masses_path, index_col=0)

        print("Performing mass scaling...")

        for organ in df["Target organ"]:

            if organ in model_masses_df.index and organ in self.results_fitting.index:
                model_mass = model_masses_df.loc[organ, "Mass_g"]
                patient_mass = self.results_fitting.loc[organ, "Volume_CT_mL"]

                if (
                    pandas.notna(patient_mass)
                    and pandas.notna(model_mass)
                    and model_mass > 0
                ):

                    if "AD_beta[Gy/GBq]" in df.columns:
                        scaling_factor_beta = model_mass / patient_mass
                        print(f"Scaling factor for {organ} [β]: {scaling_factor_beta}")
                        df.loc[
                            df["Target organ"] == organ, "AD_beta[Gy/GBq]"
                        ] *= scaling_factor_beta
                        print(
                            f"[β] {organ}: model={model_mass}, patient={patient_mass}, factor={scaling_factor_beta}"
                        )

                    if "AD_gamma[Gy/GBq]" in df.columns:
                        scaling_factor_gamma = (model_mass / patient_mass) ** (2 / 3)
                        df.loc[
                            df["Target organ"] == organ, "AD_gamma[Gy/GBq]"
                        ] *= scaling_factor_gamma
                        print(
                            f"[γ] {organ}: model={model_mass}, patient={patient_mass}, factor={scaling_factor_gamma}"
                        )

        return df

    def create_Olinda_file(self, dirname: str, savefile: bool = False) -> None:
        """Create .cas file that can be exported to Olinda/EXM."""
        this_dir = path.dirname(__file__)
        TEMPLATE_PATH = path.join(this_dir, "olindaTemplates")

        if self.config["Gender"] == "Male":
            template = pandas.read_csv(path.join(TEMPLATE_PATH, "adult_male.cas"))
        elif self.config["Gender"] == "Female":
            template = pandas.read_csv(path.join(TEMPLATE_PATH, "adult_female.cas"))
        else:
            print(
                "Ensure that you correctly wrote patient gender in config file. Olinda supports: Male and Female."
            )

        template.columns = ["Data"]
        match = re.match(r"([a-zA-Z]+)([0-9]+)", self.config["Radionuclide"])
        letters, numbers = match.groups()
        formatted_radionuclide = f"{letters}-{numbers}"

        ind = template[template["Data"] == "[BEGIN NUCLIDES]"].index
        template.loc[ind[0] + 1, "Data"] = formatted_radionuclide + "|"

        for organ in self.results_fitting.index:
            indices = template[template["Data"].str.contains(organ)].index

            source_organ = template.iloc[indices[0]].str.split("|")[0][0]
            mass_phantom = template.iloc[indices[0]].str.split("|")[0][1]
            kinetic_data = self.results_fitting.loc[organ]["TIA_h"]
            mass_data = round(self.results_fitting.loc[organ]["Volume_CT_mL"], 1)

            # Update the template DataFrame
            template.iloc[indices[0]] = (
                f"{source_organ}|{mass_phantom}|{'{:7f}'.format(kinetic_data)}"
            )

            if len(indices) == 2:
                template.iloc[indices[1]] = (
                    f"{source_organ}|{'{:7f}'.format(kinetic_data)}"
                )
            elif len(indices) == 3:
                template.iloc[indices[1]] = f"{source_organ}|{mass_data}"
                template.iloc[indices[2]] = (
                    f"{source_organ}|{'{:7f}'.format(kinetic_data)}"
                )
            else:
                print("Double-check where the organ appears in the template.")

        template = template.replace(
            "TARGET_ORGAN_MASSES_ARE_FROM_USER_INPUT|FALSE",
            "TARGET_ORGAN_MASSES_ARE_FROM_USER_INPUT|TRUE",
        )

        template.columns = [
            "Saved on "
            + datetime.datetime.now().strftime("%m.%d.%Y")
            + " at "
            + datetime.datetime.now().strftime("%H:%M:%S")
        ]

        if savefile:
            if not path.exists(dirname):
                makedirs(dirname)

            template.to_csv(
                str(dirname) + "/" + f"{self.config['PatientID']}.cas", index=False
            )

    def read_results(self, olinda_results_path: str) -> None:
        """Read results from external software based on the output format specified in the configuration."""
        if self.config["OutputFormat"] == "Olinda":
            self.read_Olinda_results(olinda_results_path)

    def read_Olinda_results(self, olinda_results_path: str) -> None:
        """Read .txt results file from Olinda."""
        if not olinda_results_path.endswith(".txt"):
            print(
                "Please export result from Olinda in .txt file."
            )  # TODO: Add .csv extension results
            return

        data = []
        with open(olinda_results_path, "r") as file:

            extract_lines = False
            for line in file:
                stripped_line = line.strip()
                if stripped_line.startswith(
                    "Target Organ,Alpha,Beta,Gamma,Total,ICRP-103 ED"
                ):
                    extract_lines = True
                elif stripped_line.startswith("Target Organ Name,Mass [g],"):
                    extract_lines = False
                if extract_lines:
                    if stripped_line.startswith("Effective Dose"):
                        stripped_line = stripped_line.replace(
                            "Effective Dose", "Effective Dose,,,,"
                        )
                    stripped_line = stripped_line.rstrip(",")
                    data.append(stripped_line.split(","))

        if not data:
            print("No relevant data found in the file.")
            return

        df_ad = pandas.DataFrame(data[1:], columns=data[0])
        df_ad = df_ad.set_index("Target Organ")
        df_ad = df_ad.dropna(axis=0)
        df_ad["Total"] = pandas.to_numeric(df_ad["Total"], errors="coerce")
        df_ad["Total"].fillna(0, inplace=True)
        df_ad["AD[Gy]"] = df_ad["Total"] * float(self.config["InjectedActivity"]) / 1000
        df_ad = df_ad.rename(columns={"Total": "AD[Gy/GBq]"})
        self.df_ad = df_ad

    def compute_dose(self):
        """Compute Time Integrated Activity."""
        self.compute_tia()
