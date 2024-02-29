from os import path
from pathlib import Path
import datetime
import numpy
import pandas


def load_s_values(gender: str, radionuclide: str) -> pandas.DataFrame:
    """Load S-values Dataframes"""
    path_to_sv = Path(f"./phantomdata/{radionuclide}-{gender}-Svalues.csv")
    if not path_to_sv.exists():
        raise FileExistsError(f"S-values for {gender}, {radionuclide} not found. Please make sure"
                              " gender is ['Male', 'Female'] and radionuclide SymbolMass e.g., Lu177")

    s_df = pandas.read_csv(path_to_sv)
    s_df.set_index(keys=["Target"], drop=True, inplace=True)
    s_df = s_df.drop(labels=["Target"], axis=1)

    return s_df


def load_phantom_mass(gender: str, organ: str) -> float:
    """Load the mass of organs in the standar ICRP Male/Female phantom"""
    phantom_data_path = path.dirname(__file__) + "/phantomdata/human_phantom_masses.csv"
    masses = pandas.read_csv(phantom_data_path)

    if organ not in masses["Organ"].to_list():
        raise ValueError(f"Organ {organ} not found in phantom data.")

    return masses.loc[masses["Organ"] == organ].iloc[0][gender]

