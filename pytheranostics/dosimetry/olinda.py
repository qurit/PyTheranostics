"""Helpers for reading Olinda/EXM phantom tables shipped with PyTheranostics."""

import pandas

from pytheranostics.shared.resources import resource_path


def load_s_values(gender: str, radionuclide: str) -> pandas.DataFrame:
    """Load the S-value table for a gender/radionuclide pair."""
    relative_path = f"phantomdata/{radionuclide}-{gender}-Svalues.csv"
    try:
        with resource_path("pytheranostics.dosimetry", relative_path) as path_to_sv:
            s_df = pandas.read_csv(path_to_sv)
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        raise FileNotFoundError(
            f"S-values for {gender}, {radionuclide} not found. Ensure gender is "
            "one of ['Male', 'Female'] and radionuclide uses the SymbolMass format (e.g., Lu177)."
        ) from exc
    s_df.set_index(keys=["Target"], drop=True, inplace=True)
    s_df = s_df.drop(labels=["Target"], axis=1)

    return s_df


def load_phantom_mass(gender: str, organ: str) -> float:
    """Return the ICRP phantom mass for the requested organ and gender."""
    with resource_path(
        "pytheranostics.dosimetry", "phantomdata/human_phantom_masses.csv"
    ) as phantom_data_path:
        masses = pandas.read_csv(phantom_data_path)

    if organ not in masses["Organ"].to_list():
        raise ValueError(f"Organ {organ} not found in phantom data.")

    return masses.loc[masses["Organ"] == organ].iloc[0][gender]
