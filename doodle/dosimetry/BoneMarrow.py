import json
import math

import pandas
from dosimetry.olinda import load_phantom_mass, load_s_values

from doodle.fits import fit_tac


def marrow_selfdose(df_blood: pandas.DataFrame, radionuclide: str, gender: str, exp_order: int) -> float:

    """
    Computes the bone-marrow self-dose due to activity concentration in the red-marrow, following
    EANM Recommendations. This is the blood-based method, Equation (13) in:

    https://link.springer.com/article/10.1007/s00259-022-05727-7

    Returns:
    Dose (RM <-- RM)

    """
    # Load Radionuclide data
    with open("../data/isotopes.json", "r") as rad_data:
        radnuclide_data = json.load(rad_data)
    
    if radionuclide not in radnuclide_data:
        raise ValueError(f"Data for {radionuclide} is not available.")
    # TODO: half_life should be in seconds in the database.
    decay_constant = math.log(2) / (radnuclide_data[radionuclide]["half_life"] * 24 * 3600)

    # Load S-Values
    s_values = load_s_values(gender=gender, radionuclide=radionuclide)
    s_bm_bm =  s_values["Red Marrow"]["Red Mar."] # Red-Marrow -> Red-Marrow S-Factor, Adult in mSv/MBq s.
    m_bm = load_phantom_mass(gender=gender, organ="Red  Marrow")  # in Grams.

    # Parameters:
    RMBLR = 1.0  # Activity concentration in Red Marrow over blood. Generally 1 for PSMA and SSRT.

    # Compute Time-Integrated Activity Concetrations. 
    TAC_blood_param, _ = fit_tac(time=df_blood["Time"].to_numpy(),
                                activity=df_blood["Activity_Conc"].to_numpy(),
                                decayconst=decay_constant,
                                exp_order=exp_order)
    TIAC_blood = TAC_blood_param[0] / TAC_blood_param[1]  # Use analytical formula for mono-exponential decay.


    # Compute Dose: Activity Concentration should be in MBq / grams.
    dose_rm_rm = TIAC_blood * RMBLR * m_bm * s_bm_bm

    return dose_rm_rm