import pandas
from typing import Tuple
from doodle.fits import fit_monoexp

def blood_based_selfdose(df: pandas.DataFrame, decay_constant: float) -> Tuple[float, float]:

    """
    Computes the bone-marrow self-dose due to activity concentration in the bone-marrow, following
    EANM Guidelines: https://www.eanm.org/publications/guidelines/EJNMMI_Bone_Marrow_Dosimetry_06_2010.pdf

    Returns:
    Dose_BM <- Extra Cellular Fluid, Dose_BM <- Blood Cells

    Limitations:
     - Assumes mono-exponential model of Time Activity Concentration vs Time.
     - The activity within the bone-marrow cells must be calculated from quantitative imaging
       in different regions of the bone barrow.

    """

    # TODO: Have S-factors stored and accesible somewhere
    s_bm_bm = 1 
    m_bm = 1

    # Parameters:
    RMECFF = 0.19
    HCT = 1  # Can be used to convert from Ablood to Aplasma.
    RMBLR = 0.36  # From Sgouros; J Nucl Med 1993;34(4):689–94 ?
                  # A RMBLR = 1 also found for Lu-177 peptides?


    # Compute Time-Integrated Activity Concetrations. 
    TAC_blood_param, _ = fit_monoexp(time=df["Time"].to_numpy(),
                             activity=df["Activity_Conc"].to_numpy(),
                             decayconst=decay_constant,
                             monoguess=(df["Time"].iloc[0], decay_constant))
    TIAC_blood = TAC_blood_param[0] / TAC_blood_param[1]  # Use analytical formula for mono-exponential decay.
    

    # Compute Dose: Assume TIAC_blood = TIAC_plasma
    dose_ECF = TIAC_blood * RMECFF * m_bm * s_bm_bm  # Should use Plasma TIAC.
    dose_Blood = TIAC_blood * RMBLR * m_bm * s_bm_bm

    return dose_ECF, dose_Blood