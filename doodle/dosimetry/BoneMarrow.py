import pandas
from typing import Tuple


def blood_based_selfdose(df: pandas.DataFrame) -> Tuple[float, float]:

    """
    Computes the bone-marrow self-dose due to activity concentration in the bone-marrow, following
    EANM Guidelines: https://www.eanm.org/publications/guidelines/EJNMMI_Bone_Marrow_Dosimetry_06_2010.pdf

    Returns:
    Dose_BM <- Extra Cellular Fluid, Dose_BM <- Blood Cells

    """

    # TODO: Have S-factors stored and accesible somewhere
    s_bm_bm = 1 
    m_bm = 1

    # Parameters:
    RMECFF = 0.19
    HTC = 1
    RMBLR = 0.36  # From Sgouros; J Nucl Med 1993;34(4):689–94 ?
                  # A RMBLR = 1 also found for Lu-177 peptides?


    # TODO: Compute Time-Integrated Activity Concetrations


    # Compute Dose 
    dose_ECF = TIA_plasma * RMECFF * m_bm * s_bm_bm
    dose_Blood = TIA_blood * RMBLR * m_bm * s_bm_bm

    return dose_ECF, dose_Blood