from typing import Optional
from doodle.dosimetry.olinda import load_phantom_mass
import os

def bm_scaling_factor(gender: str = "Male", 
                      mass_bm: Optional[float] = None,
                      hematocrit: Optional[float] = None) -> float:
    """Determine Bone Marrow factor to estimate integrated activity in bone-marrow from 
    activity concentration in blood.
    """
    
    RMBLR = 1.0  # Activity concentration in Red Marrow over blood. Generally 1 for PSMA and SSRT.
    
    if hematocrit is not None:
        RMBLR = 0.19 / (1 - hematocrit)

    # If the bone-marrow mass is unknown, use phantom mass.
    if mass_bm is None:
            mass_bm = load_phantom_mass(gender=gender, organ="Red Marrow")  # in Grams.

    print(f"Phantom BM Mass (in grams): {mass_bm}")
    print(f"RMBLR = {RMBLR}")

    return RMBLR * mass_bm
