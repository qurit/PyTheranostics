import numpy as np

def perc_diff(measured_value,expected_value,decimals=2):

    return np.round((measured_value - expected_value) / expected_value *100,2)