import numpy as np

def decay_act(a_initial,delta_t,half_life):

    return a_initial * np.exp(-np.log(2)/half_life * delta_t)
