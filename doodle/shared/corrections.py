

def tew_scatt(window_dic):

    Cp = {}
    
    ls_width = window_dic['low_scatter']['width']
    us_width = window_dic['upper_scatter']['width']
    pp_width = window_dic['photopeak']['width']

    for det in window_dic['photopeak']['counts']:

        Cs = (window_dic['low_scatter']['counts'][det] / ls_width + window_dic['upper_scatter']['counts'][det] / us_width) * pp_width / 2
        Cpw = window_dic['photopeak']['counts'][det]

        Cp[det] = Cpw - Cs

    # return the primary counts after scatter correction
    return Cp