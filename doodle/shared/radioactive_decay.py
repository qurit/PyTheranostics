import numpy as np
from datetime import datetime

def decay_act(a_initial,delta_t,half_life):

    return a_initial * np.exp(-np.log(2)/half_life * delta_t)




def get_activity_at_injection(injection_date,pre_inj_activity,pre_inj_time,post_inj_activity,post_inj_time,injection_time,half_life):

    # Pass half-life in seconds

    # Set the times and the time deltas to injection time
    pre_datetime = datetime.strptime(injection_date + pre_inj_time + '00.00','%Y%m%d%H%M%S.%f')
    post_datetime = datetime.strptime(injection_date + post_inj_time + '00.00','%Y%m%d%H%M%S.%f')
    inj_datetime = datetime.strptime(injection_date + injection_time + '00.00','%Y%m%d%H%M%S.%f')

    delta_inj_pre = (inj_datetime - pre_datetime).total_seconds()
    delta_post_inj = (inj_datetime - post_datetime).total_seconds()

    pre_activity = decay_act(pre_inj_activity,delta_inj_pre,half_life)
    post_activity = decay_act(post_inj_activity,delta_post_inj,half_life)

    injected_activity = pre_activity - post_activity

    return inj_datetime, injected_activity