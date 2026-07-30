import numpy as np
from config import dt

def make_time(T):
    """
    for the simulation of total time T generate:
        - time vector from 0 to T-dt with step size dt
        - number of discrete simulation steps
    """
    steps = int(T/dt)
    time = np.arange(steps) * dt
    return steps, time