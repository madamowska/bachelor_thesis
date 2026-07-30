import numpy as np
from utils import make_time

# T:      total simulation time
# T_in:   time interval during which the external current is applied

# Currently, T_in = T in all simulations (the external current is present for the entire simulation). 
# T_in is included for future studies involving self-sustained oscillations, 
# where the input may be removed before the end of the simulation.

def sinusoidal_current(
        T,
        T_in,
        I0,
        f
    ):
    """
    I(t) = I₀⋅sin(2π⋅f⋅t)
    """
    n, t = make_time(T)
    input_current = np.zeros(n, dtype=float)
    input_current[t < T_in] = I0 * np.sin(2 * np.pi * f * t[t < T_in])
    return input_current

def constant_current(
        T,
        T_in,
        I0
    ):
    """
    I(t) = I₀
    """
    n, t = make_time(T)
    input_current = np.zeros(n, dtype=float)
    input_current[t < T_in] = I0
    return input_current

def pulse_current(
        T,
        T_in,
        I0,
        pulse_duration,
        pulse_times
    ):
    """
    I(t) = I₀ for pulses duration, starting at each pulse time
    """
    n, t = make_time(T)
    input_current = np.zeros(n, dtype=float)
    for pulse_time in pulse_times:
        mask = (t >= pulse_time) & (t < pulse_time + pulse_duration) & (t < T_in)
        input_current[mask] = I0
    return input_current

def pulse_current_2(
        T,
        T_in,
        I1,
        I2,
        pulse1_duration,
        pulse2_duration,
        pulse1_times,
        pulse2_times
    ):
    """
    I(t) =  I₁ for pulse duration 1, starting at each pulse time 1
            I₂ for pulse duration 2, starting at each pulse time 2
    """
    n, t = make_time(T)
    input_current = np.zeros(n, dtype=float)
    for pulse_time in pulse1_times:
        mask = (t >= pulse_time) & (t < pulse_time + pulse1_duration) & (t < T_in)
        input_current[mask] = I1
    for pulse_time in pulse2_times:
        mask = (t >= pulse_time) & (t < pulse_time + pulse2_duration) & (t < T_in)
        input_current[mask] = I2
    return input_current

def pulse_current_smooth(
        T,
        T_in,
        I0,
        pulse_duration,
        pulse_times,
        tau_rise,
        tau_decay
    ):
    """
    I(t) = I₀ for pulses duration, starting at each pulse time with exponential raide and fall
    """
    n, t = make_time(T)
    input_current = np.zeros(n, dtype=float)
    for pulse_time in pulse_times:
        rise = 1.0 / (1.0 + np.exp(-(t - pulse_time) / tau_rise))
        fall = 1.0 / (1.0 + np.exp(-(t - (pulse_time + pulse_duration)) / tau_decay))
        pulse = I0 * (rise - fall)
        pulse[t >= T_in] = 0.0
        input_current += pulse
    return input_current

def damped_sinusoidal_current(
        T,
        T_in,
        I0,
        f,
        tau
    ):
    """
    I(t) = I₀·exp(-t/τ)·sin(2π·f·t)
    """
    n, t = make_time(T)
    input_current = np.zeros(n, dtype=float)
    mask = t < T_in
    tt = t[mask]
    input_current[mask] = (I0 * np.exp(-tt/tau) * np.sin(2 * np.pi * f * tt))
    return input_current

