import numpy as np
import pandas as pd
from utils import make_time
from math import gcd
from config import dt

def spike_count(spk):
    return int(sum(spk))

def spike_times(spk):
    return np.where(spk)[0] * dt

def spike_frequency(spk, T):
    return spike_count(spk)/T

def isi(spk):
    return np.diff(spike_times(spk))

def isi_mean(spk):
    return np.mean(isi(spk))

def isi_std(spk, ddof=1):
    return np.std(isi(spk), ddof=ddof)

def isi_cv(spk):
    m = isi_mean(spk)
    s = isi_std(spk)
    return s/m if m > 0 else np.nan

def adaptation_index(spk, k=4):
    if k < 2: raise ValueError("k must be >= 2")
    isi_values = isi(spk)
    N = len(isi_values) # number of ISIs = = number of spikes - 1
    if N-k <= 0:
        return np.nan
    A = sum(
        (isi_values[i] - isi_values[i - 1]) / (isi_values[i] + isi_values[i - 1])
        for i in range(k, N)
    )
    return A / (N - k)

def calculate_metrics_firing_patterns(spk):
    if spike_count(spk) == 0:
        return None
    return {
        "n_spikes": spike_count(spk), 
         "t_first": float(spike_times(spk)[0]),
         "t_last": float(spike_times(spk)[-1]), 
         "mean_isi": float(isi_mean(spk)),
         "cv_isi": float(isi_cv(spk)), 
         "a": float(adaptation_index(spk)), 
    }

def plv_11(spk_times, f, min_spikes=10):
    if len(spk_times) < min_spikes: # not enough spikes
        return np.nan
    phases = 2.0 * np.pi * f * spk_times
    return float(np.abs(np.mean(np.exp(1j * phases))))

def plv_max_nm(spk_abs, f, T, nm_max=4, min_spikes=10):
    n_spk = len(spk_abs)
    if n_spk < min_spikes or f <= 0: # not enough spikes or invalid drive frequency
        return np.nan, np.nan, np.nan

    spk = np.asarray(spk_abs, dtype=float)

    fr = n_spk/T
    r_obs = fr/f

    candidates = []
    for n in range(1, nm_max + 1):
        for m in range(1, nm_max + 1):
            # only consider ratios in lowest terms to avoid redundancy (2:4 is the same locking as 1:2)
            if gcd(n, m) != 1:
                continue

            macro_period = m/f  # m drive cycles after which the n:m pattern repeats
            if T/macro_period < 5: # too few repetitions can lead to unreliable PLV
                continue

            # prune candidates whose n/m ratio deviates more than 2 times from observed firing-rate ratio.
            # this allows estimation of noise while rejecting clearly incompatible locking ratios
            # tighter bounds risk false negatives, looser bounds admit unlike matches
            if r_obs > 0 and not (0.5 * r_obs <= n / m <= 2.0 * r_obs):
                continue

            candidates.append((n, m))

    if not candidates:
        return np.nan, np.nan, np.nan

    best_plv = -1.0
    best_n, best_m = np.nan, np.nan
    two_pi_f = 2.0 * np.pi * f

    # tests if spikes are phase-consistent across m-cycle windows
    for n, m in candidates:
        psi = (n * two_pi_f * spk) % (2.0 * np.pi * m) # map each spike time to a phase in the cycle [0, 2π·m)
        plv = float(np.abs(np.mean(np.exp(1j * psi / m)))) # normalise to [0, 2π)

        # find best plv
        if plv > best_plv:
            best_plv = plv
            best_n, best_m = n, m

    return best_plv, best_n, best_m

