import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed

from models.qmlif import QMLIFNeuron
from models.synapse import Synapse
from models.current import sinusoidal_current, constant_current
from models.A_exc_B_inh_A import A_exc_B_inh_A
from analysis.metrics import *

def _sinusoidal_point(
    I0,
    f,
    T,
    transient,
    neuronA_param,
    neuronB_param,
    synapse_exc_param,
    synapse_inh_param,
):
    """
    run simulation under sinusoidal drive I(t) = I₀⋅sin(2π⋅f⋅t)
    return its phase-locking metrics for neuron A
    """

    network = A_exc_B_inh_A(
        QMLIFNeuron(**neuronA_param),
        QMLIFNeuron(**neuronB_param),
        Synapse(**synapse_exc_param),
        Synapse(**synapse_inh_param),
    )

    input_current = sinusoidal_current(T, T, I0, f)

    simulation_data = network.simulate(T, input_current)

    # transient: initial settling time discarded before measuring locking
    T_analysis = T - transient

    # get all spikes
    spk_times_A = spike_times(simulation_data["spkA"])

    # drop transient spikes
    spk_times_A = spk_times_A[spk_times_A >= transient] - transient

    n_spk_A = len(spk_times_A) # number of spikes of A

    fr_A = n_spk_A / T_analysis # firing rate of A

    plv_11_A = plv_11(spk_times_A, f) # 1:1 phase-locking value between the external drive and A
    plv_nm_A, n_A, m_A = plv_max_nm(spk_times_A, f, T_analysis) # best n:m locking

    return {
        "f": float(f),
        "I0": float(I0),
        "n_spikes_A": n_spk_A,
        "last_spk_time_A": float(spk_times_A[-1]) if n_spk_A else np.nan,
        "fr_A": fr_A,
        "plv_11_A": plv_11_A,
        "plv_nm_A": plv_nm_A,
        "lock_n_A": n_A,
        "lock_m_A": m_A,
    }

def _constant_point(
    I0,
    T,
    transient,
    neuronA_param,
    neuronB_param,
    synapse_exc_param,
    synapse_inh_param,
):
    """
    run simulation under constant drive I(t) = I₀
    return metrics for neuron A
    """

    network = A_exc_B_inh_A(
        QMLIFNeuron(**neuronA_param),
        QMLIFNeuron(**neuronB_param),
        Synapse(**synapse_exc_param),
        Synapse(**synapse_inh_param),
    )
    
    input_current = constant_current(T, T, I0)

    simulation_data = network.simulate(T, input_current)

    # transient: initial settling time discarded before measuring locking
    T_analysis = T - transient

    # get all spikes of A
    spk_times_A = spike_times(simulation_data["spkA"])

    # drop transient spikes
    spk_times_A = spk_times_A[spk_times_A >= transient] - transient

    n_spk_A = len(spk_times_A)    # number of spikes

    if n_spk_A >= 2:      # ISI statistics need at least two spikes
        isis = np.diff(spk_times_A)
        isi_m = float(isis.mean())
        isi_cv = float(isis.std(ddof=1) / isi_m) if isi_m > 0 else np.nan
    else:
        isi_m = isi_cv = np.nan

    return {
        "I0": float(I0),
        "n_spikes_A": int(n_spk_A),
        "last_spk_time_A": float(spk_times_A[-1]) if n_spk_A else np.nan,
        "fr_A": n_spk_A / T_analysis,
        "isi_mean_A": isi_m,
        "isi_cv_A": isi_cv,
    }


def run_constant_locking_map(
    I0_values,
    T,
    transient,
    neuronA_param,
    neuronB_param,
    synapse_exc_param,
    synapse_inh_param,
    n_workers,
):
    """
    sweep constant-current amplitude I0 in parallel
    return a DataFrame of firing statistics for each sweep
    """

    # use parallel execution to handle high number of sweeps
    results = Parallel(n_jobs=n_workers, prefer="processes", verbose=10)(
        delayed(_constant_point)(
            I0,
            T,
            transient,
            neuronA_param,
            neuronB_param,
            synapse_exc_param,
            synapse_inh_param,
        )
        for I0 in I0_values
    )

    return pd.DataFrame(results).sort_values("I0")

def run_sinusoidal_locking_map(
    I0_values,
    f_values,
    T,
    transient,
    neuronA_param,
    neuronB_param,
    synapse_exc_param,
    synapse_inh_param,
    n_workers,
):
    """
    sweep (f, I0) grid under sinusoidal drive in parallel
    return a DataFrame of locking metrics for each (f, I0) point
    """

    grid = list(itertools.product(I0_values, f_values))

    print(
        f"\nLOCKING MAP\n"
        f"  grid   : {len(f_values)} f × {len(I0_values)} I₀ = {len(grid)} points\n"
        f"  f      : [{f_values.min():.3f}, {f_values.max():.3f}] Hz\n"
        f"  I₀     : [{I0_values.min():.3f}, {I0_values.max():.3f}]\n"
        f"  w_exc  : {synapse_exc_param['w']}\n"
        f"  tau_exc: {synapse_exc_param['tau_syn']}\n"
        f"  w_inh  : {synapse_inh_param['w']}\n"
        f"  tau_inh: {synapse_inh_param['tau_syn']}\n"
        f"  T      : {T}s  (transient: {transient}s)\n"
        f"  workers: {n_workers}\n"
    )

    # use parallel execution to handle high number of sweeps
    results = Parallel(n_jobs=n_workers, prefer="processes", verbose=10)(
        delayed(_sinusoidal_point)(
            I0,
            f,
            T,
            transient,
            neuronA_param,
            neuronB_param,
            synapse_exc_param,
            synapse_inh_param,
        )
        for I0, f in grid
    )

    results_df = pd.DataFrame(results)

    return results_df
