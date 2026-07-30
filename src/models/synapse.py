import numpy as np

class Synapse:
    def __init__(
        self,
        w,          # weight
        tau_syn,    # time constant
    ):
        self.w = w
        self.tau_syn = tau_syn
        self.norm = np.exp(1)   # for normalising alpha kernel

    def _alpha_kernel(self, delta_t):
        if delta_t < 0.0:
            return 0.0
        return (delta_t/self.tau_syn) * np.exp(-delta_t/self.tau_syn)

    def _conductance(self, t_now, spike_log):
        if not spike_log:
            return 0.0
        
        # include significant spikes only
        limit = 10 * self.tau_syn
        spike_log[:] = [(t_spk, mu) for t_spk, mu in spike_log if (t_now-t_spk) < limit]

        # calculate conductance
        g = sum(
            mu * self._alpha_kernel(t_now-t_spk)
            for t_spk, mu in spike_log
        )
        return g
    
    # excitatory_current and inhibitory_leak are identical
    # they are keept as 2 separate methods intentionally
    # leaving opportunity for  further modifictions and naming clarity in A_exc_B_inh_A.py 
    def excitatory_current(self, t_now, spike_log):
        """
        excitatory current = conductance scaled by the excitatory weight
        """
        if self.w == 0 or self.tau_syn == 0:
            return 0.0
        # alpha kernel normalisation applied here
        return self.norm * self.w * self._conductance(t_now, spike_log)
    
    def inhibitory_leak(self, t_now, spike_log):
        """
        inhibitory leak = conductance scaled by the inhibitory weight
        """
        if self.w == 0 or self.tau_syn == 0:
            return 0.0
        # alpha kernel normalisation applied here
        return self.norm * self.w * self._conductance(t_now, spike_log)