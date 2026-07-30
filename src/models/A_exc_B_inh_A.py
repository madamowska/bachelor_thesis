import numpy as np
from utils import make_time

class A_exc_B_inh_A:
    def __init__(
        self,
        neuronA,
        neuronB,
        synapse_exc,
        synapse_inh,
    ):
        """
        input current ──► [Neuron A] ───(excitatory)───► [Neuron B]
                              ▲                               │
                              └─────────(inhibitory)──────────┘
        """
        self.neuronA = neuronA
        self.neuronB = neuronB
        self.synapse_exc = synapse_exc
        self.synapse_inh = synapse_inh

    def simulate(
        self,
        T,
        input_current,
    ):
        _, time = make_time(T)
        
        # storage 
        memA_log, memB_log = [], []         # normalized membrane voltage
        spkA_log, spkB_log = [], []         # spike
        I_exc_log, gamma_inh_log = [], []   # excitatory current and inhibitory leak
        muA_log, muB_log = [], []           # spike amplitude
        q_memA_log, q_memB_log = [], []     # memristor charge state
        spkA_list, spkB_list = [], []       # list of spike times and mu (for synapse conductance calculation)

        for i, t in enumerate(time):
            # input current -> A
            I_in = input_current[i]

            # synapse A -> B
            I_exc = self.synapse_exc.excitatory_current(t, spkA_list)
            
            # synapse B -> A
            gamma_inh = self.synapse_inh.inhibitory_leak(t, spkB_list)

            # neurons response
            vA, spikedA, muA, q_memA = self.neuronA.step(I = I_in, gamma = gamma_inh)
            vB, spikedB, muB, q_memB = self.neuronB.step(I = I_exc)

            # log spikes for synapse conductance calculation
            if spikedA: spkA_list.append((t, muA))
            if spikedB: spkB_list.append((t, muB))

            # save values
            memA_log.append(vA/self.neuronA.V_th)
            memB_log.append(vB/self.neuronB.V_th)
            spkA_log.append(spikedA)
            spkB_log.append(spikedB)
            I_exc_log.append(I_exc)
            gamma_inh_log.append(gamma_inh)
            muA_log.append(muA)
            muB_log.append(muB)
            q_memA_log.append(q_memA)
            q_memB_log.append(q_memB)

        return {
            "T": T,
            "input_current": np.array(input_current),
            "memA": np.array(memA_log),
            "memB": np.array(memB_log),
            "spkA": np.array(spkA_log),
            "spkB": np.array(spkB_log),
            "I_exc": np.array(I_exc_log),
            "gamma_inh": np.array(gamma_inh_log),
            "muA": np.array(muA_log),
            "muB": np.array(muB_log),
            "q_memA": np.array(q_memA_log),
            "q_memB": np.array(q_memB_log),
        }
    
