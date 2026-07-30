import numpy as np
from qutip import destroy, basis, expect, mesolve
from config import dt
from utils import make_time

class QMLIFNeuron:
    def __init__(
        self,
        *,
        name = "",
        N = 20,
        Cm = 1.0,
        Ron = 1e3,
        Roff = 1e5,
        q_max = 1.0,
        V_th = 0.02,
        tau_ref = 0.1,
    ):
        """
        Quantum Memristive Leaky Integrate-and-Fire neuron model
        """
        self.name = name
        self.N = N
        self.Cm = Cm
        self.Ron = Ron
        self.Roff = Roff
        self.q_max = q_max
        self.V_th = V_th
        self.tau_ref = tau_ref

        a = destroy(N)
        self.a = a
        hbar = 1.0
        Z0_guess = (Ron + Roff) / 2.0
        self.Z0_guess = Z0_guess

        phi_zpf = np.sqrt(hbar * Z0_guess / 2.0)
        Q_zpf = np.sqrt(hbar / (2.0 * Z0_guess))

        self.phi_op = phi_zpf * (a + a.dag())
        Q_op = 1j * Q_zpf * (a.dag() - a)
        self.V_op = Q_op / Cm
        self.n_op = a.dag() * a

        omega0 = 1.0 / np.sqrt(Cm * Z0_guess)
        self.H0 = hbar * omega0 * (a.dag() * a + 0.5)

        self.rho = basis(N, 0) * basis(N, 0).dag()

        self.q_mem = 0.0
        self.Z0 = Z0_guess
        self.last_spk = -np.inf
        self.t_now = 0.0

    def step(self, I = 0.0, gamma = 0.0):
        in_ref = self.t_now < self.last_spk + self.tau_ref  # refractory period

        I_drive = 0.0 if in_ref else float(I)       # excitatory contribution (supressed during refractory period)
        gamma_inh = 0.0 if in_ref else float(gamma) # inhibitory contribution (supressed during refractory period)

        # Hamiltonian (includes: external current & excitation)
        H_t = self.H0 - I_drive * self.phi_op

        # Dissipation (includes: membrane leak & inhibition)
        gamma_m = 1.0 / (self.Z0 * self.Cm)
        c_ops = [np.sqrt(gamma_m + gamma_inh) * self.a]

        # solve
        result = mesolve([H_t], self.rho, [0.0, dt], c_ops, e_ops=[])
        self.rho = result.states[-1]

        # membrane voltage and current
        Vn = float(np.real(expect(self.V_op, self.rho)))
        In = Vn / self.Z0

        mu = 0  # spike amplitude
        spiked = False

        if not in_ref:
            # update memristive state
            self.q_mem += In * dt
            self.q_mem = float(np.clip(self.q_mem, 0.0, self.q_max))
            self.Z0 = self.Ron * (self.q_mem / self.q_max) + self.Roff * (1.0 - self.q_mem / self.q_max)
            self.Z0 = float(np.clip(self.Z0, self.Ron, self.Roff))

            # membrane voltage exceeds threshold
            if Vn > self.V_th:
                # pre-spike amplitude BEFORE collapsing to |0⟩
                mu = float(np.real(expect(self.n_op, self.rho)))

                # collapse to |0⟩
                self.rho = basis(self.N, 0) * basis(self.N, 0).dag()
                
                # start refractory period
                self.last_spk = self.t_now
                spiked = True

        self.t_now += dt

        return Vn, spiked, mu, self.q_mem