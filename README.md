# QMLIF Neurons Simulations

Simulation and analysis code for a bachelor's thesis on quantum memristive leaky fire-and-integrate (QMLIF) neurons under excitatory and inhibitory coupling.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

### LaTeX requirement

Figure generation (`src/analysis/plot.py`) uses matplotlib's LaTeX text. A LaTeX distribution must be installed separately.

If LaTeX is not available, disable rendering by editing the top of `plot.py`:

```python
plt.rc('text', usetex=False)
```

Labels will fall back to matplotlib's default font; figures will still generate correctly.

## Structure

```
src/
  models/
    qmlif.py          # core QMLIF neuron
    synapse.py        # excitatory and inhibitory synaptic model
    current.py        # input current (sinusoidal, constant, pulse, damped)
    A_exc_B_inh_A.py  # two-neuron network: A excites B, B inhibits A, input current applied to A
  analysis/
    metrics.py        # spike statistics, ISI, CV, PLV, locking ratio
    locking_map.py    # parallel parameter sweeps for phase-locking maps
    plot.py           # all figure-generating functions
  config.py           # shared simulation constants
  utils.py            # helpers
notebooks/            # experiments
results/              # saved CSV outputs
fig/                  # saved figures
```

## Notebooks (Experiments)

| Notebook | Description |
|---|---|
| `neurons_behaviour` | Neurons response under a sinusoidal external current |
| `firing_patterns` | Firing modes classification |
| `parameter_space_exploration` | Analysis of range of parameter configurations |
| `phase_locking_search` | Search for phase-locking regimes |
| `phase_locking_analysis` | Analysis and visualisation of locking maps |
| `mu_analysis` | Synaptic amplitude analysis |
