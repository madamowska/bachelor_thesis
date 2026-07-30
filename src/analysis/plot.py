import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from utils import make_time

# ~~~ SETTINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BLUE = "#070090"
RED = "#A90000"
GRAY = "#585C5E"
BLACK = "#000000"
WHITE = "#FFFFFF"
LIGHT_BLUE = "#C9D7FF"

BLUE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "blue_theme",
    [WHITE, LIGHT_BLUE, BLUE]
)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Times New Roman')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~ NETWORK RESPONSE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def network_response(data, title=None):
    _, time = make_time(data["T"])
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 1.5, 1.5, 1.5])

    # input current
    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.plot(time, data["input_current"], color=BLACK)
    ax_input.set_xlim(0, time[-1])
    ax_input.set_title("External Input Current")
    ax_input.set_ylabel(r"$I_\mathrm{in}$")

    # inhibitory leak
    ax_inh = fig.add_subplot(gs[1, 0])
    ax_inh.plot(time, data["gamma_inh"], color=BLUE)
    ax_inh.set_xlim(0, time[-1])
    ax_inh.set_title("Inhibitory Leak")
    ax_inh.set_ylabel(r"$\gamma_\mathrm{inh}$")

    # excitatory current
    ax_exc = fig.add_subplot(gs[1, 1])
    ax_exc.plot(time, data["I_exc"], color=RED)
    ax_exc.set_xlim(0, time[-1])
    ax_exc.set_title("Excitatory Current")
    ax_exc.set_ylabel(r"$I_\mathrm{exc}$")

    # neuron A membrane potential
    ax_mem_A = fig.add_subplot(gs[2, 0])
    ax_mem_A.plot(time, data["memA"], color=BLUE)
    ax_mem_A.axhline(1.0, ls='--', color=GRAY, alpha=0.5)
    ax_mem_A.set_xlim(0, time[-1])
    ax_mem_A.set_title("Neuron A: Membrane Potential")
    ax_mem_A.set_ylabel(r"$V / V_\mathrm{th}$")

    # neuron B membrane potential
    ax_mem_B = fig.add_subplot(gs[2, 1])
    ax_mem_B.plot(time, data["memB"], color=RED)
    ax_mem_B.axhline(1.0, ls='--', color=GRAY, alpha=0.5)
    ax_mem_B.set_xlim(0, time[-1])
    ax_mem_B.set_title("Neuron B: Membrane Potential")
    ax_mem_B.set_ylabel(r"$V / V_\mathrm{th}$")

    # A spikes
    ax_spikes_A = fig.add_subplot(gs[3, 0])
    ax_spikes_A.vlines(time[data["spkA"]], 0.5, 1.5, color=BLUE)
    ax_spikes_A.set_ylim(0, 2)
    ax_spikes_A.set_xlim(0, time[-1])
    ax_spikes_A.set_yticks([])
    ax_spikes_A.set_title(f"Neuron A: Spike Timing")
    ax_spikes_A.set_xlabel("Time")

    # B spikes
    ax_spikes_B = fig.add_subplot(gs[3, 1])
    ax_spikes_B.vlines(time[data["spkB"]], 0.5, 1.5, color=RED)
    ax_spikes_B.set_ylim(0, 2)
    ax_spikes_B.set_xlim(0, time[-1])
    ax_spikes_B.set_yticks([])
    ax_spikes_B.set_title(f"Neuron B: Spike Timing")
    ax_spikes_B.set_xlabel("Time")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig

def network_response_long(data, title=None):
    _, time = make_time(data["T"])
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 1.5, 1.5, 1.5])

    # input current
    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.plot(time, data["input_current"], color=BLACK)
    ax_input.set_xlim(0, time[-1])
    ax_input.set_title("External Input Current")
    ax_input.set_ylabel(r"$I_\mathrm{in}$")

    # inhibitory current
    ax_inh = fig.add_subplot(gs[1, 0])
    ax_inh.plot(time, data["gamma_inh"], color=BLUE)
    ax_inh.set_xlim(0, time[-1])
    ax_inh.set_title("Inhibitory Leak")
    ax_inh.set_ylabel(r"$\gamma_\mathrm{inh}$")

    # excitatory current
    ax_exc = fig.add_subplot(gs[1, 1])
    ax_exc.plot(time, data["I_exc"], color=RED)
    ax_exc.set_xlim(0, time[-1])
    ax_exc.set_title("Excitatory Current")
    ax_exc.set_ylabel(r"$I_\mathrm{exc}$")

    # neuron A membrane potential
    ax_mem_A = fig.add_subplot(gs[2, 0])
    ax_mem_A.plot(time, data["memA"], color=BLUE)
    ax_mem_A.axhline(1.0, ls='--', color=GRAY, alpha=0.5)
    ax_mem_A.set_xlim(0, time[-1])
    ax_mem_A.set_title("Neuron A: Membrane Potential")
    ax_mem_A.set_ylabel(r"$V / V_\mathrm{th}$")

    # neuron B membrane potential
    ax_mem_B = fig.add_subplot(gs[2, 1])
    ax_mem_B.plot(time, data["memB"], color=RED)
    ax_mem_B.axhline(1.0, ls='--', color=GRAY, alpha=0.5)
    ax_mem_B.set_xlim(0, time[-1])
    ax_mem_B.set_title("Neuron B: Membrane Potential")
    ax_mem_B.set_ylabel(r"$V / V_\mathrm{th}$")

    # A spikes
    ax_spikes_A = fig.add_subplot(gs[3, 0])
    ax_spikes_A.vlines(time[data["spkA"]], 0.5, 1.5, color=BLUE)
    ax_spikes_A.set_ylim(0, 2)
    ax_spikes_A.set_xlim(0, time[-1])
    ax_spikes_A.set_yticks([])
    ax_spikes_A.set_title("Spike Timing")
    ax_spikes_A.set_xlabel("Time")

    # B spikes
    ax_spikes_B = fig.add_subplot(gs[3, 1])
    ax_spikes_B.vlines(time[data["spkB"]], 0.5, 1.5, color=RED)
    ax_spikes_B.set_ylim(0, 2)
    ax_spikes_B.set_xlim(0, time[-1])
    ax_spikes_B.set_yticks([])
    ax_spikes_B.set_title("Spike Timing")

    # A mu
    ax_mu_A = fig.add_subplot(gs[4, 0])
    muA = np.array([0 if x is None else x for x in data["muA"]])
    ax_mu_A.plot(time, muA, color=BLUE)
    ax_mu_A.set_xlim(0, time[-1])
    ax_mu_A.set_title("Neuron A: Spike Amplitude")
    ax_mu_A.set_ylabel(r"$\mu$")
    ax_mu_A.set_xlabel("Time")

    # B mu
    ax_mu_B = fig.add_subplot(gs[4, 1])
    muB = np.array([0 if x is None else x for x in data["muB"]])
    ax_mu_B.plot(time, muB, color=RED)
    ax_mu_B.set_xlim(0, time[-1])
    ax_mu_B.set_title("Neuron B: Spike Amplitude")
    ax_mu_B.set_ylabel(r"$\mu$")
    ax_mu_B.set_xlabel("Time")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig

def _plot_A_response(data, fig, gs, title=None):
    _, time = make_time(data["T"])

    ax_input = fig.add_subplot(gs[0])
    ax_input.plot(time, data["input_current"], color=BLACK)
    ax_input.set_xlim(0, time[-1])
    ax_input.set_ylabel(r"$I_\mathrm{in}$")
    ax_input.set_title("External Input Current" if title is None else title)

    ax_mem = fig.add_subplot(gs[1], sharex=ax_input)
    ax_mem.plot(time, data["memA"], color=BLUE)
    ax_mem.axhline(1.0, ls="--", color=GRAY, alpha=0.5)
    ax_mem.set_ylabel(r"$V/V_\mathrm{th}$")

    ax_spikes = fig.add_subplot(gs[2], sharex=ax_input)
    ax_spikes.vlines(time[data["spkA"]], 0.5, 1.5, color=BLUE)
    ax_spikes.set_ylim(0, 2)
    ax_spikes.set_yticks([])
    ax_spikes.set_xlabel("Time")

def A_response(data, title=None):
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.5, 1.5])

    _plot_A_response(data, fig, gs, title)

    fig.tight_layout()
    return fig
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~ FIRING PATTERNS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def firing_patterns_panel(patterns, title=None):
    fig = plt.figure(figsize=(20, 16))
    outer = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.15)

    for i, (name, data) in enumerate(patterns.items()):
        r, c = divmod(i, 2)

        inner = outer[r, c].subgridspec(
            3, 1,
            height_ratios=[1.5, 1.5, 1],
            hspace=0.05,
        )

        _plot_A_response(data, fig, inner, name)

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~ SILENCE REGION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def last_spike_vs_w_inh(df, title = None):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(df["w_inh"], df["last_spk_time"], color=BLUE, s=30, alpha=0.8)

    ax.set_ylim(0, df["last_spk_time"].max() * 1.1)
    ax.set_xlabel(r"$w_\mathrm{inh}$")
    ax.set_ylabel("Last Spike Time of Neuron A")
    ax.set_title(title)

    ax.grid(True, alpha=0.5)

    fig.tight_layout()
    return fig

def firing_rate_vs_f(df, title = None):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(df["f"], df["fr"], color=BLUE, s=30, alpha=0.8)

    ax.set_ylim(0, df["fr"].max() * 1.1)
    ax.set_xlabel(r"$f$")
    ax.set_ylabel(r"$fr_A$")
    ax.set_title(title)

    ax.grid(True, alpha=0.5)

    fig.tight_layout()
    return fig

def last_spike_vs_f(df, title = None):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(df["f"], df["last_spk_time"], color=BLUE, s=30, alpha=0.8)

    ax.set_ylim(0, df["last_spk_time"].max() * 1.1)
    ax.set_xlabel(r"$f$")
    ax.set_ylabel("Last Spike Time of Neuron A")
    ax.set_title(title)

    ax.grid(True, alpha=0.5)

    fig.tight_layout()
    return fig

def silence_heatmap(df, index, columns, T, title=None, xlabel=None, ylabel=None):
    # reshape to a 2D matrix  
    piv = df.pivot(index=index, columns=columns, values="last_spk_time")
    col = piv.columns.values.astype(float)
    idx = piv.index.values.astype(float)
    last_spk_time = piv.values.astype(float)

    fig, ax = plt.subplots(figsize=(6, 5))
    # light blue = last spike near t=0
    # dark blue = last spike near t=T
    mesh = ax.pcolormesh(col, idx, last_spk_time, cmap=BLUE_CMAP, vmin=0, vmax=T, shading="nearest")
    fig.colorbar(mesh, ax=ax, label="Last spike time")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig

def silence_heatmap_side_by_side(df1, df2, index1, index2, columns1, columns2, T, xlabel1=None, xlabel2=None, ylabel1=None, ylabel2=None, title1=None, title2=None):
    # reshape to a 2D matrix    
    piv1 = df1.pivot(index=index1, columns=columns1, values="last_spk_time")
    col1 = piv1.columns.values.astype(float)
    idx1 = piv1.index.values.astype(float)
    last_spk_time1 = piv1.values.astype(float)

    piv2 = df2.pivot(index=index2, columns=columns2, values="last_spk_time")
    col2 = piv2.columns.values.astype(float)
    idx2 = piv2.index.values.astype(float)
    last_spk_time2 = piv2.values.astype(float)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    
    mesh = ax[0].pcolormesh(col1, idx1, last_spk_time1, cmap=BLUE_CMAP, vmin=0, vmax=T, shading="nearest")
    ax[0].set_xlabel(xlabel1)
    ax[0].set_ylabel(ylabel1)
    ax[0].set_title(title1)
    ax[0].set_box_aspect(1)

    ax[1].pcolormesh(col2, idx2, last_spk_time2, cmap=BLUE_CMAP, vmin=0, vmax=T, shading="nearest")
    ax[1].set_xlabel(xlabel2)
    ax[1].set_ylabel(ylabel2)
    ax[1].set_title(title2)
    ax[1].set_box_aspect(1)

    # shared colorbar
    fig.colorbar(mesh, ax=ax, label="Last spike time")

    return fig
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~ PHASE-LOCKING MAP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _pivot(df, metric):
    # reshape to a 2D matrix with:
    #   - drive amplitude on rows, 
    #   - drive frequency on columns,
    #   - `metric`` as values
    piv = df.pivot(index="I0", columns="f", values=metric)
    freqs = piv.columns.values.astype(float)
    I0s = piv.index.values.astype(float)
    Z = piv.values.astype(float)
    return freqs, I0s, Z

def plv_locking_map(df, metric, title=None, cbar_label=None, T_mask=None, f_pref_df=None):
    # pivot `column` and number of spikes of A
    freqs, I0s, Z = _pivot(df, metric)
    _, _, T = _pivot(df, "last_spk_time_A")

    cmap = BLUE_CMAP.copy()

    # mask points where A fired last spike before T_mask
    if T_mask is not None:
        Z = np.ma.masked_where(T < T_mask, Z)
        cmap.set_bad(color=GRAY, alpha=0.3) # rendered masked points in GRAY

    fig, ax = plt.subplots(figsize=(6, 5))
    # light blue = low Z
    # dark blue = high Z
    mesh = ax.pcolormesh(freqs, I0s, Z, cmap=cmap, vmin=0, vmax=1, shading="nearest")
    fig.colorbar(mesh, ax=ax, label=cbar_label)

    # overlay the preferred frequency curve f_pref(I0)
    if f_pref_df is not None:
        ax.plot(f_pref_df["f_preferred"], f_pref_df["I0"], color=RED)

    ax.set_xlabel(r"$f$")
    ax.set_ylabel(r"$I_0$")
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plv_locking_map_side_by_side(df1, df2, column1, column2, title1=None, title2=None, T_mask=None):
    # pivot `column` and number of spikes of A
    freqs, I0s, Z1 = _pivot(df1, column1)
    _, _, Z2 = _pivot(df2, column2)

    _, _, T1 = _pivot(df1, "last_spk_time_A")
    _, _, T2 = _pivot(df2, "last_spk_time_A")

    cmap = BLUE_CMAP.copy()

    # mask points where A fired last spike before T_mask
    if T_mask is not None:
        Z1 = np.ma.masked_where(T1 < T_mask, Z1)
        Z2 = np.ma.masked_where(T2 < T_mask, Z2)
        cmap.set_bad(color=GRAY, alpha=0.3) # rendered mased points in GRAY

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    
    mesh = ax[0].pcolormesh(freqs, I0s, Z1, cmap=cmap, vmin=0, vmax=1, shading="nearest")
    ax[0].set_xlabel(r"$f$")
    ax[0].set_ylabel(r"$I_0$")
    ax[0].set_title(title1)
    ax[0].set_box_aspect(1)

    ax[1].pcolormesh(freqs, I0s, Z2, cmap=cmap, vmin=0, vmax=1, shading="nearest")
    ax[1].set_xlabel(r"$f$")
    ax[1].set_ylabel(r"$I_0$")
    ax[1].set_title(title2)
    ax[1].set_box_aspect(1)

    # shared colorbar
    fig.colorbar(mesh, ax=ax, label="PLV")

    return fig

def plv_locking_map_4(df1, df2, df3, df4, metric="plv_11_A", title1=None, title2=None, title3=None, title4=None):
    # pivot `column` and number of spikes of A
    freqs, I0s, Z1 = _pivot(df1, metric)
    _, _, Z2 = _pivot(df2, metric)
    _, _, Z3 = _pivot(df3, metric)
    _, _, Z4 = _pivot(df4, metric)

    fig, ax = plt.subplots(2, 2, figsize=(12, 12), layout="constrained")

    mesh = ax[0][0].pcolormesh(freqs, I0s, Z1, cmap=BLUE_CMAP, vmin=0, vmax=1, shading="nearest")
    ax[0][0].set_xlabel(r"$f$")
    ax[0][0].set_ylabel(r"$I_0$")
    ax[0][0].set_title(title1)
    ax[0][0].set_box_aspect(1)

    ax[0][1].pcolormesh(freqs, I0s, Z2, cmap=BLUE_CMAP, vmin=0, vmax=1, shading="nearest")
    ax[0][1].set_xlabel(r"$f$")
    ax[0][1].set_ylabel(r"$I_0$")
    ax[0][1].set_title(title2)
    ax[0][1].set_box_aspect(1)

    ax[1][0].pcolormesh(freqs, I0s, Z3, cmap=BLUE_CMAP, vmin=0, vmax=1, shading="nearest")
    ax[1][0].set_xlabel(r"$f$")
    ax[1][0].set_ylabel(r"$I_0$")
    ax[1][0].set_title(title3)
    ax[1][0].set_box_aspect(1)

    ax[1][1].pcolormesh(freqs, I0s, Z4, cmap=BLUE_CMAP, vmin=0, vmax=1, shading="nearest")
    ax[1][1].set_xlabel(r"$f$")
    ax[1][1].set_ylabel(r"$I_0$")
    ax[1][1].set_title(title4)
    ax[1][1].set_box_aspect(1)

    # shared colorbar
    fig.colorbar(mesh, ax=ax, label="PLV")

    return fig

def f_preferred_curve(df, title=None):
    # drop rows where either coordinate is NaN
    df = df.dropna(subset=["I0", "f_preferred"]).sort_values("I0")
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(df["I0"], df["f_preferred"], color=BLUE)
    ax.set_xlabel(r"$I_0$")
    ax.set_ylabel(r"$f_{\mathrm{pref}}$")
    ax.set_title(title or r"$f_{\mathrm{pref}}(I_0)$")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig

def frA_I_curve(df_dc):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(df_dc["I0"], df_dc["fr_A"], color=BLUE)
    ax.set_xlabel(r"$I_0$")
    ax.set_ylabel(r"$f_A$")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig

def staircase(df_const, df_sin, I0_target):
    # free-running rate f0: neuron A's firing rate under constant drive I(t) = I0_target
    f0 = float(df_const.loc[np.isclose(df_const["I0"], I0_target), "fr_A"].iloc[0])
    print(f"Free-running rate at I0={I0_target:.2f}: f0 = {f0:.2f} Hz")

    # sinusoidal sweep at I0_target, ordered by drive frequency
    row = df_sin[np.isclose(df_sin["I0"], I0_target)].sort_values("f")
    f_drive = row["f"].to_numpy()   # drive frequency
    fr_sin = row["fr_A"].to_numpy() # A's firing rate


    firing_number  = fr_sin/f_drive # spikes per drive cycle (η=1 → 1:1 locking, η=2 → 2:1 locking, ...)
    xnorm = f_drive/f0              # normalized drive frequency

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xnorm, firing_number, "o-", color="#070090", ms=4)

    # reference curve η = f0/f
    # (the firing number a neuron would show if it kept firing at its free rate f0 regardless of the drive)
    xx = np.linspace(xnorm.min(), xnorm.max(), 200)
    ax.plot(xx, 1.0/xx, "--", color="0.5", lw=1, label=r"$\eta=f_0/f$ (free)")

    # guide lines for the plateaus
    for r in (3, 2, 1):
        ax.axhline(r, color="0.88", lw=0.8, zorder=0)

    ax.set_xlabel(r"$f_\mathrm{drive}/f_0$")
    ax.set_ylabel(r"firing number $\eta = f_{A}/f_\mathrm{drive}$")

    fig.tight_layout()
    return fig
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~ MU ANALYSIS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def mu_distribution(values, title=None):
    fig, axes = plt.subplots(figsize=(7, 3))

    axes.hist(values, bins=20, density=True, color=BLUE)
    axes.set_title(title)
    axes.set_xlabel(r"$\mu$")
    axes.set_ylabel("Density")
    axes.grid(alpha=0.3)

    fig.tight_layout()
    return fig

def mu_distribution_panel(data_N20, data_N40, data_N60, data_N80, title=None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)

    axes[0][0].hist(data_N20, bins=20, density=True, color=BLUE)
    axes[0][0].set_title(r"$N=20$")
    axes[0][0].set_xlabel(r"$\mu$")
    axes[0][0].set_ylabel("Density")
    axes[0][0].grid(alpha=0.3)

    axes[0][1].hist(data_N40, bins=20, density=True, color=BLUE)
    axes[0][1].set_title(r"$N=40$")
    axes[0][1].set_xlabel(r"$\mu$")
    axes[0][1].set_ylabel("Density")
    axes[0][1].grid(alpha=0.3)

    axes[1][0].hist(data_N60, bins=20, density=True, color=BLUE)
    axes[1][0].set_title(r"$N=60$")
    axes[1][0].set_xlabel(r"$\mu$")
    axes[1][0].set_ylabel("Density")
    axes[1][0].grid(alpha=0.3)

    axes[1][1].hist(data_N80, bins=20, density=True, color=BLUE)
    axes[1][1].set_title(r"$N=80$")
    axes[1][1].set_xlabel(r"$\mu$")
    axes[1][1].set_ylabel("Density")
    axes[1][1].grid(alpha=0.3)

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~