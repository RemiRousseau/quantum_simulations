# %%
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from tqdm import tqdm

kHz = 2 * np.pi
Hz = kHz / 1e3
us = 1e-3
ms = 1
s = 1e3


# %%
def correlation_amps(
    N,
    kappa_mem,
    nth,
    kappa_1,
    kappa_2,
    g,
    prep_fidelity,
    T_int,
    T_idle,
    threshold_init=1e-6,
    verbose=False,
):
    IN, I2 = qt.qeye(N), qt.qeye(2)
    a = qt.destroy(N)
    ac = qt.tensor(a, I2)
    sm = qt.tensor(IN, qt.sigmam())
    sz = qt.tensor(IN, qt.sigmaz())

    H_JC = g * (ac * sm.dag() + ac.dag() * sm)

    Lmd = np.sqrt(kappa_mem * (nth + 1)) * a
    Lmu = np.sqrt(kappa_mem * nth) * a.dag()
    Ld = np.sqrt(kappa_1 / 2) * sm
    Lu = np.sqrt(kappa_1 / 2) * sm.dag()
    kappa_phi = kappa_2 - kappa_1 / 2
    assert kappa_phi >= 0
    Lphi = np.sqrt(kappa_phi / 2) * sz
    c_ops = [Lmd, Lmu]
    c_ops_comb = [Ld, Lu, Lphi]
    c_ops_comb += [qt.tensor(L, I2) for L in c_ops]

    p = (prep_fidelity + 1) / 2
    qb_g = qt.fock_dm(2, 1) * p + qt.fock_dm(2, 0) * (1 - p)
    qb_e = qt.fock_dm(2, 0) * p + qt.fock_dm(2, 1) * (1 - p)

    t_evol = np.linspace(0, T_int, 2)
    t_evol_idle = np.linspace(0, T_idle, 2)

    def one_step_evolution(rho_mem, rho_qb):
        if T_idle > 0:
            output = qt.mesolve(0 * a, rho_mem, t_evol_idle, c_ops)
            rho_mem = output.final_state
        rho = qt.tensor(rho_mem, rho_qb)
        output = qt.mesolve(H_JC, rho, t_evol, c_ops_comb)
        return output.final_state

    def evolve_state(threshold):
        rho_mem = qt.thermal_dm(N, nth)
        nbars = []
        for k in range(4):
            rho_qb = qb_g if k % 2 == 0 else qb_e
            rho = one_step_evolution(rho_mem, rho_qb)
            rho_mem = rho.ptrace(0)
            nbars.append(qt.expect(ac.dag() * ac, rho))
        mean_nb_prev = np.mean(nbars[:-2])
        mean_nb = np.mean(nbars[-2:])
        k = 0
        while np.abs(mean_nb - mean_nb_prev) / mean_nb > threshold:
            rhos = []
            for k in range(2):
                rho_qb = qb_g if k % 2 == 0 else qb_e
                rho = one_step_evolution(rho_mem, rho_qb)
                rhos.append(rho)
                rho_mem = rho.ptrace(0)
                nbars.append(qt.expect(ac.dag() * ac, rho))
            mean_nb_prev = mean_nb
            mean_nb = np.mean(nbars[-2:])
            k += 1
        return nbars, rhos

    nbars, rhos_init = evolve_state(threshold_init)
    if verbose:
        _, ax = plt.subplots()
        ax.plot(nbars)
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean number of photons")
        plt.show()

    a = qt.tensor(qt.destroy(N), I2)
    ac = qt.tensor(a, I2)
    sm = qt.tensor(IN, I2, qt.sigmam())
    sx = qt.tensor(IN, I2, qt.sigmax())
    sz = qt.tensor(IN, I2, qt.sigmaz())

    H_JC = g * (ac * sm.dag() + ac.dag() * sm)

    Lmd = np.sqrt(kappa_mem * (nth + 1)) * a
    Lmu = np.sqrt(kappa_mem * nth) * a.dag()
    Ld = np.sqrt(kappa_1 / 2) * sm
    Lu = np.sqrt(kappa_1 / 2) * sm.dag()
    kappa_phi = kappa_2 - kappa_1 / 2
    assert kappa_phi >= 0
    Lphi = np.sqrt(kappa_phi / 2) * sz
    c_ops = [Lmd, Lmu]
    c_ops_comb = [Ld, Lu, Lphi]
    c_ops_comb += [qt.tensor(L, I2) for L in c_ops]

    def one_step_evolution2(rho_mem, rho_qb):
        if T_idle > 0:
            output = qt.mesolve(0 * a, rho_mem, t_evol_idle, c_ops)
            rho_mem = output.final_state
        rho = qt.tensor(rho_mem, rho_qb)
        output = qt.mesolve(H_JC, rho, t_evol, c_ops_comb)
        return output.final_state

    def evolve_state2(rho_mem, N_step, k0):
        rhos = []
        itera = range(k0, N_step + k0)
        if verbose:
            itera = tqdm(itera)
        for k in itera:
            rho_qb = qb_g if k % 2 == 0 else qb_e
            rho = one_step_evolution2(rho_mem, rho_qb)
            rho_mem = rho.ptrace((0, 1))
            rhos.append(rho)
        return rhos

    N_step = 2
    rhos_g = evolve_state2(rhos_init[0], N_step, 1)
    rhos_e = evolve_state2(rhos_init[1], N_step, 0)
    amp_g = qt.expect(qt.tensor(IN, qt.sigmax(), qt.sigmax()), rhos_g[1])
    amp_e = qt.expect(qt.tensor(IN, qt.sigmax(), qt.sigmax()), rhos_e[1])

    if verbose:
        N_step = 500
        rhos_g = evolve_state2(rhos_init[0], N_step, 1)
        rhos_e = evolve_state2(rhos_init[1], N_step, 0)
        fig, ax = plt.subplots(figsize=(6, 5))
        times = np.arange(2, N_step + 2, 2) * (T_int + T_idle)
        theta = g * T_int
        for ind, rhos_el in enumerate([rhos_g, rhos_e]):
            autocorr = qt.expect(qt.tensor(IN, qt.sigmax(), qt.sigmax()), rhos_el[1::2])
            ax.plot(times / ms, autocorr.real)
            pp = 1 - p if ind else p
            amp = 2 * pp * theta**2 * nth * (2 * pp - 1)
            amp += 2 * (1 - pp) * theta**2 * (nth + 1) * (1 - 2 * pp)
            thr_curve = amp * np.exp(-kappa_mem / 2 * times)
            ax.plot(times / ms, thr_curve, "k--")
        plt.show()

    return np.mean(nbars[-2:]), (amp_g, amp_e)


# %%
params = dict(
    N=30,
    kappa_mem=200 * Hz,
    kappa_1=0,  # 1 / (12 * us),
    kappa_2=0,  # 1 / (6 * us),
    g=1 * kHz,
    T_int=4 * us,
    T_idle=13 * us,
    nth=2,
    prep_fidelity=0.9,
)
correlation_amps(**params, verbose=True)


# %%
######################## 1. Varying the preparation fidelity ########################
params = dict(
    N=20,
    kappa_mem=200 * Hz,
    kappa_1=0,  # 1 / (12 * us)
    kappa_2=0,  # 1 / (6 * us)
    g=1 * kHz,
    T_int=4 * us,
    T_idle=13 * us,
)
prep_fidelities = np.linspace(0.05, 1, 11)
nths = np.linspace(0, 3, 5)
amps = np.zeros((len(nths), len(prep_fidelities), 2))
nbars = np.zeros((len(nths), len(prep_fidelities)))
for indn, nth in enumerate(tqdm(nths)):
    params.update(nth=nth)
    for indp, prep_fidelity in enumerate(tqdm(prep_fidelities)):
        params.update(prep_fidelity=prep_fidelity)
        nbars[indn, indp], amps[indn, indp] = correlation_amps(**params)

# %%
fig, ax = plt.subplots(3, figsize=(8, 8), sharex=True)

colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(nths)))

theta = params["g"] * params["T_int"]
prep_fidelitiess, nthss = np.meshgrid(prep_fidelities, nths)
p = (1 + prep_fidelitiess) / 2
theory_g = 2 * p * theta**2 * nthss * (2 * p - 1)
theory_g += 2 * (1 - p) * theta**2 * (nthss + 1) * (1 - 2 * p)
theory_g *= np.exp(-params["kappa_mem"] / 2 * 2 * (params["T_int"] + params["T_idle"]))
p = 1 - (1 + prep_fidelitiess) / 2
theory_e = 2 * p * theta**2 * nthss * (2 * p - 1)
theory_e += 2 * (1 - p) * theta**2 * (nthss + 1) * (1 - 2 * p)
theory_e *= np.exp(-params["kappa_mem"] / 2 * 2 * (params["T_int"] + params["T_idle"]))

for ind_nth, nth in enumerate(nths):
    ax[0].plot(prep_fidelities, amps[ind_nth, :, 0], "o", color=colors[ind_nth])
    ax[0].plot(prep_fidelities, theory_g[ind_nth], "-", color=colors[ind_nth])
    ax[1].plot(prep_fidelities, amps[ind_nth, :, 1], "o", color=colors[ind_nth])
    ax[1].plot(prep_fidelities, theory_e[ind_nth], "-", color=colors[ind_nth])
    ax[2].plot(
        prep_fidelities,
        amps[ind_nth, :, 0] / (amps[ind_nth, :, 1] - amps[ind_nth, :, 0]),
        "o",
        color=colors[ind_nth],
    )
    ax[2].plot(
        prep_fidelities,
        theory_g[ind_nth] / (theory_e[ind_nth] - theory_g[ind_nth]),
        "-",
        color=colors[ind_nth],
    )

ax[0].set_ylabel("$\\langle \\sigma_x(1) \\sigma_x(0) \\rangle_g$")
ax[1].set_ylabel("$\\langle \\sigma_x(1) \\sigma_x(0) \\rangle_e$")
ax[2].set_ylabel("Estimated temperature")
ax[2].set_xlabel("Preparation fidelity")

for i in range(3):
    ax[i].grid()
    ax[i].plot([], [], "k-", label="Theory")
    ax[i].plot([], [], "ko", label="Simulation")
    ax[i].legend()
fig.colorbar(
    plt.cm.ScalarMappable(norm=mcolors.Normalize(nths[0], nths[-1]), cmap="viridis"),
    ax=ax,
    label="$n_{th}$",
)
plt.show()

# %%
######################## 2. Varying qubit T2 ########################
params = dict(
    N=20,
    kappa_mem=200 * Hz,
    prep_fidelity=1.0,
    kappa_1=0,  # 1 / (12 * us)
    g=2 * kHz,
    T_int=4 * us,
    T_idle=13 * us,
)
kappa_2s = np.linspace(0, 1.5 / params["T_idle"], 11)
nths = np.linspace(0, 3, 5)
amps = np.zeros((len(nths), len(kappa_2s), 2))
nbars = np.zeros((len(nths), len(kappa_2s)))
for indn, nth in enumerate(tqdm(nths)):
    params.update(nth=nth)
    for indk, kappa_2 in enumerate(tqdm(kappa_2s)):
        params.update(kappa_2=kappa_2)
        nbars[indn, indk], amps[indn, indk] = correlation_amps(**params)

# %%
fig, ax = plt.subplots(3, figsize=(8, 8), sharex=True)

colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(nths)))

for ind_nth, nth in enumerate(nths):
    ax[0].plot(
        kappa_2s * params["T_idle"], amps[ind_nth, :, 0], "--o", color=colors[ind_nth]
    )
    ax[1].plot(
        kappa_2s * params["T_idle"], amps[ind_nth, :, 1], "--o", color=colors[ind_nth]
    )
    ax[2].plot(
        kappa_2s * params["T_idle"],
        amps[ind_nth, :, 0] / (amps[ind_nth, :, 1] - amps[ind_nth, :, 0]),
        "--o",
        color=colors[ind_nth],
    )

ax[0].set_ylabel("$\\langle \\sigma_x(1) \\sigma_x(0) \\rangle_g$")
ax[1].set_ylabel("$\\langle \\sigma_x(1) \\sigma_x(0) \\rangle_e$")
ax[2].set_ylabel("Estimated temperature")
ax[2].set_xlabel("$\\kappa_\\phi T_{idle}$")

for i in range(3):
    ax[i].grid()
    ax[i].plot([], [], "ko--", label="Simulation")
    ax[i].legend()
fig.colorbar(
    plt.cm.ScalarMappable(norm=mcolors.Normalize(nths[0], nths[-1]), cmap="viridis"),
    ax=ax,
    label="$n_{th}$",
)
plt.show()

# %%
######################## 3. Varying qubit T1 ########################
params = dict(
    N=200,
    kappa_mem=200 * Hz,
    prep_fidelity=1.0,
    g=1 * kHz,
    T_int=4 * us,
    T_idle=13 * us,
)
kappa_1s = np.linspace(0, 3 / params["T_idle"], 6)
kappa_2s = kappa_1s / 2
nths = np.linspace(5, 20, 5)
amps = np.zeros((len(nths), len(kappa_2s), 2))
nbars = np.zeros((len(nths), len(kappa_2s)))
for indn, nth in enumerate(tqdm(nths)):
    params.update(nth=nth)
    for indk, (kappa_1, kappa_2) in enumerate(zip(kappa_1s, tqdm(kappa_2s))):
        params.update(kappa_1=kappa_1, kappa_2=kappa_2)
        nbars[indn, indk], amps[indn, indk] = correlation_amps(**params)

# %%
fig, ax = plt.subplots(3, figsize=(8, 8), sharex=True)

colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(nths)))

fits = np.zeros((len(nths), 2))
for ind_nth, nth in enumerate(nths):
    ax[0].plot(
        kappa_1s * params["T_idle"], amps[ind_nth, :, 0], "-o", color=colors[ind_nth]
    )
    ax[1].plot(
        kappa_1s * params["T_idle"], amps[ind_nth, :, 1], "-o", color=colors[ind_nth]
    )
    temp_est = amps[ind_nth, :, 0] / (amps[ind_nth, :, 1] - amps[ind_nth, :, 0])
    ax[2].plot(kappa_1s * params["T_idle"], temp_est, "o", color=colors[ind_nth])
    fits[ind_nth] = np.polyfit(kappa_1s, temp_est, 1)
    ax[2].plot(
        kappa_1s * params["T_idle"],
        np.polyval(fits[ind_nth], kappa_1s),
        "--",
        color=colors[ind_nth],
    )

ax[0].set_ylabel("$\\langle \\sigma_x(1) \\sigma_x(0) \\rangle_g$")
ax[1].set_ylabel("$\\langle \\sigma_x(1) \\sigma_x(0) \\rangle_e$")
ax[2].set_ylabel("Estimated temperature")
ax[2].set_xlabel("$\\kappa_1 T_{idle}$")

for i in range(2):
    ax[i].plot([], [], "ko-", label="Simulation")
ax[2].plot([], [], "ko", label="Simulation")
ax[2].plot([], [], "k--", label="Linear fit")
for i in range(3):
    ax[i].grid()
    ax[i].legend()

fig.colorbar(
    plt.cm.ScalarMappable(norm=mcolors.Normalize(nths[0], nths[-1]), cmap="viridis"),
    ax=ax,
    label="$n_{th}$",
)
plt.show()
# %%
for ind_nth, nth in enumerate(nths):
    if ind_nth == 0:
        continue
    temp_est = amps[ind_nth, :, 0] / (amps[ind_nth, :, 1] - amps[ind_nth, :, 0])
    plt.plot(
        kappa_1s * params["T_idle"],
        temp_est / nth,
        color=colors[ind_nth],
        label=f"$n_{{th}}={nth}$",
    )
plt.legend()
plt.xlabel("$\\kappa_1 T_{idle}$")
plt.ylabel("$\hat{n}_{th}/n_{th}$")
plt.grid()
# %%
fig, ax = plt.subplots(2)
ax[0].plot(nths, fits[:, 1], "o")
ax[0].plot(nths, nths, "k--", label="Identity")
ax[0].grid()
ax[0].set_xlabel("$n_{th}$")
ax[0].set_ylabel("Offset of the fit")
ax[0].legend()

ax[1].plot(nths, fits[:, 0], "o")
fitfit = np.polyfit(nths, fits[:, 0], 1)
ax[1].plot(nths, np.polyval(fitfit, nths), "k--", label="Fit of the slopes")
ax[1].grid()
ax[1].set_xlabel("$n_{th}$")
ax[1].set_ylabel("Slope of the fit")
fig.tight_layout()
plt.show()
# %%
