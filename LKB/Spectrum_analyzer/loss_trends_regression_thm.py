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


def analytical_amps(
    times, prep_fidelity, nth, delta_mem, kappa_mem, g, T_int, kappa_1, kappa_2, **_
):
    p = (prep_fidelity + 1) / 2
    if g * T_int > 0.05:
        print(
            f"Theta = {g * T_int:.2f} is large, the analytical formula might not be accurate"
        )
    amps = []
    for p in [p, 1 - p]:
        zeta_0 = 2 * p * T_int
        epsilon_0 = 2 * (1 - p) * T_int

        kappa_2_fact = (1 - np.exp(-kappa_2 * T_int)) / kappa_2
        comb_fact = T_int * np.exp(-kappa_1 * T_int)
        zeta_cond = -(1 - 2 * p) * comb_fact + kappa_2_fact
        epsilon_cond = (1 - 2 * p) * comb_fact + kappa_2_fact

        comb_fact = (np.exp(-kappa_2 * T_int) - np.exp(-kappa_1 * T_int)) / (
            kappa_1 - kappa_2
        )
        zeta_g = -(1 - 2 * p) * comb_fact + kappa_2_fact
        epsilon_g = (1 - 2 * p) * comb_fact + kappa_2_fact

        zeta = np.where(
            kappa_2 == 0, zeta_0, np.where(kappa_1 - kappa_2 == 0, zeta_cond, zeta_g)
        )
        epsilon = np.where(
            kappa_2 == 0,
            epsilon_0,
            np.where(kappa_1 - kappa_2 == 0, epsilon_cond, epsilon_g),
        )

        amp = (
            g**2
            / 2
            * (zeta**2 * nth + epsilon**2 * (nth + 1) - epsilon * zeta * (2 * nth + 1))
        )
        amp *= np.exp(-kappa_mem / 2 * times) * np.cos(delta_mem * times)
        amps.append(amp)
    return amps


def correlation_amps(
    N,
    delta_mem,
    kappa_mem,
    nth,
    kappa_1,
    kappa_2,
    g,
    prep_fidelity,
    T_int,
    T_idle,
    verbose=False,
):
    IN, I2 = qt.qeye(N), qt.qeye(2)
    a = qt.destroy(N)
    ac = qt.tensor(a, I2)
    sm = qt.tensor(IN, qt.sigmam())
    sx = qt.tensor(IN, qt.sigmax())
    sz = qt.tensor(IN, qt.sigmaz())

    H_idle = delta_mem * a.dag() * a
    H_JC = g * (ac * sm.dag() + ac.dag() * sm)
    H_comb = qt.tensor(H_idle, I2) + H_JC

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
            output = qt.mesolve(H_idle, rho_mem, t_evol_idle, c_ops)
            rho_mem = output.final_state
        rho = qt.tensor(rho_mem, rho_qb)
        output = qt.mesolve(H_comb, rho, t_evol, c_ops_comb)
        return output.final_state

    def evolve_state(N_step, k0):
        rho_mem = qt.thermal_dm(N, nth)
        rhos = []
        itera = range(k0, k0 + N_step)
        if verbose:
            itera = tqdm(itera)
        for k in itera:
            rho_qb = qb_g if k % 2 == 0 else qb_e
            rho = one_step_evolution(rho_mem, rho_qb)
            if k == k0:
                rho = sx * rho
            else:
                rhos.append(rho)
            rho_mem = rho.ptrace(0)
        return rhos

    rhos_g = evolve_state(3, 0)
    rhos_e = evolve_state(3, 1)
    amp_g = qt.expect(sx, rhos_g[-1])
    amp_e = qt.expect(sx, rhos_e[-1])

    if verbose:
        N_step = 100
        rhos_g = evolve_state(N_step, 0)
        rhos_e = evolve_state(N_step, 1)
        fig, ax = plt.subplots(figsize=(6, 5))
        times = np.arange(2, N_step, 2) * (T_int + T_idle)
        thrg, thre = analytical_amps(
            times, prep_fidelity, nth, delta_mem, kappa_mem, g, T_int, kappa_1, kappa_2
        )
        for rhos_el, thr in zip([rhos_g, rhos_e], [thrg, thre]):
            autocorr = qt.expect(sx, rhos_el[1::2])
            ax.plot(times / ms, autocorr.real)
            ax.plot(times / ms, thr, "k--")
        plt.show()

    return amp_g, amp_e


# %%
params = dict(
    N=50,
    delta_mem=2 * kHz,
    kappa_mem=200 * Hz,
    kappa_1=1 / (8 * us),
    kappa_2=1 / (4 * us),
    g=1 * kHz,
    T_int=4 * us,
    T_idle=13 * us,
    nth=1,
    prep_fidelity=0.75,
)
print(correlation_amps(**params, verbose=True))


# %%
######################## 1. Varying the preparation fidelity ########################
params = dict(
    N=100,
    delta_mem=2 * kHz,
    kappa_mem=200 * Hz,
    kappa_1=0,  # 1 / (12 * us),
    kappa_2=0,  # 1 / (6 * us),
    g=2 * kHz,
    T_int=1.5 * us,
    T_idle=13 * us,
)
prep_fidelities = np.linspace(0.05, 1, 11)
nths = np.linspace(0, 7, 5)
amps = np.zeros((len(nths), len(prep_fidelities), 2))
loc_params = params.copy()
for indn, nth in enumerate(tqdm(nths)):
    loc_params.update(nth=nth)
    for indp, prep_fidelity in enumerate(prep_fidelities):
        loc_params.update(prep_fidelity=prep_fidelity)
        amps[indn, indp] = correlation_amps(**loc_params)

# %%

colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(nths)))

prep_fidelitiess, nthss = np.meshgrid(prep_fidelities, nths)
theory_g, theory_e = analytical_amps(
    2 * (params["T_int"] + params["T_idle"]), prep_fidelitiess, nthss, **params
)

fig, ax = plt.subplots(3, figsize=(8, 8), sharex=True)
for ind_nth, nth in enumerate(nths):
    kwgs = {"fillstyle": "none", "color": colors[ind_nth]}
    ax[0].plot(prep_fidelities, amps[ind_nth, :, 0], "o", **kwgs)
    ax[0].plot(prep_fidelities, theory_g[ind_nth], "-", **kwgs)
    ax[1].plot(prep_fidelities, amps[ind_nth, :, 1], "o", **kwgs)
    ax[1].plot(prep_fidelities, theory_e[ind_nth], "-", **kwgs)
    t_est = amps[ind_nth, :, 0] / (amps[ind_nth, :, 1] - amps[ind_nth, :, 0])
    ax[2].plot(prep_fidelities, t_est, "o", **kwgs)
    thr_t = theory_g[ind_nth] / (theory_e[ind_nth] - theory_g[ind_nth])
    ax[2].plot(prep_fidelities, thr_t, "-", **kwgs)

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
    N=100,
    delta_mem=0 * kHz,
    kappa_mem=200 * Hz,
    prep_fidelity=1.0,
    kappa_1=0,  # 1 / (12 * us)
    g=1 * kHz,
    T_int=4 * us,
    T_idle=15 * us,
)
kappa_2s = np.linspace(0, 2 / params["T_idle"], 11)
nths = np.linspace(0, 7, 5)
amps = np.zeros((len(nths), len(kappa_2s), 2))
loc_params = params.copy()
for indn, nth in enumerate(tqdm(nths)):
    loc_params.update(nth=nth)
    for indk, kappa_2 in enumerate(kappa_2s):
        loc_params.update(kappa_2=kappa_2)
        amps[indn, indk] = correlation_amps(**loc_params)

# %%

colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(nths)))

kappa_2ss, nthss = np.meshgrid(kappa_2s, nths)
theory_g, theory_e = analytical_amps(
    2 * (params["T_int"] + params["T_idle"]), nth=nthss, kappa_2=kappa_2ss, **params
)

fig, ax = plt.subplots(3, figsize=(8, 8), sharex=True)
kappa_t = kappa_2s * params["T_idle"]
for ind_nth, nth in enumerate(nths):
    kwgs = {"fillstyle": "none", "color": colors[ind_nth]}
    ax[0].plot(kappa_t, amps[ind_nth, :, 0], "o", **kwgs)
    ax[0].plot(kappa_t, theory_g[ind_nth], **kwgs)
    ax[1].plot(kappa_t, amps[ind_nth, :, 1], "o", **kwgs)
    ax[1].plot(kappa_t, theory_e[ind_nth], **kwgs)
    t_est = amps[ind_nth, :, 0] / (amps[ind_nth, :, 1] - amps[ind_nth, :, 0])
    ax[2].plot(kappa_t, t_est, "o", **kwgs)
    thr_t = theory_g[ind_nth] / (theory_e[ind_nth] - theory_g[ind_nth])
    ax[2].plot(kappa_t, thr_t, "-", **kwgs)

ax[0].set_ylabel("$\\langle \\sigma_x(1) \\sigma_x(0) \\rangle_g$")
ax[1].set_ylabel("$\\langle \\sigma_x(1) \\sigma_x(0) \\rangle_e$")
ax[2].set_ylabel("Estimated temperature")
ax[2].set_xlabel("$\\kappa_\\phi T_{idle}$")

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
######################## 3. Varying qubit T1 ########################
params = dict(
    N=100,
    delta_mem=0 * kHz,
    kappa_mem=200 * Hz,
    prep_fidelity=1.0,
    g=0.5 * kHz,
    T_int=4 * us,
    T_idle=13 * us,
)
kappa_1s = np.linspace(0, 2 / params["T_idle"], 11)
kappa_2s = kappa_1s / 2
nths = np.linspace(0, 7, 5)
amps = np.zeros((len(nths), len(kappa_2s), 2))
loc_params = params.copy()
for indn, nth in enumerate(tqdm(nths)):
    loc_params.update(nth=nth)
    for indk, (kappa_1, kappa_2) in enumerate(zip(kappa_1s, kappa_2s)):
        loc_params.update(kappa_1=kappa_1, kappa_2=kappa_2)
        amps[indn, indk] = correlation_amps(**loc_params)


# %%
colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(nths)))

kappa_1ss, nthss = np.meshgrid(kappa_1s, nths)
kappa_2ss = kappa_1ss / 2
t_eval = 2 * (params["T_int"] + params["T_idle"])
theory_g, theory_e = analytical_amps(
    t_eval, nth=nthss, kappa_1=kappa_1ss, kappa_2=kappa_2ss, **params
)

kappa_t = kappa_2s * params["T_idle"]
fig, ax = plt.subplots(3, figsize=(8, 8), sharex=True)
for ind_nth, nth in enumerate(nths):
    kwgs = {"fillstyle": "none", "color": colors[ind_nth]}
    ax[0].plot(kappa_t, amps[ind_nth, :, 0], "o", **kwgs)
    ax[0].plot(kappa_t, theory_g[ind_nth], **kwgs)
    ax[1].plot(kappa_t, amps[ind_nth, :, 1], "o", **kwgs)
    ax[1].plot(kappa_t, theory_e[ind_nth], **kwgs)
    temp_est = amps[ind_nth, :, 0] / (amps[ind_nth, :, 1] - amps[ind_nth, :, 0])
    ax[2].plot(kappa_t, temp_est, "o", **kwgs)
    thr_t = theory_g[ind_nth] / (theory_e[ind_nth] - theory_g[ind_nth])
    ax[2].plot(kappa_t, thr_t, "-", **kwgs)

ax[0].set_ylabel("$\\langle \\sigma_x(1) \\sigma_x(0) \\rangle_g$")
ax[1].set_ylabel("$\\langle \\sigma_x(1) \\sigma_x(0) \\rangle_e$")
ax[2].set_ylabel("Estimated temperature")
ax[2].set_xlabel("$\\kappa_1 T_{idle}$")

for i in range(2):
    ax[i].plot([], [], "ko-", label="Simulation")
ax[2].plot([], [], "ko", label="Simulation")
ax[2].plot([], [], "k-", label="Theory")
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
sm = qt.sigmam()
sp = qt.sigmap()
I = qt.qeye(2)
qt.tensor(I, sm)
# %%
