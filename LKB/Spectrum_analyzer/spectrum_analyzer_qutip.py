# %%
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from tqdm import tqdm

kHz = 2 * np.pi
Hz = kHz / 1e3
us = 1e-3
ms = 1
s = 1e3

N = 20
kappa_mem = 200 * Hz
nth = 0.73
kappa_1 = 0  # 1 / (12 * us)
kappa_2 = 0  # 1 / (6 * us)
g = 1 * kHz
delta_mem = 0.2 * kHz
prep_fidelity = 0.8
T_int = 4 * us
T_idle = 16 * us

# %%
IN, I2 = qt.qeye(N), qt.qeye(2)
a = qt.destroy(N)
ac = qt.tensor(a, I2)
sm = qt.tensor(IN, qt.sigmam())
sx = qt.tensor(IN, qt.sigmax())
sy = qt.tensor(IN, qt.sigmay())
sz = qt.tensor(IN, qt.sigmaz())

H_det = delta_mem * a.dag() * a
H_JC = g * (ac * sm.dag() + ac.dag() * sm)
H = qt.tensor(H_det, I2) + H_JC

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
        output = qt.mesolve(H_det, rho_mem, t_evol_idle, c_ops)
        rho_mem = output.final_state
    rho = qt.tensor(rho_mem, rho_qb)
    output = qt.mesolve(H, rho, t_evol, c_ops_comb)
    return output.final_state


def evolve_state(N_step):
    rho_mem = qt.thermal_dm(N, nth)
    rhos = []
    for k in tqdm(range(N_step)):
        rho_qb = qb_g if k % 2 == 0 else qb_e
        rho = one_step_evolution(rho_mem, rho_qb)
        rho_mem = rho.ptrace(0)
        rhos.append(rho)
    return rhos


# %%
N_step = 2000
rhos = evolve_state(N_step)
# %%
fig, ax = plt.subplots(2, figsize=(6, 5))
nbars = qt.expect(ac.dag() * ac, rhos)
ax[0].plot(nbars.real)
szs = qt.expect(sz, rhos)
ax[1].plot(szs.real * (-1.0) ** np.arange(N_step))

# %%
a = qt.tensor(qt.destroy(N), I2)
ac = qt.tensor(a, I2)
sm = qt.tensor(IN, I2, qt.sigmam())
sx = qt.tensor(IN, I2, qt.sigmax())
sy = qt.tensor(IN, I2, qt.sigmay())
sz = qt.tensor(IN, I2, qt.sigmaz())

H_det = delta_mem * a.dag() * a
H_JC = g * (ac * sm.dag() + ac.dag() * sm)
H = qt.tensor(H_det, I2) + H_JC

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
        output = qt.mesolve(H_det, rho_mem, t_evol_idle, c_ops)
        rho_mem = output.final_state
    rho = qt.tensor(rho_mem, rho_qb)
    output = qt.mesolve(H, rho, t_evol, c_ops_comb)
    return output.final_state


def evolve_state2(rho_mem, N_step, k0):
    rhos = []
    for k in tqdm(range(k0, N_step + k0)):
        rho_qb = qb_g if k % 2 == 0 else qb_e
        rho = one_step_evolution2(rho_mem, rho_qb)
        rho_mem = rho.ptrace((0, 1))
        rhos.append(rho)
    return rhos


# %%
N_step = 500
rhos_g = evolve_state2(rhos[-2], N_step, 1)
rhos_e = evolve_state2(rhos[-1], N_step, 0)
# %%
fig, ax = plt.subplots(figsize=(6, 5))
times = np.arange(2, N_step + 2, 2) * (T_int + T_idle)
theta = g * T_int
for ind, rhos_el in enumerate([rhos_g, rhos_e]):
    autocorr = qt.expect(qt.tensor(IN, qt.sigmax(), qt.sigmax()), rhos_el[1::2])
    ax.plot(times / ms, autocorr.real)
    pp = 1 - p if ind else p
    amp = 2 * pp * theta**2 * nth * (2 * pp - 1)
    amp += 2 * (1 - pp) * theta**2 * (nth + 1) * (1 - 2 * pp)
    thr_curve = amp * np.exp(-kappa_mem / 2 * times) * np.cos(delta_mem * times)
    ax.plot(times / ms, thr_curve, "k--")
# %%
