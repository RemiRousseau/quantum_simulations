# %%

import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.constants as csts
from tqdm import tqdm

# %%
MHz = 2 * jnp.pi
kHz = MHz / 1e3
Hz = kHz / 1e3
us = 1
ms = 1e3

# %%
N = 15
kappa_mem = 100 * Hz
nth = 2
g = 1 * kHz
eps = 0.0 * kHz
delta_drive = 5.0 * kHz
delta_mem = 2 * kHz
T_int = 4 * us
T_idle = 13 * us


# %%
def nth_eval(kappa_1, kappa_2, prep_fidelity):
    a = dq.destroy(N)
    IN, I2 = dq.eye(N), dq.eye(2)
    sm = dq.sigmam()
    sx = dq.sigmax()
    sy = dq.sigmay()
    sz = dq.sigmaz()

    def H_drive(smf):
        if eps == 0:
            return 0 * smf
        H = dq.modulated(lambda t: eps * jnp.exp(1j * delta_drive * t), smf)
        H += dq.modulated(lambda t: eps * jnp.exp(-1j * delta_drive * t), dq.dag(smf))
        return H

    H_det = delta_mem * dq.dag(a) @ a
    H_JC_2 = g * (dq.tensor(a, dq.dag(sm)) + dq.tensor(dq.dag(a), sm))

    Lmd = jnp.sqrt(kappa_mem * (nth + 1)) * a
    Lmu = jnp.sqrt(kappa_mem * nth) * dq.dag(a)
    L1d = jnp.sqrt(kappa_1 / 2) * sm
    L1u = jnp.sqrt(kappa_1 / 2) * dq.dag(sm)
    kappa_phi = kappa_2 - kappa_1 / 2
    assert kappa_phi >= 0
    Lphi = jnp.sqrt(kappa_phi / 2) * sz

    H_comb = dq.tensor(H_det, I2) + H_JC_2 + H_drive(dq.tensor(IN, sm))
    c_ops_comb = [dq.tensor(L, I2) for L in [Lmd, Lmu]]
    c_ops_comb += [dq.tensor(dq.eye(N), L) for L in [L1d, L1u, Lphi]]

    H_free = H_det
    c_ops_free = [Lmd, Lmu]

    p = (prep_fidelity + 1) / 2
    qb_g = dq.fock_dm(2, 1) * p + dq.fock_dm(2, 0) * (1 - p)
    qb_e = dq.fock_dm(2, 0) * p + dq.fock_dm(2, 1) * (1 - p)

    options_solve = dq.Options(save_states=False, progress_meter=None)

    # @jax.jit
    def one_step_evolution(t, rho_mem, rho_qb, Nt=2) -> tuple[jax.Array, jax.Array]:
        if T_idle != 0:
            t_evol = jnp.linspace(t, t + T_idle, Nt)
            output0 = dq.mesolve(
                H_free, c_ops_free, rho_mem, t_evol, options=options_solve
            )
            rho_mem = output0.final_state
        t_evol = jnp.linspace(t + T_idle, t + T_idle + T_int, Nt)
        rho = dq.tensor(rho_mem, rho_qb)
        output1 = dq.mesolve(H_comb, c_ops_comb, rho, t_evol, options=options_solve)
        return output1.final_state

    # @partial(jax.jit, static_argnums=(0,))
    def evolve_state(Nstep, rho_mem):
        def rho_qb_func(k):
            return jnp.where(k % 2 == 0, qb_g, qb_e)

        def scan_body(rho_mem, k):
            rho = one_step_evolution(k * (T_int + T_idle), rho_mem, rho_qb_func(k))
            rho_mem = dq.ptrace(rho, 0, (N, 2))
            return rho_mem, rho

        ks = jax.numpy.arange(Nstep)
        _, rhos = jax.lax.scan(scan_body, rho_mem, ks)

        times = (1 + ks) * (T_int + T_idle)
        return times, rhos

    Nstep = 3_000
    ns = jnp.arange(N)
    rho_mem = dq.unit(jnp.diag((1 + nth) ** (-1.0) * (nth / (1 + nth)) ** ns)).astype(
        complex
    )
    times, rhos = evolve_state(Nstep, rho_mem)

    H_JC_3 = g * (dq.tensor(a, I2, dq.dag(sm)) + dq.tensor(dq.dag(a), I2, sm))
    H_comb_3 = dq.tensor(H_det, I2, I2) + H_JC_3 + H_drive(dq.tensor(IN, I2, sm))
    c_ops_comb_3 = [dq.tensor(L, I2, I2) for L in [Lmd, Lmu]]
    c_ops_comb_3 += [dq.tensor(IN, I2, L) for L in [L1d, L1u, Lphi]]

    H_free_2 = dq.tensor(H_det, I2)
    c_ops_free_2 = [dq.tensor(l, I2) for l in [Lmd, Lmu]]

    options_solve = dq.Options(save_states=False, progress_meter=None)

    # @jax.jit
    def one_step_evolution3(t, rho_mem, rho_qb, Nt=2) -> tuple[jax.Array, jax.Array]:
        if T_idle != 0:
            t_evol = jnp.linspace(t, t + T_idle, Nt)
            output0 = dq.mesolve(
                H_free_2, c_ops_free_2, rho_mem, t_evol, options=options_solve
            )
            rho_mem = output0.final_state
        t_evol = jnp.linspace(t + T_idle, t + T_int + T_idle, Nt)
        rho = dq.tensor(rho_mem, rho_qb)
        output1 = dq.mesolve(H_comb_3, c_ops_comb_3, rho, t_evol, options=options_solve)
        return output1.final_state

    # @partial(jax.jit, static_argnums=(1,))
    def evolve_state3(k0, Nstep, rho_mem):
        def rho_qb_func3(k):
            return jnp.where(k % 2 == 0, qb_g, qb_e)

        def scan_body(rho_mem, k):
            rho = one_step_evolution3(k * (T_int + T_idle), rho_mem, rho_qb_func3(k))
            rho_mem = dq.ptrace(rho, (0, 1), (N, 2, 2))
            return rho_mem, rho

        ks = jax.numpy.arange(Nstep) + k0
        _, rhos = jax.lax.scan(scan_body, rho_mem, ks)

        times = (1 + ks) * (T_int + T_idle)
        return times, rhos

    T_start = jnp.inf
    Nstep = 2000

    ind_start = len(times) - 1
    if ind_start % 2 == 1:
        ind_start -= 1
    T_start = times[ind_start]

    times_auto_g, rhos_g_auto = evolve_state3(ind_start, Nstep, rhos[ind_start])
    times_auto_e, rhos_e_auto = evolve_state3(ind_start + 1, Nstep, rhos[ind_start + 1])

    tg = times_auto_g[::2] - T_start + 2 * (T_int + T_idle)
    auto_corr_g = dq.expect(dq.tensor(dq.eye(N), sy, sy), rhos_g_auto[::2]).real
    delta_t = tg[1] - tg[0]
    ft = jnp.fft.rfft(auto_corr_g, n=len(auto_corr_g) * 10) * delta_t
    spectrum_g = jnp.abs(ft) ** 2
    freqs = jnp.fft.rfftfreq(len(auto_corr_g) * 10, delta_t)

    te = times_auto_e[1::2] - T_start
    auto_corr_e = dq.expect(dq.tensor(dq.eye(N), sy, sy), rhos_e_auto[::2]).real
    ft = jnp.fft.rfft(auto_corr_e, n=len(auto_corr_e) * 10) * delta_t
    spectrum_e = jnp.abs(ft)
    freqs = jnp.fft.rfftfreq(len(auto_corr_e) * 10, delta_t)

    ind_lorentz_center = jnp.argmin(jnp.abs(freqs / kHz * 2 * jnp.pi - delta_mem / kHz))
    GH, EH = spectrum_g[ind_lorentz_center], spectrum_e[ind_lorentz_center]
    if tg[0] == 0:
        GH -= 1
        EH -= 1
    nth_eval = GH / (EH - GH)
    temp_eval = csts.h * 5e6 / csts.k / jnp.log(1 + 1 / nth_eval)
    return GH, EH, nth_eval, temp_eval


# %%
prep_fidelities = jnp.linspace(0.1, 1, 5)
GHs, EHs, nths, temps = [], [], [], []
for p_fid in tqdm(prep_fidelities):
    gh, eh, nth_ev, temp = nth_eval(0, 0, p_fid)
    GHs.append(gh)
    EHs.append(eh)
    nths.append(nth_ev)
    temps.append(temp)
GHs, EHs = jnp.array(GHs), jnp.array(EHs)
# %%

ps = (prep_fidelities + 1) / 2
plt.plot(prep_fidelities, nth * (2 * ps - 1) - (1 - ps))
plt.plot(prep_fidelities[1:], nths[1:], "o")
nths_bis = jnp.sqrt(GHs) / (jnp.sqrt(EHs) - jnp.sqrt(GHs))
plt.plot(prep_fidelities[1:], nths_bis[1:], "o")

# %%
