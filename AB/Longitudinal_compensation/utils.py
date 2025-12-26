import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.constants as csts

nH = 1e-9
m = 1
GHz = 2 * jnp.pi * 1e3
MHz = 2 * jnp.pi
kHz = 2 * jnp.pi * 1e-3
us = 1
s = 1e6
phi_0 = csts.hbar / 2 / csts.e


def lj_to_ej(lj):
    return phi_0**2 / lj / csts.h * 1e-6


def hamiltonian_terms(
    phase, vatt, amp, ej, el, phi_a, phi_b, wa, wb, wro, kappa_a, kappa_b, phi_0, **_
):
    dc_l, dc_r = jnp.exp(1j * (phase + phi_0)), 10 ** (-vatt) * jnp.exp(1j * phi_0)
    phi_sigma = (dc_r - dc_l) / 2 * amp
    phi_delta = (dc_r + dc_l) / 2 * amp

    g1 = -(2 * ej * phi_sigma + el * phi_delta)
    gl = 2 * ej * phi_sigma * phi_a**2 * phi_b
    g3 = 2 * ej * phi_sigma * phi_b**3

    phase_phi_sigma = jnp.angle(phi_sigma)
    phi_sigma *= jnp.exp(-1j * phase_phi_sigma)
    phi_delta *= jnp.exp(-1j * phase_phi_sigma)

    g1_ss = -(2 * ej * phi_sigma + el * phi_delta)
    xi_a = -1j * g1_ss * phi_a / (1j * (wa - wro) + kappa_a / 2)
    xi_b = -1j * g1_ss * phi_b / (1j * (wb - wro) + kappa_b / 2)

    fact = 2 * ej * phi_sigma * (phi_a * xi_a.real + phi_b * xi_b.real)
    delta_a = fact * phi_a**2
    delta_b = fact * phi_b**2

    return g1, gl, g3, delta_a.real, delta_b.real


def simulate_memory(
    g1,
    gl,
    g3,
    delta_a,
    delta_b,
    Na_2,
    Nb_2,
    kappa_a,
    kappa_b,
    kappa_2,
    tstab,
    nsave,
    **_,
):
    a, b = dq.destroy(Na_2, Nb_2)
    g2 = jnp.sqrt(kappa_b * kappa_2) / 2

    H = (
        g1 * dq.dag(b)
        + gl * dq.dag(a) @ a @ dq.dag(b)
        + g2 * a @ a @ dq.dag(b)
        + g3 * dq.dag(b) @ b @ dq.dag(b)
    )
    H += dq.dag(H)
    H += delta_a * dq.dag(a) @ a + delta_b * dq.dag(b) @ b

    c_ops = [jnp.sqrt(kappa_a) * a, jnp.sqrt(kappa_b) * b]
    tmem = jnp.linspace(0, tstab, nsave)
    rho_0 = dq.tensor(dq.fock(Na_2, 0), dq.fock(Nb_2, 0))
    e_ops = [dq.dag(a) @ a, b]
    output = dq.mesolve(H, c_ops, rho_0, tmem, exp_ops=e_ops)
    return output.expects[:, -1]


def simulate_readout(g1, gl, g3, delta_a, delta_b, na, Nb_1, kappa_b, tro, nsave, **_):
    b = dq.destroy(Nb_1)
    H = (g1 + na * gl) * dq.dag(b) + g3 * dq.dag(b) @ b @ dq.dag(b)
    H += dq.dag(H)
    H += delta_b * dq.dag(b) @ b

    c_ops = [jnp.sqrt(kappa_b) * b]
    tmem = jnp.linspace(0, tro, nsave)
    rho_0 = dq.fock(Nb_1, 0)
    e_ops = [b]
    output = dq.mesolve(H, c_ops, rho_0, tmem, exp_ops=e_ops)
    return output.expects[0, -1]


def simulate_sequence(phase, vatt, amp_map, amp_ro, params):
    g1, gl, g3, delta_a, delta_b = hamiltonian_terms(phase, vatt, amp_map, **params)
    # g2 = jnp.sqrt(params["kappa_b"] * params["kappa_2"]) / 2
    # print(f'Maximum g1/g2 = {jnp.max(jnp.abs(res[0]))/g2:.2f}')
    # print(f'Maximum gl/g2 = {jnp.max(jnp.abs(res[1]))/g2:.2f}')
    # print(f'Maximum delta_a = {jnp.max(jnp.abs(res[3]))/MHz:.2f} MHz')
    # print(f'Maximum delta_b = {jnp.max(jnp.abs(res[4]))/MHz:.2f} MHz')
    nbar, b_stab = simulate_memory(g1, gl, g3, delta_a, delta_b, **params)
    # # print(f'Maximum nbar stab = {jnp.max(nbar.real):.2f}')
    g1, gl, g3, delta_a, delta_b = hamiltonian_terms(phase, vatt, amp_ro, **params)
    b_ro = simulate_readout(g1, gl, g3, delta_a, delta_b, nbar, **params)
    b_ro_cal = simulate_readout(g1, gl, g3, delta_a, delta_b, 0, **params)
    return b_ro_cal, b_ro, nbar, b_stab


def plot_results(results, phases, vatts):
    fig, ax = plt.subplots(3, 2, figsize=(8, 9))
    for ind, (res, name) in enumerate(
        zip([results[1], results[3]], ["S - S_cal", r"$\beta$ stab"])
    ):
        for ind_2, (f, cmap) in enumerate(
            zip([jnp.abs, jnp.angle], ["hot", "twilight"])
        ):
            im = ax[ind, ind_2].imshow(
                f(res).T,
                extent=[phases[0], phases[-1], vatts[0], vatts[-1]],
                aspect="auto",
                cmap=cmap,
                origin="lower",
            )
            fig.colorbar(im, ax=ax[ind, ind_2])
            ax[ind, ind_2].set_xlabel("Phase")
            ax[ind, ind_2].set_ylabel("Vatt")
            ax[ind, ind_2].set_title(
                f"{f.__name__}({name})" + ("" if ind_2 == 0 else " rad")
            )
    ax[2, 0].set_title("nbar")
    im = ax[2, 0].imshow(
        results[2].real.T,
        extent=[phases[0], phases[-1], vatts[0], vatts[-1]],
        aspect="auto",
        cmap="hot",
        origin="lower",
    )
    fig.colorbar(im, ax=ax[2, 0])
    ax[2, 0].set_xlabel("Phase")
    ax[2, 0].set_ylabel("Vatt")
    fig.delaxes(ax[2, 1])
    fig.tight_layout()
    plt.show()
