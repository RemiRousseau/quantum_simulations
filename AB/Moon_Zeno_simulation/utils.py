from datetime import datetime

import dynamiqs as dq
import equinox as eqx
import jax
import jax.numpy as jnp
import scipy.optimize
from jax.typing import ArrayLike

MHz = 2 * jnp.pi
kHz = 1e-3 * MHz
us = 1
ns = 1e-3


def get_drive_amp(T_gate: float, alpha: float) -> float:
    return jnp.pi / 4 / T_gate / alpha


def get_t_gate(drive_amp: float, alpha: float) -> float:
    return jnp.pi / 4 / drive_amp / alpha


def get_H(P: ArrayLike, T_gate: float, alpha: float, kerr: float) -> ArrayLike:
    N0, _ = P.shape
    drive_amp = get_drive_amp(T_gate, alpha)
    a = dq.dag(P) @ dq.destroy(N0) @ P
    H = drive_amp * (a + dq.dag(a))
    H += kerr * dq.dag(a) @ dq.dag(a) @ a @ a / 2
    return H


def get_moon_dissipator(a: ArrayLike, alpha: float, lam: float) -> ArrayLike:
    """Returns the dissipator of the moon state.
    L = a**2 - alpha**2*(1+lambda) + lambda*a_dag*a"""
    alpha_moon = alpha * jnp.sqrt(1 + lam)
    dissip = a @ a - alpha_moon**2 * dq.eye(a.shape[-1]) + lam * dq.dag(a) @ a
    return dissip


def get_Ls(
    P: ArrayLike,
    alpha: float,
    lam: float,
    kappa_2: float,
    kappa_a: float,
    kappa_phi: float,
    nth: float,
) -> list[ArrayLike]:
    N0, _ = P.shape
    a = dq.dag(P) @ dq.destroy(N0) @ P
    k_sqrt = jnp.sqrt(
        jnp.array([kappa_a * (1 + nth), kappa_a * nth, 2 * kappa_phi, kappa_2])
    )
    ops = [a, dq.dag(a), dq.dag(a) @ a, get_moon_dissipator(a, alpha, lam)]
    dissipators = [ks * op for ks, op in zip(k_sqrt, ops)]
    return dissipators


def get_moon_base(N0: int, N1: int, alpha: float, lam: float) -> ArrayLike:
    """Returns the eigenvalues and eigenvectors of the moon Kerr Hamiltonian."""
    a = dq.destroy(N0)
    dissip = get_moon_dissipator(a, alpha, lam)
    Hkerr = dq.dag(dissip) @ dissip
    _, P = jnp.linalg.eigh(Hkerr)
    P = P[:, :N1]
    parity_0 = dq.expect(dq.parity(P.shape[-2]), P[:, 0].reshape(P.shape[:-1] + (1,)))
    cat_p = jnp.where(parity_0 > 0, P[:, 0], P[:, 1])
    cat_m = jnp.where(parity_0 > 0, P[:, 1], P[:, 0])
    cat_m = cat_m * jnp.sign(jnp.sum(cat_m).real)
    P = P.at[..., :, :2].set(jnp.stack([cat_p, cat_m], axis=-1))
    return P


def get_moon_cat_from_base(base_moon: ArrayLike) -> tuple[ArrayLike]:
    """Returns the cat states from the eigenvectors of the moon Kerr Hamiltonian."""
    N1 = base_moon.shape[-1]
    base_moon = base_moon.reshape(base_moon.shape + (1,))
    return dq.unit(base_moon[..., :N1, 0, :]), dq.unit(base_moon[..., :N1, 1, :])


def get_moon_cat(N0: int, N1: int, alpha: complex, lam: complex) -> tuple[ArrayLike]:
    """Returns the cat states of the moon Kerr Hamiltonian."""
    return get_moon_cat_from_base(get_moon_base(N0, N1, alpha, lam))


def get_rho0(N1: int, ind_state: int) -> ArrayLike:
    """Returns the initial state of the system.
    0 for |0>, 1 for |1>, 2 for |+>, 3 for |->."""
    cat_p, cat_m = dq.fock(N1, 0), dq.fock(N1, 1)
    log_0, log_1 = dq.unit(cat_p + cat_m), dq.unit(cat_p - cat_m)
    return jnp.array([log_0, log_1, cat_p, cat_m])[ind_state]


def get_signX(N0: int) -> ArrayLike:
    """Returns the signX operator."""
    a = dq.destroy(N0)
    D, P = jnp.linalg.eigh(dq.dag(a) + a)
    return P @ jnp.diag(jnp.sign(D)) @ dq.dag(P)


def get_exp_ops(P: ArrayLike) -> list[ArrayLike]:
    """Returns the expectation operators in the following order:
    - signX
    - parity
    - commutator
    - nbar"""
    N0, _ = P.shape
    a = dq.destroy(N0)
    ops = [get_signX(N0), dq.parity(N0), a @ dq.dag(a) - dq.dag(a) @ a, dq.dag(a) @ a]
    return [dq.dag(P) @ op @ P for op in ops]


def to_fock_basis(states: ArrayLike, P: ArrayLike) -> ArrayLike:
    """Transforms the states to the Fock basis."""
    N1 = P.shape[-1]
    P_cut = P[:N1]
    fock_basis_states = P_cut @ states @ dq.dag(P_cut)
    return dq.unit(fock_basis_states)


def expected_parity(
    alpha: float,
    lam: float,
    kappa_a: float,
    kappa_2: float,
    amp_gate: float,
    nth: float,
    times: ArrayLike,
    **_,
) -> ArrayLike:
    amp_gate += get_opt_amp_gate(alpha, lam, kappa_a, kappa_2, nth) * (amp_gate == 0)
    omega = 4 * alpha * amp_gate
    gamma = 2 * alpha**2 * kappa_a * (1 + 2 * nth)
    gamma += 2 * amp_gate**2 / kappa_2 / alpha**2 / (1 + lam) ** 2
    return jnp.exp(-gamma * times) * jnp.cos(omega * times)


def get_opt_T_gate(
    alpha: float, lam: float, kappa_a: float, kappa_2: float, nth: float
) -> float:
    return (
        jnp.pi / 4 / alpha**3 / (1 + lam) / jnp.sqrt(kappa_2 * kappa_a * (1 + 2 * nth))
    )


def get_opt_amp_gate(
    alpha: float, lam: float, kappa_a: float, kappa_2: float, nth: float
) -> float:
    T_gate = get_opt_T_gate(alpha, lam, kappa_a, kappa_2, nth)
    return get_drive_amp(T_gate, alpha)


def get_opt_eps_X(
    alpha: float,
    lam: float,
    kappa_a: float,
    kappa_2: float,
    nth: float,
    **_,
) -> ArrayLike:
    exp_term = jnp.pi * jnp.sqrt(kappa_a * (1 + 2 * nth) / kappa_2) / alpha / (1 + lam)
    return (1 - jnp.exp(-exp_term)) / 2


def get_eps_X_from_expects(expects: ArrayLike) -> ArrayLike:
    x_prop = jnp.abs(jnp.min(expects.real, axis=-1) / expects.real[..., 0])
    return (1 - x_prop) / 2


def simulate_zeno(
    N0: int,
    N1: int,
    alpha: float,
    lam: complex,
    kappa_2: float,
    kappa_a: float,
    kappa_phi: float,
    nth: float,
    kerr: float,
    amp_gate: float = 0.0,
    tsave: ArrayLike | None = None,
    Nt: int | None = 101,
    rho_0: int = 2,
    save_states: bool | None = False,
    progress_bar: bool | None = True,
):
    amp_gate += get_opt_amp_gate(alpha, lam, kappa_a, kappa_2, nth) * (amp_gate == 0)
    T_gate = get_t_gate(amp_gate, alpha)
    if tsave is None and Nt is None:
        raise ValueError("Either tsave or Nt must be given.")
    if tsave is None:
        tsave = jnp.linspace(0, 1.2 * T_gate, Nt)
    else:
        tsave = jnp.array(tsave)
        Nt = tsave.size

    P = get_moon_base(N0, N1, alpha, lam)
    rho0 = get_rho0(N1, rho_0)
    hamiltonian = get_H(P, T_gate, alpha, kerr)
    dissipators = get_Ls(P, alpha, lam, kappa_2, kappa_a, kappa_phi, nth)
    exp_ops = get_exp_ops(P)
    kwargs = {"progress_meter": None} if not progress_bar else {}
    output = dq.mesolve(
        hamiltonian,
        dissipators,
        rho0,
        tsave,
        exp_ops=exp_ops,
        options=dq.Options(save_states=save_states, verbose=progress_bar, **kwargs),
    )

    if save_states:
        fock_basis_states = to_fock_basis(output.states, P)
        output = eqx.tree_at(
            lambda m: m._saved.ysave, output, replace=fock_basis_states
        )
    return output


def zeno_eps_X(
    N0: int,
    N1: int,
    alpha: float,
    lam: complex,
    kappa_2: float,
    kappa_a: float,
    kappa_phi: float,
    nth: float,
    kerr: float,
    amp_gate: float,
    tsave: float,
    Nt: int,
    rho_0: int,
):
    output = simulate_zeno(
        N0,
        N1,
        alpha,
        lam,
        kappa_2,
        kappa_a,
        kappa_phi,
        nth,
        kerr,
        amp_gate,
        tsave,
        Nt,
        rho_0,
    )
    return get_eps_X_from_expects(output.expects[1]), output.expects[3, 0].real


def sweep_params_simulation(
    lambdas: ArrayLike,
    alphas: ArrayLike,
    relative_amps: ArrayLike,
    params: dict[str, float],
    BATCH_SIZE: int = 500,
) -> tuple[jax.Array]:
    N_lam, N_alpha, N_T = lambdas.size, alphas.size, relative_amps.size

    lambdas_map, alphas_map = jnp.meshgrid(lambdas, alphas, indexing="ij")
    t_gates = get_opt_T_gate(
        alphas_map,
        lambdas_map,
        params["kappa_a"],
        params["kappa_2"],
        params["nth"],
    )
    amps = get_drive_amp(t_gates, alphas_map)
    amps_map = relative_amps * amps[:, :, None]
    els = [lambdas_map, alphas_map]
    for _i in range(len(els)):
        els[_i] = jnp.repeat(els[_i], N_T, axis=-1)
        els[_i] = els[_i].reshape(amps_map.shape)
        els[_i] = els[_i].flatten()
    lambdas_map, alphas_map = els
    amps_map = amps_map.flatten()

    n_to_batch = alphas_map.size
    n_per_batch = jnp.ceil(n_to_batch / BATCH_SIZE).astype(int)
    cutted_lambdas_map = jnp.array_split(lambdas_map, n_per_batch, axis=-1)
    cutted_alphas_map = jnp.array_split(alphas_map, n_per_batch, axis=-1)
    cutted_amps_map = jnp.array_split(amps_map, n_per_batch, axis=-1)
    length_cutted = jnp.array([el.shape[-1] for el in cutted_lambdas_map])
    indexes = jnp.concatenate((jnp.array([0]), jnp.cumsum(length_cutted)))
    n_batch = len(length_cutted)

    in_axes = [None for _ in range(13)]
    for ind in [2, 3, 9]:
        in_axes[ind] = 0
    v_zeno_eps_X = jax.vmap(zeno_eps_X, in_axes=tuple(in_axes))

    eps_Xs = jnp.zeros(n_to_batch)
    nbars = jnp.zeros(n_to_batch)

    t_beg = datetime.now()
    print(f"{n_batch} batches. {n_to_batch} simulations total.")
    for i, (als, lms, ams) in enumerate(
        zip(cutted_alphas_map, cutted_lambdas_map, cutted_amps_map)
    ):
        eps_X, nb = v_zeno_eps_X(
            params["N0"],
            params["N1"],
            als,
            lms,
            params["kappa_2"],
            params["kappa_a"],
            params["kappa_phi"],
            params["nth"],
            params["kerr"],
            ams,
            params["tsave"],
            params["Nt"],
            params["rho_0"],
        )
        eps_Xs = eps_Xs.at[indexes[i] : indexes[i + 1]].set(eps_X)
        nbars = nbars.at[indexes[i] : indexes[i + 1]].set(nb)
        t_loop = datetime.now()
        elapsed_time = t_loop - t_beg
        t_per_loop = elapsed_time / (i + 1)
        residual_time = t_per_loop * (n_batch - i - 1)
        to_print = f"{i + 1}/{n_batch}"
        to_print += f" | elapsed time = {str(elapsed_time).split('.')[0]}"
        to_print += f" | time per batch = {str(t_per_loop).split('.')[0]}"
        to_print += f" | time to end = {str(residual_time).split('.')[0]}"
        print(to_print)
    eps_Xs = eps_Xs.reshape((N_lam, N_alpha, N_T))
    nbars = nbars.reshape((N_lam, N_alpha, N_T))
    amps_map = amps_map.reshape((N_lam, N_alpha, N_T))
    return eps_Xs, nbars, amps_map


def optimal_gates(
    lambdas: ArrayLike,
    alphas: ArrayLike,
    relative_amps: ArrayLike,
    params: dict[str, float],
    BATCH_SIZE: int = 500,
):
    N_amp = relative_amps.size
    eps_Xs, nbars, amps_map = sweep_params_simulation(
        lambdas, alphas, relative_amps, params, BATCH_SIZE
    )
    ind_mins = jnp.argmin(eps_Xs, axis=-1)
    if jnp.sum(ind_mins == 0) > 0:
        print(f"Warning: {jnp.sum(ind_mins==0)} minimum pZ at amps[0].")
    if jnp.sum(ind_mins == N_amp - 1) > 0:
        print(f"Warning: {jnp.sum(ind_mins==N_amp-1)} minimum pZ at amps[-1].")
    j, k = jnp.indices(ind_mins.shape)
    amps_opt = amps_map[j, k, ind_mins]
    eps_X_opt = eps_Xs[j, k, ind_mins]
    return eps_X_opt, nbars[j, k, ind_mins], amps_opt


def fit_function_zeno_calib(x, alpha, lam, tsave, amps):
    k2, k1, t0, A, B = x
    outputs = jax.vmap(
        simulate_zeno, in_axes=(None,) * 9 + (0, None, None, None, None, None)
    )(
        150,
        20,
        alpha,
        lam,
        k2,
        k1,
        0.0,
        0.0,
        15 * kHz,
        amps,
        tsave + t0,
        None,
        2,
        False,
        False,
    )
    parity = outputs.expects[:, 1].real
    return A * parity + B


def square_dist(x, alpha, lam, tsave, amps, data):
    Ss = fit_function_zeno_calib(x, alpha, lam, tsave, amps)
    return (Ss - data).flatten() ** 2


def fit_experimental_data(alpha, lam, tsave, amps, data, guess):
    # jac = jax.jacobian(square_dist)
    res = scipy.optimize.least_squares(
        square_dist,
        guess,
        args=(alpha, lam, tsave, amps, data),
        bounds=[[0, 0, 0, -jnp.inf, -jnp.inf], jnp.inf],
        ftol=1e-15,
        gtol=1e-15,
        xtol=1e-15,
    )
    return res
