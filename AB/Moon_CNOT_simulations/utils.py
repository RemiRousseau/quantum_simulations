from collections import OrderedDict
from datetime import datetime
from typing import Optional

import dynamiqs as dq
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

MHz = 2 * jnp.pi
kHz = 1e-3 * MHz
us = 1
ns = 1e-3


def get_theoretical_gamma_z_control(
    T_gate: float, alpha: float, lam: float, kappa_a: float, kappa_2: float
) -> float:
    g_cnot = get_gcnot(T_gate, alpha)
    return 2 * alpha**2 * kappa_a + g_cnot**2 / kappa_2 / (1 + lam) ** 2


def get_theoretical_gamma_z_target(alpha: float, kappa_a: float) -> float:
    return 2 * alpha**2 * kappa_a


def get_pz_from_kappa(kappa: float, t: float) -> float:
    return (1 - jnp.exp(-kappa * t)) / 2


def get_optimal_T_gate(
    alpha: float, lam: float, kappa_a: float, kappa_2: float
) -> float:
    """Returns the optimal CNOT gate duration."""
    return jnp.pi / jnp.sqrt(kappa_a * kappa_2) / 8 / alpha**2 / (1 + lam)


def get_moon_dissipator(a: ArrayLike, alpha: float, lam: float) -> ArrayLike:
    """Returns the dissipator of the moon state.
    L = a**2 - alpha**2*(1+lambda) + lambda*a_dag*a.
    """
    alpha_moon = alpha * jnp.sqrt(1 + lam)
    dissip = a @ a - alpha_moon**2 * dq.eye(a.shape[-1]) + lam * dq.dag(a) @ a
    return dissip


def get_moon_base(N0: int, N1: int, alpha: complex, lam: complex) -> ArrayLike:
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


def get_gcnot(T_gate: complex, alpha: float) -> complex:
    """Returns the coupling strength of the CNOT gate."""
    return jnp.pi / T_gate / alpha / 4


def get_H(P: ArrayLike, P_left: ArrayLike, T_gate: float, alpha: float, kerr: float):
    """Returns the Hamiltonian of the CNOT gate."""
    N0, N1 = P.shape
    g_cnot = get_gcnot(T_gate, alpha)
    P_l = dq.tensor(P_left, dq.eye(N1))
    a_kerr = dq.dag(P) @ dq.destroy(N0) @ P
    a_ctrl, a_target = dq.tensor(a_kerr, dq.eye(N1)), dq.tensor(dq.eye(N1), a_kerr)

    Ia = dq.eye(N1**2)
    H = g_cnot * (dq.dag(a_ctrl) + a_ctrl - 2 * alpha * Ia)
    H = H @ (dq.dag(a_target) @ a_target - alpha**2 * Ia)
    H += jnp.pi / T_gate * P_l @ (dq.dag(a_target) @ a_target - alpha**2 * Ia)

    H += -kerr * dq.dag(a_ctrl) @ dq.dag(a_ctrl) @ a_ctrl @ a_ctrl / 2
    H += -kerr * dq.dag(a_target) @ dq.dag(a_target) @ a_target @ a_target / 2
    return H


def get_Ls(
    P: ArrayLike,
    alpha: complex,
    lam: complex,
    kappa_2: complex,
    kappa_a: complex,
    kappa_phi: complex,
    stabilized_target: bool,
) -> list[ArrayLike]:
    """Returns the dissipators of the CNOT gate."""
    N0, N1 = P.shape
    a_kerr = dq.dag(P) @ dq.destroy(N0) @ P
    a_ctrl, a_target = dq.tensor(a_kerr, dq.eye(N1)), dq.tensor(dq.eye(N1), a_kerr)
    k_sqrt = jnp.sqrt(jnp.array([kappa_a, 2 * kappa_phi, kappa_2]))
    dissipators = [
        k_sqrt[0] * a_ctrl,
        k_sqrt[0] * a_target,
        k_sqrt[1] * dq.dag(a_ctrl) @ a_ctrl,
        k_sqrt[1] * dq.dag(a_target) @ a_target,
        k_sqrt[2] * get_moon_dissipator(a_ctrl, alpha, lam),
        k_sqrt[2] * get_moon_dissipator(a_target, alpha, lam) * stabilized_target,
    ]
    # if stabilized_target:
    #     dissipators.append(
    #         jnp.sqrt(kappa_2) * get_moon_dissipator(a_target, alpha, lam)
    #     )
    return dissipators


def get_signX(N0: int) -> ArrayLike:
    """Returns the signX operator."""
    a = dq.destroy(N0)
    D, P = jnp.linalg.eigh(dq.dag(a) + a)
    return P @ jnp.diag(jnp.sign(D)) @ dq.dag(P)


def get_P_right_P_left(P: ArrayLike) -> ArrayLike:
    """Returns the right and left projectors."""
    N0 = P.shape[0]
    signX = get_signX(N0)
    P_right, P_left = (dq.eye(N0) + signX) / 2, (dq.eye(N0) - signX) / 2
    P_right, P_left = dq.dag(P) @ P_right @ P, dq.dag(P) @ P_left @ P
    return P_right, P_left


def get_exp_ops(P: ArrayLike) -> list[ArrayLike]:
    """Returns the expectation operators in the following order:
    - signX_ctrl @ Ia
    - Ia @ signX_target
    - parity_ctrl @ Ia
    - Ia @ parity_target
    - commutator_ctrl @ Ia
    - Ia @ commutator_target
    - nbar_ctrl @ Ia
    - Ia @ nbar_target.
    """
    N0, N1 = P.shape
    Ia = dq.eye(N1)
    a = dq.destroy(N0)
    ops = [get_signX(N0), dq.parity(N0), a @ dq.dag(a) - dq.dag(a) @ a, dq.dag(a) @ a]
    expect = []
    for op in ops:
        op_kerr = dq.dag(P) @ op @ P
        expect.append(dq.tensor(op_kerr, Ia))
        expect.append(dq.tensor(Ia, op_kerr))
    return expect


def get_U_lab(P, P_right, P_left, time_o_t_gate, alpha):
    """Returns the unitary transformation to the lab frame."""
    N0, N1 = P.shape[-2:]
    P_right, P_left = dq.tensor(P_right, dq.eye(N1)), dq.tensor(P_left, dq.eye(N1))
    a_target = dq.tensor(dq.eye(N1), dq.dag(P) @ dq.destroy(N0) @ P)
    H0 = dq.dag(a_target) @ a_target - alpha**2 * dq.eye(N1**2)
    return P_right + P_left @ dq.expm(-1j * jnp.pi * time_o_t_gate * H0)


def to_lab_frame(states, P, P_right, P_left, time_o_t_gate, alpha):
    """Transforms the states to the lab frame."""
    U_t = get_U_lab(P, P_right, P_left, time_o_t_gate, alpha)
    return dq.unit(dq.dag(U_t) @ states @ U_t)


def to_fock_basis(states, P):
    """Transforms the states to the Fock basis."""
    N1 = P.shape[-1]
    P_both = dq.tensor(P[:N1], P[:N1])
    fock_basis_states = P_both @ states @ dq.dag(P_both)
    return dq.unit(fock_basis_states)


def get_rho0(N1: int, ind_state: int):
    """Returns the initial state of the system.
    0 for |0>, 1 for |1>, 2 for |+>, 3 for |->.
    """
    cat_p, cat_m = dq.fock(N1, 0), dq.fock(N1, 1)
    log_0, log_1 = dq.unit(cat_p + cat_m), dq.unit(cat_p - cat_m)
    return jnp.array([log_0, log_1, cat_p, cat_m])[ind_state]


def simulate_cnot(
    N0: int,
    N1: int,
    alpha: float,
    lam: complex,
    kappa_2: float,
    kappa_a: float,
    kappa_phi: float,
    kerr: float,
    T_gate: float,
    Nt: int,
    stabilized: bool,
    rho_ctrl_0: int,
    rho_target_0: int,
    save_states: Optional[bool] = False,
    refocus: Optional[bool] = True,
) -> dq.result.MESolveResult:
    """Simulates the CNOT gate operation on a quantum system.

    Args:
        N0 (int): Number of basis states for the kerr basis computation.
        N1 (int): Number of basis states for the simulation.
        alpha (complex): Moon cat blob center position.
        lam (complex): Moon cat squeezing parameter.
        kappa_2 (float): Two photon decay rate.
        kappa_a (float): One photon decay rate.
        kappa_phi (float): Dephasing rate.
        kerr (float): Kerr nonlinearity.
        T_gate (float): CNOT gate duration. If T_gate = 0, the optimal duration is used.
        Nt (int): Number of time steps.
        stabilized (bool): Flag indicating whether to stabilized target.
        rho_ctrl_0 (int): Initial state of the control qubit. 0 for |0>, 1 for |1>, 2 for |+>, 3 for |->.
        rho_target_0 (int): Initial state of the target qubit. 0 for |0>, 1 for |1>, 2 for |+>, 3 for |->.
        save_states (bool, optional): Flag indicating whether to save density matrices. Defaults to False.

    Returns:
        dq.MEResult: Result of the simulation.

    """
    T_gate += get_optimal_T_gate(alpha, lam, kappa_a, kappa_2) * (T_gate == 0)
    T_gate = jnp.where(jnp.isnan(T_gate), -1, T_gate)
    T_gate += (1 + 1 / (kappa_2 * (1 + lam))) * (T_gate == -1)
    tsave = jnp.linspace(0, T_gate, Nt)

    P = get_moon_base(N0, N1, alpha, lam)
    rho0 = dq.tensor(get_rho0(N1, rho_ctrl_0), get_rho0(N1, rho_target_0))
    P_right, P_left = get_P_right_P_left(P)
    H_cnot = get_H(P, P_left, T_gate, alpha, kerr)
    dissipators = get_Ls(P, alpha, lam, kappa_2, kappa_a, kappa_phi, stabilized)
    exp_ops = get_exp_ops(P)
    output = dq.mesolve(
        H_cnot,
        dissipators,
        rho0,
        tsave,
        exp_ops=exp_ops,
        options=dq.Options(save_states=save_states),
    )

    refocus_time = 1 / (kappa_2 * (1 + lam) ** 2) * (1 / 100 + refocus * 99 / 100)
    tsave_refocus = jnp.linspace(0, refocus_time, Nt)
    rho0_refocus = output.final_state
    H_refocus = get_H(P, P_left, jnp.inf, alpha, kerr)
    dissipators_refocus = get_Ls(P, alpha, lam, kappa_2, kappa_a, kappa_phi, True)
    output_ref = dq.mesolve(
        H_refocus,
        dissipators_refocus,
        rho0_refocus,
        tsave_refocus,
        exp_ops=exp_ops,
        options=dq.Options(save_states=save_states),
    )

    if save_states:
        lab_frame_states = to_lab_frame(
            output.states, P, P_right, P_left, tsave[:, None, None] / T_gate, alpha
        )
        lab_frame_states_refocus = to_lab_frame(
            output_ref.states, P, P_right, P_left, 1, alpha
        )
        full_state = jnp.concatenate(
            (lab_frame_states, lab_frame_states_refocus[1:]), axis=0
        )
        expects = jnp.concatenate((output.expects, output_ref.expects[:, 1:]), axis=-1)
        expects = expects.at[..., :-1].set(dq.expect(exp_ops, full_state[:-1]))
        output = eqx.tree_at(lambda m: m._saved.Esave, output, replace=expects)
        fock_basis_states = to_fock_basis(full_state, P)
        output = eqx.tree_at(
            lambda m: m._saved.ysave, output, replace=fock_basis_states
        )
    else:
        state_concat = jnp.concatenate((output.states, output_ref.states[1:]), axis=0)
        exp_concat = jnp.concatenate(
            (output.expects, output_ref.expects[:, 1:]), axis=1
        )
        output = eqx.tree_at(lambda m: m._saved.ysave, output, replace=state_concat)
        output = eqx.tree_at(lambda m: m._saved.Esave, output, replace=exp_concat)
    tsave_tot = jnp.concatenate((output.tsave, output_ref.tsave[1:] + output.tsave[-1]))
    output = eqx.tree_at(lambda m: m.tsave, output, replace=tsave_tot)
    return output


def get_eps_from_expect(expects: ArrayLike) -> ArrayLike:
    return jnp.abs((1 - jnp.abs((expects[-1] / expects[0]).real)) / 2)


def get_error_cnot(kwargs: dict[str : complex | float | bool]) -> ArrayLike:
    """Returns the error of the CNOT gate."""
    plot_gifs = kwargs.pop("plot_gifs", False)
    kwargs["rho_ctrl_0"] = jnp.array([1, 2])
    kwargs["rho_target_0"] = jnp.array([1, 2])
    kwargs["save_states"] = plot_gifs
    axes_map = {k: None for k in kwargs}
    axes_map["rho_ctrl_0"] = 0
    axes_map["rho_target_0"] = 0
    output = jax.vmap(lambda params: simulate_cnot(**params), in_axes=(axes_map,))(
        kwargs
    )
    if plot_gifs:
        plot_gif(output.states[0], 0)
        plot_gif(output.states[0], 1)
        plot_gif(output.states[1], 0)
        plot_gif(output.states[1], 1)
    epsZ_ctrl = get_eps_from_expect(output.expects[0, 0])
    epsZ_target = get_eps_from_expect(output.expects[0, 1])
    epsX_control = get_eps_from_expect(output.expects[1, 2])
    epsX_target = get_eps_from_expect(output.expects[1, 3])

    nbars = output.expects[1, -2:, 0]
    T_gate = output.tsave[0, -1]
    return (
        jnp.array([epsZ_ctrl, epsZ_target, epsX_control, epsX_target]),
        nbars.real,
        T_gate,
    )


def plot_gif(states: ArrayLike, ind: int):
    """Plots the Wigner function gif of the states."""
    N1 = jnp.sqrt(states.shape[-1]).astype(int)
    dq.plot.wigner_gif(dq.ptrace(states, ind, (N1, N1)))


def batch_cnot_errors(
    params: dict[str : complex | float | bool],
    sweep_params: dict[str, ArrayLike],
    max_batch_size: int,
    cathesian_batching: Optional[bool] = True,
):
    """Returns the error of the CNOT gate for a batch of parameters."""
    valid_params = [
        "N0",
        "N1",
        "alpha",
        "lam",
        "kappa_2",
        "kappa_a",
        "kappa_phi",
        "kerr",
        "T_gate",
        "Nt",
        "stabilized",
    ]
    for k in params:
        assert k in valid_params, f"Invalid parameter: {k}."
    sweepable_params = [
        "alpha",
        "lam",
        "kappa_2",
        "kappa_a",
        "kappa_phi",
        "kerr",
        "T_gate",
        "stabilized",
    ]
    for k in sweep_params:
        assert k in sweepable_params, f"Invalid parameter: {k}."
    assert len(sweep_params) > 0, "No sweepable parameters."
    assert jnp.all(
        jnp.array([k not in params for k in sweep_params])
    ), "Sweepable parameters must not be in the parameters dict."
    n_sweep = len(sweep_params)
    assert n_sweep + len(params) == len(valid_params), "Invalid parameters number."

    sweep_params = OrderedDict(sweep_params)
    if cathesian_batching:
        dimensions = tuple()
        for v in sweep_params.values():
            dimensions += v.shape
        dimensions = dimensions[::-1]
        cart_prod = jnp.array(
            jnp.meshgrid(*sweep_params.values(), indexing="ij")
        ).reshape(n_sweep, -1)
    else:
        dimensions = sweep_params[list(sweep_params.keys())[0]].shape
        assert jnp.all(
            jnp.array([v.shape == dimensions for v in sweep_params.values()])
        ), "All sweeped parameters must have the same shape for not carthesian batching."
        cart_prod = jnp.stack(list(sweep_params.values()), axis=0)
    n_to_batch = cart_prod.shape[-1]
    n_per_batch = jnp.ceil(n_to_batch / max_batch_size).astype(int)
    cutted_params = jnp.array_split(cart_prod, n_per_batch, axis=-1)
    length_cutted = jnp.array([el.shape[-1] for el in cutted_params])
    indexes = jnp.concatenate((jnp.array([0]), jnp.cumsum(length_cutted)))
    n_batch = len(length_cutted)

    axes_map = {k: None for k in params}
    axes_map.update({k: 0 for k in sweep_params})
    vget_error_cnot = jax.vmap(get_error_cnot, in_axes=(axes_map,))
    errors = jnp.zeros((n_to_batch, 4))
    nbars = jnp.zeros((n_to_batch, 2))
    tgates = jnp.zeros(n_to_batch)

    t_beg = datetime.now()
    print(f"{n_batch} batches.")
    for i, par in enumerate(cutted_params):
        batch_params = {k: v for k, v in zip(sweep_params.keys(), par)}
        batch_params.update(params)
        sub_err, sub_n, tg = vget_error_cnot(batch_params)
        errors = errors.at[indexes[i] : indexes[i + 1]].set(sub_err)
        nbars = nbars.at[indexes[i] : indexes[i + 1]].set(sub_n)
        tgates = tgates.at[indexes[i] : indexes[i + 1]].set(tg)

        t_loop = datetime.now()
        elapsed_time = t_loop - t_beg
        t_per_loop = elapsed_time / (i + 1)
        residual_time = t_per_loop * (n_batch - i - 1)
        to_print = f"{i + 1}/{n_batch}"
        to_print += f" | elapsed time = {str(elapsed_time).split('.')[0]}"
        to_print += f" | time per batch = {str(t_per_loop).split('.')[0]}"
        to_print += f" | time to end = {str(residual_time).split('.')[0]}"
        print(to_print)
    errors = errors.reshape(dimensions[::-1] + (4,))
    nbars = nbars.reshape(dimensions[::-1] + (2,))
    tgates = tgates.reshape(dimensions[::-1])
    return errors, nbars, tgates
