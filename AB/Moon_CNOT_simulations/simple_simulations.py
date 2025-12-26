# %%

import dynamiqs as dq
import equinox as eqx
import GPUtil
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def set_free_gpu():
    gpus = GPUtil.getGPUs()
    gpu = min(gpus[::-1], key=lambda x: x.load)
    device = jax.devices("gpu")[gpu.id]
    jax.config.update("jax_default_device", device)
    print(f"Default device: GPU {gpu.id} ({gpu.name}) load={gpu.load*100}%")


set_free_gpu()

# %%
MHz, kHz = jnp.pi * 2, jnp.pi * 2 / 1e3
us = 1

# %%
N = 20
Nt = 101
alpha = jnp.sqrt(5)
kappa_2 = 1 * MHz
lambdas = jnp.linspace(0, 1, 3)


# %%
a = dq.destroy(N)
D, P = jnp.linalg.eigh(dq.dag(a) + a)
signX = P @ jnp.diag(jnp.sign(D)) @ dq.dag(P)


# %%
def get_moon_cat(alpha, lam):
    a = dq.destroy(N)
    L2 = a @ a - alpha**2 * (1 + jnp.abs(lam)) * dq.eye(N) + lam * dq.dag(a) @ a
    Hk = dq.dag(L2) @ L2
    _, eigv = jnp.linalg.eigh(Hk)
    eigv = eigv.reshape(N, N, 1)
    parity_0 = dq.expect(dq.parity(N), eigv[:, 0])
    catp = jnp.where(parity_0 > 0, eigv[:, 0], eigv[:, 1])
    catm = jnp.where(parity_0 > 0, eigv[:, 1], eigv[:, 0])
    catm *= jnp.sign(jnp.sum(catm).real)
    catp *= jnp.sign(jnp.sum(catp).real)
    log0, log1 = dq.unit(catp + catm), dq.unit(catp - catm)
    return jnp.array([catp, catm, log0, log1])


def simulate_CNOT(lam, kappa_a, kappa_phi, kerr, prep_c, prep_t, save_states=False):
    T_gate = 1 / kappa_2 / (1 + lam)
    ac, at = dq.destroy(N, N)
    I = dq.eye(N, N)
    L1c = jnp.sqrt(kappa_a) * ac
    L1t = jnp.sqrt(kappa_a) * at
    Lphic = jnp.sqrt(2 * kappa_phi) * dq.dag(ac) @ ac
    Lphit = jnp.sqrt(2 * kappa_phi) * dq.dag(at) @ at
    L2c = jnp.sqrt(kappa_2) * (
        ac @ ac - alpha**2 * I + lam * (dq.dag(ac) @ ac - alpha**2 * I)
    )
    c_ops = [L1c, L1t, L2c, Lphic, Lphit]
    gcnot = jnp.pi / T_gate / alpha / 4
    H_cnot = (
        gcnot * (dq.dag(ac) + ac - 2 * alpha * I) @ (dq.dag(at) @ at - alpha**2 * I)
    )
    H_k = -kerr / 2 * dq.dag(ac) @ dq.dag(ac) @ ac @ ac
    H_k += -kerr / 2 * dq.dag(at) @ dq.dag(at) @ at @ at
    tsave = jnp.linspace(0, T_gate, Nt)

    states = get_moon_cat(alpha, lam)
    rho_0 = dq.tensor(states[prep_c], states[prep_t])

    expects = [
        dq.tensor(dq.parity(N), dq.eye(N)),
        dq.tensor(dq.eye(N), dq.parity(N)),
        dq.tensor(signX, dq.eye(N)),
        dq.tensor(dq.eye(N), signX),
    ]

    output = dq.mesolve(
        H_cnot + H_k,
        c_ops,
        rho_0,
        tsave,
        exp_ops=expects,
        options=dq.Options(save_states=save_states),
    )
    angle = jnp.angle(dq.expect(a @ a, dq.ptrace(output.final_state, 1, (N, N)))) / 2
    alpha_ref = alpha * jnp.exp(1j * angle)
    lam_ref = lam * jnp.exp(2j * angle)
    L2t = jnp.sqrt(kappa_2) * (
        at @ at - alpha_ref**2 * (1 + lam) * I + lam_ref * dq.dag(at) @ at
    )
    output_ref = dq.mesolve(
        H_k,
        c_ops + [L2t],
        output.final_state,
        tsave,
        exp_ops=expects,
        options=dq.Options(save_states=save_states),
    )
    if save_states:
        state_concat = jnp.concatenate((output.states, output_ref.states[1:]), axis=0)
        output = eqx.tree_at(lambda m: m._saved.ysave, output, replace=state_concat)
    exp_concat = jnp.concatenate((output.expects, output_ref.expects[:, 1:]), axis=1)
    output = eqx.tree_at(lambda m: m._saved.Esave, output, replace=exp_concat)
    return output


# %%
lam = 0

kappa_a = 1 * kHz
kappa_phi = 0 * kHz
kerr = 0 * kHz
output = simulate_CNOT(lam, kappa_a, kappa_phi, kerr, 0, 0, True)

dq.plot.wigner_gif(dq.ptrace(output.states, 0, (N, N)))
dq.plot.wigner_gif(dq.ptrace(output.states, 1, (N, N)))

# %%
for i in range(4):
    plt.plot(output.expects[i, :])
    plt.show()


# %%
def error_from_trace(expect):
    return (1 - jnp.abs((expect[..., -1] / expect[..., 0]).real)) / 2


def errors(
    kappa_a: float,
    kappa_phi: float,
    kerr: float,
    lam: float,
):
    preps = jnp.array([0, 3])
    output = jax.vmap(simulate_CNOT, in_axes=(None, None, None, None, 0, 0))(
        lam, kappa_a, kappa_phi, kerr, preps, preps
    )
    eps_X = error_from_trace(output.expects[0, :2])
    eps_Z = error_from_trace(output.expects[1, 2:])
    return jnp.vstack((eps_X, eps_Z))


def errors_mapped_lam(kappa_a, kappa_phi, kerr, lams):
    return jax.vmap(errors, in_axes=(None, None, None, 0))(
        kappa_a, kappa_phi, kerr, lams
    )


# %%
kappa_a = kappa_2 * jnp.geomspace(1e-3, 1e-1, 11)
kappa_phi = 0
kerr = 0

errors_kappa_a = jax.vmap(errors_mapped_lam, in_axes=(0, None, None, None))(
    kappa_a, kappa_phi, kerr, lambdas
)


# %%
fig, ax = plt.subplots(2, 2)
cmap = plt.get_cmap("viridis")
for i, sub in enumerate(["X", "Z"]):
    for j, sur in enumerate(["C", "T"]):
        for ind_lam, lam in enumerate(lambdas):
            ax[i, j].plot(
                kappa_a / kappa_2,
                errors_kappa_a[:, ind_lam, i, j],
                # color=cmap(ind_lam / len(lambdas)),
                color=f"C{ind_lam}",
                ls="",
                marker="o",
                fillstyle="none",
            )
        ax[i, j].set_xscale("log")
        ax[i, j].set_yscale("log")
        ax[i, j].set_xlabel(r"$\kappa_a/\kappa_2$")
        ax[i, j].set_ylabel(f"$\\varepsilon_{sub}^{sur}$")
        ax[i, j].grid()
fig.tight_layout()

# %%
kappa_a = 0
kappa_phi = kappa_2 * jnp.geomspace(1e-3, 1e-1, 11)
kerr = 0

errors_kappa_phi = jax.vmap(errors_mapped_lam, in_axes=(None, 0, None, None))(
    kappa_a, kappa_phi, kerr, lambdas
)
# %%
fig, ax = plt.subplots(2, 2)
cmap = plt.get_cmap("viridis")
for i, sub in enumerate(["X", "Z"]):
    for j, sur in enumerate(["C", "T"]):
        for ind_lam, lam in enumerate(lambdas):
            ax[i, j].plot(
                kappa_phi / kappa_2,
                errors_kappa_phi[:, ind_lam, i, j],
                # color=cmap(ind_lam / len(lambdas)),
                color=f"C{ind_lam}",
                ls="",
                marker="o",
                fillstyle="none",
            )
        ax[i, j].set_xscale("log")
        ax[i, j].set_yscale("log")
        ax[i, j].set_xlabel(r"$\kappa_a/\kappa_2$")
        ax[i, j].set_ylabel(f"$\\varepsilon_{sub}^{sur}$")
        ax[i, j].grid()
fig.tight_layout()
# %%
kappa_a = 0
kappa_phi = 0
kerr = kappa_2 * jnp.geomspace(1e-3, 1e-1, 11)

errors_kerr = jax.vmap(errors_mapped_lam, in_axes=(None, None, 0, None))(
    kappa_a, kappa_phi, kerr, lambdas
)
# %%
fig, ax = plt.subplots(2, 2)
cmap = plt.get_cmap("viridis")
for i, sub in enumerate(["X", "Z"]):
    for j, sur in enumerate(["C", "T"]):
        for ind_lam, lam in enumerate(lambdas):
            ax[i, j].plot(
                kerr / kappa_2,
                errors_kerr[:, ind_lam, i, j],
                # color=cmap(ind_lam / len(lambdas)),
                color=f"C{ind_lam}",
                ls="",
                marker="o",
                fillstyle="none",
            )
        ax[i, j].set_xscale("log")
        ax[i, j].set_yscale("log")
        ax[i, j].set_xlabel(r"$\kappa_a/\kappa_2$")
        ax[i, j].set_ylabel(f"$\\varepsilon_{sub}^{sur}$")
        ax[i, j].grid()
fig.tight_layout()

# %%
kappa_a = 0
kappa_phi = 0
kerr = -kappa_2 * jnp.geomspace(1e-3, 1e-1, 11)

errors_kerr_neg = jax.vmap(errors_mapped_lam, in_axes=(None, None, 0, None))(
    kappa_a, kappa_phi, kerr, lambdas
)
# %%
fig, ax = plt.subplots(2, 2)
cmap = plt.get_cmap("viridis")
for i, sub in enumerate(["X", "Z"]):
    for j, sur in enumerate(["C", "T"]):
        for ind_lam, lam in enumerate(lambdas):
            ax[i, j].plot(
                jnp.abs(kerr) / kappa_2,
                errors_kerr_neg[:, ind_lam, i, j],
                # color=cmap(ind_lam / len(lambdas)),
                color=f"C{ind_lam}",
                ls="",
                marker="o",
                fillstyle="none",
            )
        ax[i, j].set_xscale("log")
        ax[i, j].set_yscale("log")
        ax[i, j].set_xlabel(r"$\kappa_a/\kappa_2$")
        ax[i, j].set_ylabel(f"$\\varepsilon_{sub}^{sur}$")
        ax[i, j].grid()
fig.tight_layout()

# %%
