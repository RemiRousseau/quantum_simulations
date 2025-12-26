# %%
import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

MHz = 2 * jnp.pi
kHz = MHz * 1e-3
s = 1e6
us = 1
ns = 1e-3
dq.set_precision("double")


# %%
def simulate_bit_flip(N0, N1, alpha_2, kappa_2, kappa_1, xi_z, t_max):
    a = dq.destroy(N0)

    L2 = jnp.sqrt(kappa_2) * (a @ a - alpha_2 * dq.eye(N0))
    H_kerr = dq.dag(L2) @ L2
    _, U = jnp.linalg.eigh(H_kerr)
    U_cut = U[:, :N1]

    D, P = jnp.linalg.eigh(dq.dag(a) + a)
    signX = P @ jnp.diag(jnp.sign(D)) @ dq.dag(P)
    signX_k = dq.dag(U_cut) @ signX @ U_cut

    a_k = dq.dag(U_cut) @ a @ U_cut
    L2_k = dq.dag(U_cut) @ L2 @ U_cut
    H = jnp.conjugate(xi_z) * a_k + xi_z * dq.dag(a_k)
    L1_k = jnp.sqrt(kappa_1) * a_k

    par_0 = dq.expect(dq.parity(N0), U_cut[:, 0].reshape(1, -1)) > 0
    rho_0 = dq.unit(
        jnp.where(
            par_0,
            dq.fock(N1, 0) + dq.fock(N1, 1),
            dq.fock(N1, 0) - dq.fock(N1, 1),
        )
    )
    xi_z = jnp.where(par_0, xi_z, -xi_z)
    # tsave = [0, t_max/10, t_max]
    tsave = jnp.linspace(0, t_max, 100)
    output = dq.mesolve(H, [L2_k, L1_k], rho_0, tsave, exp_ops=[signX_k])
    expX = output.expects[0].real
    expX = jnp.where(expX[0] < 0, -expX, expX)
    return tsave, expX


# %%
N0, N1 = 150, 20
alpha_2 = jnp.linspace(2, 4, 11)
kappa_2 = 1 * MHz
kappa_1 = 100 * kHz
xi_z = jnp.geomspace(1e-2, 1, 61) * 1 * MHz * jnp.exp(1j * jnp.pi / 2)
t_max = 10 * us
alpha_2s, xi_zs = jnp.meshgrid(alpha_2, xi_z, indexing="ij")
mapped_s_bit_flip = jax.vmap(
    lambda alpha_2, xi: simulate_bit_flip(N0, N1, alpha_2, kappa_2, kappa_1, xi, t_max)
)
tsave, expX = jax.vmap(mapped_s_bit_flip)(alpha_2s, xi_zs)
tsave = tsave[0, 0]

# %%
colors = plt.get_cmap("viridis")(jnp.linspace(0, 1, len(alpha_2)))
fig, ax = plt.subplots()
for i, al in enumerate(alpha_2):
    ax.plot(tsave / us, expX[i].T, color=colors[i], label=f"$\\alpha_2 = {al}$")
plt.show()

Tbf = (tsave[-1] - tsave[1]) / jnp.log(expX[:, :, 1] / expX[:, :, -1])
fig, ax = plt.subplots()
for i, al in enumerate(alpha_2):
    ax.plot(jnp.abs(xi_z) / MHz, Tbf[i].T / us, color=colors[i], label=f"$\\alpha_2 = {al}$")
ax.loglog()
ax.legend()
plt.show()
