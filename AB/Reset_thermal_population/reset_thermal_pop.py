# %%
import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt

# %%
MHz = 2 * jnp.pi
kHz = MHz * 1e-3
us = 1

# %%
# Define parameters
Na, Nb = 10, 4
g_r = 0.5 * MHz
kappa_a = 2 * kHz
kappa_b = 20 * MHz
nth_a = 1
nth_b = 1e-2
tmax = 20 * us

kappa_eff = 4 * g_r**2 / (kappa_b + kappa_a)
print(f"kappa_eff = {kappa_eff/MHz:.2f} MHz")
n_th_exp = (nth_a + nth_b * kappa_eff / (kappa_b + kappa_eff)) / (
    kappa_a + kappa_eff + kappa_eff**2 / (kappa_b + kappa_eff)
)
print(f"n_th_exp = {n_th_exp:.2f}")

a, b = dq.destroy(Na, Nb)
H = g_r * (dq.dag(a) @ b + a @ dq.dag(b))
La_u = jnp.sqrt(kappa_a * (1 + nth_a)) * a
La_d = jnp.sqrt(kappa_a * nth_a) * dq.dag(a)
Lb_u = jnp.sqrt(kappa_b * (1 + nth_b)) * b
Lb_d = jnp.sqrt(kappa_b * nth_b) * dq.dag(b)

tsave = jnp.linspace(0, tmax, 100)

rho = dq.coherent((Na, Nb), (0, 0))
output = dq.mesolve(H, [La_u, La_d, Lb_u, Lb_d], rho, tsave)
ada = dq.expect(dq.dag(a) @ a, output.states).real
print(f"Simulated n_th_ss = {ada[-1]:.2f}")
plt.plot(tsave / us, ada, label="a")
plt.hlines([n_th_exp], *plt.xlim(), ls="--", color="k", label="n_th")
plt.legend()
