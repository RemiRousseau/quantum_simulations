# %%
import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt

# %%
MHz = 2 * jnp.pi
kHz = 1e-3 * MHz
us = 1
ns = 1e-3 * us

# %%
N = 200
nbars = jnp.linspace(0, 100, 101)[1:]
delta = 1 * MHz
K_4 = 1 * kHz
K_6 = 20 * kHz
t_max = 5 * ns

a = dq.destroy(N)
H = (
    delta * dq.dag(a) @ a
    - K_4 / 2 * dq.dag(a) @ dq.dag(a) @ a @ a
    - K_6 / 6 * dq.dag(a) @ dq.dag(a) @ dq.dag(a) @ a @ a @ a
)
psi0 = dq.coherent(N, jnp.sqrt(nbars))
tsave = jnp.linspace(0, t_max, 1001)
output = dq.sesolve(H, psi0, tsave, exp_ops=[a])
# %%
unwrapped_phases = jnp.unwrap(jnp.angle(output.expects[:, 0]), axis=-1)
plt.plot(tsave, unwrapped_phases.T)
res = jnp.polyfit(tsave, unwrapped_phases.T, 1)
# %%
plt.plot(nbars, res[0])
plt.plot(nbars, -(delta - K_4 * nbars - K_6 / 2 * nbars**2), "--")
# %%
