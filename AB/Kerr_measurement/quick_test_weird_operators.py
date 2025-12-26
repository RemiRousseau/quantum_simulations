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
Na, Nb = 30, 10
nbar = 10

delta = 0 * MHz
K_4 = 0 * kHz
K_6 = 0 * kHz
t_max = 2 * us
kappa_b = 20 * MHz

g = 0.2 * MHz
m, n = 4, 1

a, b = dq.destroy(Na, Nb)
H = g * dq.powm(dq.dag(a), m) @ dq.powm(b, n)
H += dq.dag(H)
H += delta * dq.dag(a) @ a
H -= K_4 / 2 * dq.dag(a) @ dq.dag(a) @ a @ a
H -= K_6 / 6 * dq.dag(a) @ dq.dag(a) @ dq.dag(a) @ a @ a @ a

Lb = jnp.sqrt(kappa_b) * b

psi0 = dq.coherent((Na, Nb), (jnp.sqrt(nbar), 0))
tsave = jnp.linspace(0, t_max, 1001)
output = dq.mesolve(H, [Lb], psi0, tsave, exp_ops=[a, b])
# %%
plt.plot(tsave, output.expects[0].real)
plt.plot(tsave, output.expects[0].imag)
plt.show()
plt.plot(tsave, output.expects[1].real)
plt.plot(tsave, output.expects[1].imag)
# %%
