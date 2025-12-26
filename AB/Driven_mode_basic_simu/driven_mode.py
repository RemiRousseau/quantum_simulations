# %%
import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt

# %%
MHz = 2 * jnp.pi
GHz = MHz * 1e3
kHz = MHz * 1e-3
us = 1
ns = 1e-3

# %%
N = 20
a = dq.destroy(N)
w_a = 7 * GHz
w_p = 2 * GHz
g1 = 1000 * kHz
kappa = 20 * MHz
Ej = 10 * GHz

t_max = 100 * ns

rho0 = dq.coherent(N, 2)

a_inf = -1j * g1 / (2j * (w_a - w_p) + kappa)
print(jnp.abs(a_inf))

# %%
H = w_a * dq.dag(a) @ a
H += g1 * dq.modulated(lambda t: jnp.cos(w_p * t), a)
H += g1 * dq.modulated(lambda t: jnp.cos(w_p * t), dq.dag(a))
H += Ej * dq.cos
L = jnp.sqrt(kappa) * a
tsave = jnp.linspace(0, t_max, 10000)
output = dq.mesolve(H, [L], rho0, tsave)

# %%
exp_a = dq.expect(a, output.states)
inds = 0, None
# plt.plot(output.tsave[inds[0] : inds[1]], exp_a[inds[0] : inds[1]].real)
# plt.plot(output.tsave[inds[0] : inds[1]], exp_a[inds[0] : inds[1]].imag)
plt.plot(output.tsave[inds[0] : inds[1]], jnp.abs(exp_a[inds[0] : inds[1]]))
plt.hlines([a_inf.real, a_inf.imag], 0, t_max, color="k", linestyles="dashed")
plt.vlines([5 / (w_a - w_p), 5 / kappa], *plt.ylim(), color="k", linestyles="dashed")
plt.show()
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.plot(exp_a.real, exp_a.imag, label="real")
ax.plot(a_inf.real, a_inf.imag, "or")

# %%
U_frame = dq.expm(1j * w_a * dq.dag(a) @ a * tsave[:, None, None])
state_frame = U_frame @ output.states @ dq.dag(U_frame)
dq.plot.wigner_gif(state_frame)
exp_a = dq.expect(a, state_frame)
plt.plot(output.tsave, exp_a.real)
plt.plot(output.tsave, exp_a.imag)
