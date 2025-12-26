# %%
import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt

# %%
MHz = 2 * jnp.pi
kHz = 1e-3 * MHz
GHz = 1e3 * MHz

us = 1
ns = 1e-3 * us

# %%
Ns = 10, 2
omega_a = 1 * GHz
omega_b = 7 * GHz

kappa_a = 10 * kHz
kappa_b = 10 * MHz

varphi_a = 0.1
varphi_b = 0.3

epsilon_b = 0 * MHz
epsilon_2 = 1e-3 * MHz
epsilon_l = 0 * MHz

rho0 = dq.coherent(Ns, (0, 0))
t_max = 1 * us
tsave = jnp.linspace(0, t_max, 1000)

# %%
a, b = dq.destroy(*Ns)
H = omega_a * a.dag() @ a + omega_b * b.dag() @ b

f = lambda t: epsilon_b * jnp.cos(omega_b * t)
H += dq.modulated(f, b + b.dag())

phi = varphi_a * (a + a.dag()) + varphi_b * (b + b.dag())
f = lambda t: epsilon_2 * jnp.cos((omega_b - 2 * omega_a) * t) + epsilon_l * jnp.cos(
    omega_b * t
)
H += dq.modulated(f, dq.cosm(phi))

# %%
output = dq.mesolve(H, [jnp.sqrt(kappa_a) * a, jnp.sqrt(kappa_b) * b], rho0, tsave)
# %%

a_exp = dq.expect(a, output.states)
b_exp = dq.expect(b, output.states)

plt.figure()
plt.plot(tsave, a_exp, label="a")
plt.plot(tsave, b_exp, label="b")
plt.xlabel("Time (s)")
plt.ylabel("Expectation values")
plt.legend()
plt.show()
# %%
