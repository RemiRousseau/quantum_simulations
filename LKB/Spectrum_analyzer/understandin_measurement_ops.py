# %%
import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt

N = 100
nth = 0.1
# alpha = 2
theta = 0.1
vmax = 2 / jnp.pi
xmax = 1

a = dq.destroy(N)
x = 1 / 2 * (a + a.dag())
p = 1j / 2 * (a.dag() - a)

# cat = dq.unit(dq.coherent(N, alpha) + dq.coherent(N, -alpha))
ns = jnp.arange(N)
vls = (nth / (1 + nth)) ** ns / (1 + nth)
rho = dq.asqarray(jnp.diag(vls))
# rho = dq.coherent_dm(N, jnp.sqrt(nth))

dq.plot.wigner(rho, vmax=vmax, xmax=xmax)
print(dq.expect(p @ p, rho))
plt.axvline(0)
plt.axhline(0)
plt.show()
fig, ax = plt.subplots(1, 2)
rho_p = dq.unit(rho + theta * a @ rho + theta * rho @ a.dag())
print(dq.expect(p @ p, rho_p))
dq.plot.wigner(rho_p, ax=ax[0], vmax=vmax, colorbar=False, xmax=xmax)
ax[0].axvline(0)
ax[0].axhline(0)
ax[0].axvline(dq.expect(x, rho_p).real, c="r")
ax[0].axvline(theta * nth, ls="--", c="k")

rho_m = dq.unit(rho - theta * a @ rho - theta * rho @ a.dag())
print(dq.expect(p @ p, rho_m))
dq.plot.wigner(rho_m, ax=ax[1], vmax=vmax, colorbar=False, xmax=xmax)
ax[1].axvline(0)
ax[1].axhline(0)
ax[1].axvline(dq.expect(x, rho_m).real, c="r")
ax[1].axvline(-theta * nth, ls="--", c="k")
plt.show()

fig, ax = plt.subplots(1, 2)
rho_p = dq.unit(rho + theta * a.dag() @ rho + theta * rho @ a)
print(dq.expect(p @ p, rho_p))
dq.plot.wigner(rho_p, ax=ax[0], vmax=vmax, colorbar=False, xmax=xmax)
ax[0].axvline(0)
ax[0].axhline(0)
ax[0].axvline(dq.expect(x, rho_p).real, c="r")
ax[0].axvline(theta * (nth + 1), ls="--", c="k")

rho_m = dq.unit(rho - theta * a.dag() @ rho - theta * rho @ a)
print(dq.expect(p @ p, rho_m))
dq.plot.wigner(rho_m, ax=ax[1], vmax=vmax, colorbar=False, xmax=xmax)
ax[1].axvline(0)
ax[1].axhline(0)
ax[1].axvline(dq.expect(x, rho_m).real, c="r")
ax[1].axvline(-theta * (nth + 1), ls="--", c="k")
plt.show()
# %%
