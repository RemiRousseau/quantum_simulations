# %%
import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils import set_free_gpu

set_free_gpu()

# %%
MHz = 2 * jnp.pi
kHz = 1e-3 * MHz
Hz = 1e-6 * MHz
us = 1


# %%
def rotating_displaced_frame(
    N: int,
    wm: float,
    wq: float,
    g: float,
    kappa_1: float,
    kappa_2: float,
    kappa_mem: float,
    nth: float,
    alpha0: float,
    eps_d: float,
    wd: float,
    t_max: float,
    Nt: int,
    save_states: bool = False,
):
    sp = dq.tensor(dq.sigmap(), dq.eye(N))
    sm = dq.tensor(dq.sigmam(), dq.eye(N))
    sx = dq.tensor(dq.sigmax(), dq.eye(N))
    sy = dq.tensor(dq.sigmay(), dq.eye(N))
    sz = dq.tensor(dq.sigmaz(), dq.eye(N))
    a = dq.tensor(dq.eye(2), dq.destroy(N))

    a = a + alpha0 * dq.tensor(dq.eye(2), dq.eye(N))

    H = g * dq.modulated(lambda t: jnp.exp(-1j * (wm - wq) * t), sp @ a)
    H += g * dq.modulated(lambda t: jnp.exp(1j * (wm - wq) * t), sm @ dq.dag(a))
    H += g * dq.modulated(lambda t: jnp.exp(1j * (wm + wq) * t), sp @ dq.dag(a))
    H += g * dq.modulated(lambda t: jnp.exp(-1j * (wm + wq) * t), sm @ a)
    H += eps_d * dq.modulated(lambda t: jnp.cos(wd * t) * jnp.exp(1j * wq * t), sp)
    H += eps_d * dq.modulated(lambda t: jnp.cos(wd * t) * jnp.exp(-1j * wq * t), sm)

    c_ops = []
    if kappa_1 > 0:
        c_ops.append(jnp.sqrt(kappa_1) * sm)
        c_ops.append(jnp.sqrt(kappa_1) * sp)
    if kappa_2 > 0:
        kappa_phi = kappa_2 - kappa_1 / 2
        c_ops.append(jnp.sqrt(kappa_phi / 2) * sz)
    if kappa_mem > 0:
        c_ops.append(jnp.sqrt(kappa_mem * (1 + nth)) * a)
    if kappa_mem * nth > 0:
        c_ops.append(jnp.sqrt(kappa_mem * nth) * dq.dag(a))

    tsave = jnp.linspace(0, t_max, Nt)
    rho_qb = dq.fock_dm(2, 1)
    if nth == 0:
        rho_mem = dq.fock_dm(N, 0)
    else:
        n = jnp.arange(N)
        rho_mem = dq.unit(jnp.diag((1.0 + nth) ** (-1.0) * (nth / (1.0 + nth)) ** (n)))
    rho_0 = dq.tensor(rho_qb, rho_mem)

    exp_ops = [sx, sy, sz, a, dq.dag(a) @ a]
    return dq.mesolve(
        H,
        c_ops,
        rho_0,
        tsave,
        exp_ops=exp_ops,
        options=dq.Options(save_states=save_states),
    )


# %%
N = 10
wm = 4.4 * MHz
wq = 2.4 * MHz
g = 1 * kHz
kappa_1 = 1 / (30 * us)
kappa_2 = 1 / (10 * us)
kappa_mem = 24 * Hz
nth = 0
alpha0 = jnp.sqrt(50e4)
eps_d = 0.05 * MHz
wd = wq + jnp.linspace(-1, 1, 501) * 0.5 * MHz
t_max = 10 * us
Nt = 1001
save_states = False

result = jax.vmap(rotating_displaced_frame, in_axes=(None,) * 10 + (0,) + (None,) * 3)(
    N,
    wm,
    wq,
    g,
    kappa_1,
    kappa_2,
    kappa_mem,
    nth,∏
    alpha0,
    eps_d,
    wd,
    t_max,
    Nt,
    save_states,
)
# %%
plt.plot(result.expects[:, 2, -1].real)
plt.show()
plt.imshow(
    result.expects[:, 2].real,
    aspect="auto",
    extent=[0, t_max / us, wd[0] / MHz, wd[-1] / MHz],
)
# %%
