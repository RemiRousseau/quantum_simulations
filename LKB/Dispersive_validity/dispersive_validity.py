# %%
import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# %%
MHz = 2 * jnp.pi
kHz = MHz * 1e-3
Hz = kHz * 1e-3
us = 1


# %%
def rotating_frame(
    N: int,
    wm: float,
    wq: float,
    g: float,
    kappa_1: float,
    kappa_2: float,
    kappa_mem: float,
    nth: float,
    alpha0: float,
    t_max: float,
    Nt: int,
    p0: float = 0.5,
    save_states: bool = False,
):
    sp = dq.tensor(dq.sigmap(), dq.eye(N))
    sm = dq.tensor(dq.sigmam(), dq.eye(N))
    sx = dq.tensor(dq.sigmax(), dq.eye(N))
    sy = dq.tensor(dq.sigmay(), dq.eye(N))
    sz = dq.tensor(dq.sigmaz(), dq.eye(N))
    a = dq.tensor(dq.eye(2), dq.destroy(N))

    H = g * dq.modulated(lambda t: jnp.exp(-1j * (wm - wq) * t), sp @ a)
    H += g * dq.modulated(lambda t: jnp.exp(1j * (wm - wq) * t), sm @ dq.dag(a))
    H += g * dq.modulated(lambda t: jnp.exp(1j * (wm + wq) * t), sp @ dq.dag(a))
    H += g * dq.modulated(lambda t: jnp.exp(-1j * (wm + wq) * t), sm @ a)

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

    ket_qb = dq.unit(p0 * dq.fock(2, 0) + (1 - p0) * 1j * dq.fock(2, 1))
    if len(c_ops) > 0:
        rho_qb = dq.todm(ket_qb)
        if nth == 0:
            rho_mem = dq.fock_dm(N, 0)
        else:
            n = jnp.arange(N)
            rho_mem = dq.unit(
                jnp.diag((1.0 + nth) ** (-1.0) * (nth / (1.0 + nth)) ** (n))
            )
        if alpha0 != 0:
            ds = dq.displace(N, alpha0)
            rho_mem = ds @ rho_mem @ dq.dag(ds)
        rho_0 = dq.tensor(rho_qb, rho_mem)
    else:
        ket_0 = dq.tensor(ket_qb, dq.coherent(N, alpha0))
    exp_ops = [sx, sy, sz, a, dq.dag(a) @ a]

    if len(c_ops) == 0:
        output = dq.sesolve(
            H,
            ket_0,
            tsave,
            exp_ops=exp_ops,
            options=dq.Options(save_states=save_states),
        )
    else:
        output = dq.mesolve(
            H,
            c_ops,
            rho_0,
            tsave,
            exp_ops=exp_ops,
            options=dq.Options(save_states=save_states),
        )
    return output


# %%
N = 100
wm = 4.4 * MHz
wq = wm + jnp.linspace(-1, 1, 201) * 1 * MHz
g = 20 * kHz
kappa_1 = 0  # 1/(100*us)
kappa_2 = 0  # 1/(100*us)
kappa_mem = 0  # 24*Hz
nth = 0
alpha0 = jnp.sqrt(50)
t_max = 100 * us
Nt = 1001
p0 = 0.5
save_states = True

output = jax.vmap(rotating_frame, in_axes=(None, None, 0) + (None,) * 10)(
    N, wm, wq, g, kappa_1, kappa_2, kappa_mem, nth, alpha0, t_max, Nt, p0, save_states
)

# %%
output.expects.shape
# %%
fig, ax = plt.subplots(3, figsize=(4, 12))
for ind in range(3):
    ax[ind].imshow(
        output.expects[:, ind].real,
        aspect="auto",
        extent=[0, t_max / us, wq[0] / MHz, wq[-1] / MHz],
        interpolation="none",
        origin="lower",
    )


# %%
