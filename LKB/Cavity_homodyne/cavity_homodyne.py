# %%
import dynamiqs as dq
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

    rho_qb = dq.unit((1 - p0) * dq.fock_dm(2, 0) + p0 * 1j * dq.fock_dm(2, 1))
    if nth == 0:
        rho_mem = dq.fock_dm(N, 0)
    else:
        n = jnp.arange(N)
        rho_mem = dq.unit(jnp.diag((1.0 + nth) ** (-1.0) * (nth / (1.0 + nth)) ** (n)))
    if alpha0 != 0:
        ds = dq.displace(N, alpha0)
        rho_mem = ds @ rho_mem @ dq.dag(ds)
    rho_0 = dq.tensor(rho_qb, rho_mem)
    exp_ops = [sx, sy, sz, a, dq.dag(a) @ a]

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
N = 20
wm = 4.4 * MHz
wq = 2.4 * MHz
g = 0 * kHz
kappa_1 = 1 / (100 * us)
kappa_2 = 1 / (100 * us)
kappa_mem = 24 * Hz
nth = 0
alpha0 = jnp.sqrt(5)
t_max = 10 * us
Nt = 1001
p0 = 0
save_states = True


# %%
p0s = jnp.array([p0, 1 - p0])
output = jax.vmap(rotating_frame, in_axes=(None,) * 11 + (0, None))(
    N, wm, wq, g, kappa_1, kappa_2, kappa_mem, nth, alpha0, t_max, Nt, p0s, save_states
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
