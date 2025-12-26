# %%
import dynamiqs as dq

# import GPUtil
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# def set_free_gpu():
#     gpus = GPUtil.getGPUs()
#     gpu = min(gpus[::-1], key=lambda x: x.load)
#     device = jax.devices("gpu")[gpu.id]
#     jax.config.update("jax_default_device", device)
#     print(f"Default device: GPU {gpu.id} ({gpu.name}) load={gpu.load*100}%")
# set_free_gpu()

MHz = 2 * jnp.pi
kHz = MHz * 1e-3
Hz = kHz * 1e-3
us = 1
ms = 1e3


# %%
def rotating_displaced_frame(
    N: int,
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

    a = a + alpha0 * dq.tensor(dq.eye(2), dq.eye(N))

    H = g * (sp @ a + sm @ dq.dag(a))

    c_ops = []
    if kappa_1 > 0:
        c_ops.append(jnp.sqrt(kappa_1 / 2) * sm)
        c_ops.append(jnp.sqrt(kappa_1 / 2) * sp)
    if kappa_2 > 0:
        kappa_phi = kappa_2 - kappa_1 / 2
        c_ops.append(jnp.sqrt(kappa_phi / 2) * sz)
    if kappa_mem > 0:
        c_ops.append(jnp.sqrt(kappa_mem * (1 + nth)) * a)
        c_ops.append(jnp.sqrt(kappa_mem * nth) * dq.dag(a))

    tsave = jnp.linspace(0, t_max, Nt)

    ket_qb = dq.unit(p0 * dq.fock(2, 0) + (1 - p0) * dq.fock(2, 1))
    rho_qb = dq.todm(ket_qb)
    if nth == 0:
        rho_mem = dq.fock_dm(N, 0)
    else:
        n = jnp.arange(N)
        rho_mem = dq.unit(jnp.diag((1.0 + nth) ** (-1.0) * (nth / (1.0 + nth)) ** (n)))
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
N = 100
g = 1 * kHz
kappa_1 = 1 / (30 * us)
kappa_2 = 1 / (30 * us)
kappa_mem = 0 * Hz
nth = 0
alpha0 = jnp.sqrt(10)
t_max = 200 * us
Nt = 501
p0 = jnp.array([0, 1])
save_states = True

output = jax.vmap(rotating_displaced_frame, in_axes=(None,) * 9 + (0, None))(
    N, g, kappa_1, kappa_2, kappa_mem, nth, alpha0, t_max, Nt, p0, save_states
)
# %%
plt.plot(output.tsave[0], output.expects[:, 2].T)
plt.xlabel("Time [us]")
plt.ylabel("sz")
plt.grid()
plt.show()

# %%
dq.plot.wigner_gif(dq.ptrace(output.states[0], 1, (2, N)))
plt.show()
dq.plot.wigner_gif(dq.ptrace(output.states[1], 1, (2, N)))
plt.show()

# %%
fig, ax = plt.subplots(5, figsize=(5, 10))
for i, label in enumerate(["sx", "sy", "sz", "a", "n"]):
    if i == 2:
        flipped = output.expects[:, i].T * (jnp.array([1, 1])[None, :])
        ax[i].plot(output.tsave[0], flipped, label=label)
    else:
        ax[i].plot(output.tsave[0], output.expects[:, i].T, label=label)
    ax[i].set_xlabel("Time [us]")
    ax[i].set_ylabel(label)
    ax[i].grid()
# %%
