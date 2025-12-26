import dynamiqs as dq
import GPUtil
import jax
import jax.numpy as jnp


def set_free_gpu():
    gpus = GPUtil.getGPUs()
    gpu = min(gpus[::-1], key=lambda x: x.load)
    device = jax.devices("gpu")[gpu.id]
    jax.config.update("jax_default_device", device)
    print(f"Default device: GPU {gpu.id} ({gpu.name}) load={gpu.load*100}%")


def lab_frame(
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
    H = g * (sp + sm) @ (a + dq.dag(a))
    H += wq * sz / 2
    H += wm * dq.dag(a) @ a

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

    rho_qb = dq.todm(dq.unit(p0 * dq.fock(2, 0) + (1 - p0) * dq.fock(2, 1)))
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
    return dq.mesolve(
        H,
        c_ops,
        rho_0,
        tsave,
        exp_ops=exp_ops,
        options=dq.Options(save_states=save_states),
    )


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

    rho_qb = dq.todm(dq.unit(p0 * dq.fock(2, 0) + (1 - p0) * dq.fock(2, 1)))
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
    return dq.mesolve(
        H,
        c_ops,
        rho_0,
        tsave,
        exp_ops=exp_ops,
        options=dq.Options(save_states=save_states),
    )


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
    rho_qb = dq.todm(dq.unit(p0 * dq.fock(2, 0) + (1 - p0) * dq.fock(2, 1)))
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
