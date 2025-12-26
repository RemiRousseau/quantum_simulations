# %%
import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

dq.set_precision("simple")

MHz = 2 * jnp.pi
kHz = MHz / 1e3
Hz = kHz / 1e3
us = 1
ms = 1e3

# %%
N = 20
g = 1 * kHz
t_max = 1 * us
Nt = 101
prep_fidelity = 1
N_loop = 3


def get_sxsx(N, g, n0, t_max, prep, prep_fidelity, N_loop):
    a = dq.destroy(N)
    sm = dq.sigmam()
    sx = dq.sigmax()
    sy = dq.sigmay()
    sz = dq.sigmaz()
    H1 = g * (dq.tensor(a, dq.dag(sm), dq.eye(2)) + dq.tensor(dq.dag(a), sm, dq.eye(2)))
    H2 = g * (dq.tensor(a, dq.eye(2), dq.dag(sm)) + dq.tensor(dq.dag(a), dq.eye(2), sm))
    times = jnp.linspace(0, t_max, Nt)

    ind = 1 if prep == "g" else 0
    p = (prep_fidelity + 1) / 2
    rho_qb = p * dq.fock_dm(2, ind) + (1 - p) * dq.fock_dm(2, 1 - ind)
    rho0 = dq.tensor(dq.fock_dm(N, n0), rho_qb, rho_qb)
    o1 = dq.mesolve(
        H1, [0 * dq.eye(4 * N)], rho0, times, options=dq.Options(progress_meter=None)
    )
    rho1 = o1.states[-1]
    for _ in range(N_loop):
        o2 = dq.mesolve(
            H2,
            [0 * dq.eye(4 * N)],
            rho1,
            times,
            options=dq.Options(progress_meter=None),
        )
        rho1 = o2.states[-1]
    return dq.expect(dq.tensor(dq.eye(N), sx, sx), rho1).real


# %%
############## Simulation with variable n0 ################
n0s = jnp.arange(N - 5)
sxsxs_g = []
sxsxs_e = []
for n0 in tqdm(n0s):
    sxsxs_g.append(get_sxsx(N, g, n0, t_max, "g", prep_fidelity, N_loop))
    sxsxs_e.append(get_sxsx(N, g, n0, t_max, "e", prep_fidelity, N_loop))

# %%
theta = g * t_max
print("theta = ", theta)


def ana_formula(p, theta, n):
    return 2 * p * jnp.sin(theta * jnp.sqrt(n)) ** 2 * jnp.cos(theta * jnp.sqrt(n)) * (
        p * jnp.cos(theta * jnp.where(n > 0, jnp.sqrt(n - 1), 0))
        - (1 - p) * jnp.cos(theta * jnp.sqrt(n + 1))
    ) + 2 * (1 - p) * jnp.sin(theta * jnp.sqrt(n + 1)) ** 2 * jnp.cos(
        theta * jnp.sqrt(n + 1)
    ) * ((1 - p) * jnp.cos(theta * jnp.sqrt(n + 2)) - p * jnp.cos(theta * jnp.sqrt(n)))


fig, ax = plt.subplots()
ax.plot(n0s, sxsxs_g, label="g")
ana_val = ana_formula((prep_fidelity + 1) / 2, theta, n0s)
ax.plot(n0s, ana_val, "k--")
# ax.plot(2*theta**2*n0s, "r:")

ax.plot(n0s, sxsxs_e, label="e")
ana_val = ana_formula(1 - (prep_fidelity + 1) / 2, theta, n0s)
ax.plot(n0s, ana_val, "k--")
# ax.plot(2*theta**2*(n0s+1), "r:")
plt.legend()
plt.grid()


# %%
############## Plotting analytical formulas ################
p = jnp.linspace(0, 1, 101)
n = 5
theta = 1
plt.plot(
    p,
    2 * p * theta**2 * n * (2 * p - 1) + 2 * (1 - p) * theta**2 * (n + 1) * (1 - 2 * p),
)
plt.grid()


# %%
def asymetry(prep_fidel, theta, n):
    p = (1 + prep_fidel) / 2
    sg = 2 * p * theta**2 * n * (2 * p - 1) + 2 * (1 - p) * theta**2 * (n + 1) * (
        1 - 2 * p
    )
    p = 1 - p
    se = 2 * p * theta**2 * n * (2 * p - 1) + 2 * (1 - p) * theta**2 * (n + 1) * (
        1 - 2 * p
    )
    return sg / (se - sg)


# %%
prep_fidelitys = jnp.linspace(0, 1, 101)
n = 50
theta = 1
plt.plot(prep_fidelitys, asymetry(prep_fidelitys, theta, n))
plt.plot(prep_fidelitys, prep_fidelitys * n)
plt.grid()


# %%
############## Simulation with varibale n_loop ################
def get_sxsx(N, g, n0, t_max, prep, prep_fidelity, N_loop):
    a = dq.destroy(N)
    sm = dq.sigmam()
    sx = dq.sigmax()
    sy = dq.sigmay()
    sz = dq.sigmaz()
    H1 = g * (dq.tensor(a, dq.dag(sm), dq.eye(2)) + dq.tensor(dq.dag(a), sm, dq.eye(2)))
    H2 = g * (dq.tensor(a, dq.eye(2), dq.dag(sm)) + dq.tensor(dq.dag(a), dq.eye(2), sm))
    times = jnp.linspace(0, t_max, Nt)

    ind = 1 if prep == "g" else 0
    p = (prep_fidelity + 1) / 2
    rho_qb = p * dq.fock_dm(2, ind) + (1 - p) * dq.fock_dm(2, 1 - ind)
    rho0 = dq.tensor(dq.fock_dm(N, n0), rho_qb, rho_qb)
    o1 = dq.mesolve(
        H1, [0 * dq.eye(4 * N)], rho0, times, options=dq.Options(progress_meter=None)
    )
    rho1 = o1.states[-1]
    vals = []
    for _ in range(N_loop):
        o2 = dq.mesolve(
            H2,
            [0 * dq.eye(4 * N)],
            rho1,
            times,
            options=dq.Options(progress_meter=None),
        )
        rho1 = o2.states[-1]
        vals.append(dq.expect(dq.tensor(dq.eye(N), sx, sx), rho1).real)
        rho_mem = dq.ptrace(rho1, (0, 1), (N, 2, 2))
        rho1 = dq.tensor(rho_mem, rho_qb)
    return vals


# %%
N = 20
g = 2 * kHz
t_max = 1 * us
Nt = 101
prep_fidelity = 0.7
N_loop = 3
n0 = 10
N_loop = 100

sxsxs_g = get_sxsx(N, g, n0, t_max, "g", prep_fidelity, N_loop)
sxsxs_e = get_sxsx(N, g, n0, t_max, "e", prep_fidelity, N_loop)
# %%
fig, ax = plt.subplots()
ax.plot(sxsxs_g, label="g")
# ax.plot(sxsxs_e, label="e")

############### Adding detunings #################

# %%
