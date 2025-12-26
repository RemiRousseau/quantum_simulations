# %%
import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from tqdm import tqdm


# %%
def get_moon(N, alpha, lam):
    a = dq.destroy(N)
    L2 = a @ a - alpha**2 * dq.eye(N) + lam * (dq.dag(a) @ a - alpha**2 * dq.eye(N))
    Hk = dq.dag(L2) @ L2
    _, eigst = jnp.linalg.eigh(Hk)
    parity = dq.parity(N)
    if dq.expect(parity, eigst[:, 0].reshape(-1, 1)).real > 0:
        cp, cm = eigst[:, :2].T
    else:
        cm, cp = eigst[:, :2].T
    return cp.reshape(-1, 1), cm.reshape(-1, 1)


def get_moon_nb(N, nbar, lam):
    a = dq.destroy(N)

    def local_cost(alpha):
        moon = get_moon(N, alpha, lam)[0]
        return (dq.expect(dq.dag(a) @ a, moon).real - nbar) ** 2

    res = minimize(
        local_cost, jnp.sqrt(nbar), bounds=((0.0, 10.0),), jac=jax.jacobian(local_cost)
    )
    return get_moon(N, res.x[0], lam), res.x[0]


def get_log(cp, cm):
    N = cp.size
    l0 = dq.unit(cp + cm)
    l1 = dq.unit(cp - cm)
    if dq.expect(dq.destroy(N), l0).real < 0:
        l0, l1 = l1, l0
    return l0, l1


# %%
(cp, cm), alpha = get_moon_nb(100, 4, 1)
l0, l1 = get_log(cp, cm)
dq.plot.wigner(l0)
# %%
N = 50
nbar = 4
lambdas = jnp.linspace(0, 0.5, 21)
l0s = []
for lam in tqdm(lambdas):
    (cp, cm), alpha = get_moon_nb(N, nbar, lam)
    # cp, cm = get_moon(N, jnp.sqrt(nbar), lam)
    l0, _ = get_log(cp, cm)
    l0s.append(l0)
l0s = jnp.array(l0s)
# %%
for l0 in l0s:
    dq.plot.wigner(l0)
    plt.show()

# %%
Nx = 201
xvec = jnp.linspace(0, 4, Nx)
yvec = jnp.linspace(-4, 4, 2 * Nx)
_, _, w = dq.wigner(l0s, xvec=xvec, yvec=yvec)
marginal = w.sum(axis=-1) * (xvec[1] - xvec[0])


# %%
def gaussian(x, sigma):
    return jnp.exp(-0.5 * (x / sigma) ** 2) / jnp.sqrt(2 * jnp.pi * sigma**2)


# %%
fig, ax = plt.subplots()
cmap = plt.get_cmap("viridis")
for ind, lam in enumerate(lambdas):
    ax.plot(yvec, marginal[ind], color=cmap(lam / lambdas[-1]), alpha=0.5)
    ax.plot(yvec, gaussian(yvec, 1 / 2 + lam / 4), color="k", linestyle="--", alpha=0.5)

# %%
fig, ax = plt.subplots()
cmap = plt.get_cmap("viridis")
for ind, lam in enumerate(lambdas):
    ax.plot(
        yvec,
        marginal[ind] - gaussian(yvec, 1 / 2 + lam / 4),
        color=cmap(lam / lambdas[-1]),
    )
# %%
fitted_width = []
for ind, lam in enumerate(lambdas):
    popt, _ = curve_fit(gaussian, yvec, marginal[ind], p0=1 / 2 + lam / 4)
    fitted_width.append(popt[0])

fig, ax = plt.subplots()
ax.plot(lambdas, fitted_width, "o")
ax.plot(lambdas, 1 / 2 + lambdas / 4)


###### Marginal x axis ######
# %%
marginal = w.sum(axis=-2) * (yvec[1] - yvec[0])


# %%
def gaussian(x, mu, sigma):
    return jnp.exp(-0.5 * ((x - mu) / sigma) ** 2) / jnp.sqrt(2 * jnp.pi * sigma**2)


# %%
fig, ax = plt.subplots()
cmap = plt.get_cmap("viridis")
for ind, lam in enumerate(lambdas):
    ax.plot(xvec, marginal[ind], color=cmap(lam / lambdas[-1]), alpha=0.5)
    ax.plot(
        xvec,
        gaussian(xvec, jnp.sqrt(nbar), 1 / 2 - lam / 4),
        color="k",
        linestyle="--",
        alpha=0.5,
    )

# %%
fig, ax = plt.subplots()
cmap = plt.get_cmap("viridis")
for ind, lam in enumerate(lambdas):
    ax.plot(
        xvec,
        marginal[ind] - gaussian(xvec, jnp.sqrt(nbar), 1 / 2 - lam / 4),
        color=cmap(lam / lambdas[-1]),
    )
# %%
fitted_width = []
for ind, lam in enumerate(lambdas):
    popt, _ = curve_fit(
        gaussian, xvec, marginal[ind], p0=(jnp.sqrt(nbar), 1 / 2 - lam / 4)
    )
    fitted_width.append(popt[1])

fig, ax = plt.subplots()
ax.plot(lambdas, fitted_width, "o")
ax.plot(lambdas, 1 / 2 - lambdas / 4)
# %%
