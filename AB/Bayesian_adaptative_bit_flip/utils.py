from functools import cache, partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.typing import ArrayLike

jax.config.update("jax_enable_x64", True)
KEY = jax.random.PRNGKey(0)
s, ms, us = 1, 1e-3, 1e-6
THRESHOLD = 1e-8


@jax.jit
def probability(t: ArrayLike, theta: ArrayLike) -> jax.Array:
    gamma, C0 = theta[..., 0], theta[..., 1]
    p = (1 + C0 * jnp.exp(-gamma * t)) / 2
    return p


@jax.jit
def measure(t: ArrayLike, theta: ArrayLike, N: int, KEY: ArrayLike) -> jax.Array:
    t, theta = jnp.asarray(t), jnp.asarray(theta)
    p = probability(t, theta)
    KEY, subkey = jax.random.split(KEY)
    return jax.random.binomial(subkey, N, p)


@cache
@jax.jit
def binomial_coefficient(k: int, N: int) -> jax.Array:
    return jnp.exp(
        jax.scipy.special.gammaln(N + 1)
        - jax.scipy.special.gammaln(k + 1)
        - jax.scipy.special.gammaln(N - k + 1)
    )


@jax.jit
def p_y_theta_xi(k: ArrayLike, t: ArrayLike, theta: ArrayLike, N: int) -> jax.Array:
    t, theta = jnp.asarray(t), jnp.asarray(theta)
    p = probability(t, theta)
    return binomial_coefficient(k, N) * p**k * (1 - p) ** (N - k)


mapped_theta_p_y_theta_xi = jax.jit(
    jax.vmap(p_y_theta_xi, in_axes=(None, None, 0, None))
)


@jax.jit
def update_ptheta(
    theta: ArrayLike, ptheta: ArrayLike, k: int, t: float, N: int
) -> jax.Array:
    p_y_theta_xi_v = mapped_theta_p_y_theta_xi(k, t, theta, N)
    res = ptheta * p_y_theta_xi_v
    return res / res.sum()


def boundary_refine(
    ptheta: ArrayLike, thr: float = THRESHOLD
) -> tuple[int, int, int, int]:
    gamma_thr = jnp.any(ptheta > thr, axis=1)
    ind_gamma_min = jnp.argmax(gamma_thr)
    ind_gamma_max = len(gamma_thr) - jnp.argmax(gamma_thr[::-1]) - 1

    C0_thr = jnp.any(ptheta > thr, axis=0)
    ind_C0_min = jnp.argmax(C0_thr)
    ind_C0_max = len(C0_thr) - jnp.argmax(C0_thr[::-1]) - 1
    return ind_gamma_min, ind_gamma_max, ind_C0_min, ind_C0_max


def refine_theta(theta: ArrayLike, ptheta: ArrayLike) -> tuple[jax.Array, jax.Array]:
    ig_min, ig_max, ic_min, ic_max = boundary_refine(ptheta)

    new_gamma = jnp.geomspace(theta[ig_min, 0, 0], theta[ig_max, 0, 0], theta.shape[0])
    new_C0 = jnp.geomspace(theta[0, ic_min, 1], theta[0, ic_max, 1], theta.shape[1])
    new_theta = jnp.array(jnp.meshgrid(new_gamma, new_C0, indexing='ij'))
    new_theta = jnp.moveaxis(new_theta, 0, -1)

    interp = jax.scipy.interpolate.RegularGridInterpolator(
        (
            theta[ig_min : ig_max + 1, 0, 0],
            theta[0, ic_min : ic_max + 1, 1],
        ),
        ptheta[ig_min : ig_max + 1, ic_min : ic_max + 1],
        method='linear',
        bounds_error=False,
        fill_value=None,
    )
    new_ptheta = interp(tuple(jnp.meshgrid(new_gamma, new_C0, indexing='ij')))
    return new_theta, new_ptheta


def check_and_refine_theta(
    theta: ArrayLike, ptheta: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    ig_min, ig_max, ic_min, ic_max = boundary_refine(ptheta)
    if (ig_max - ig_min + 1) * (ic_max - ic_min + 1) > theta.size / 2:
        return theta, ptheta
    return refine_theta(theta, ptheta)


def plot_ptheta(ptheta: ArrayLike, theta: ArrayLike, theta_target) -> None:
    fig = plt.figure(figsize=(5, 5))
    grid = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
    ax = grid.subplots()
    ax[0, 0].pcolormesh(theta[:, 0, 0], theta[0, :, 1], ptheta.T, vmin=0.0)
    ax[0, 0].set_xlabel("gamma")
    ax[0, 0].set_ylabel("C0")
    ax[0, 0].set_xscale("log")
    ax[0, 0].plot(theta_target[0], theta_target[1], "ro")

    ax[0, 1].plot(ptheta.sum(axis=0), theta[0, :, 1])
    ax[0, 1].set_xlabel("p")
    ax[0, 1].set_ylabel("C0")
    ax[0, 1].grid()
    ax[0, 1].axhline(theta_target[1], color="r")

    ax[1, 0].plot(theta[:, 0, 0], ptheta.sum(axis=1))
    ax[1, 0].set_xlabel("gamma")
    ax[1, 0].set_ylabel("p")
    ax[1, 0].grid()
    ax[1, 0].set_xscale("log")
    ax[1, 0].axvline(theta_target[0], color="r")

    fig.delaxes(ax[1, 1])

    fig.tight_layout()
    plt.show()


@jax.jit
def _eig_y_xi(
    k: int,
    t: float,
    theta: ArrayLike,
    ptheta: ArrayLike,
    N: int,
    thr: float = THRESHOLD,
) -> float:
    p_y_theta_xi_v = mapped_theta_p_y_theta_xi(k, t, theta, N)
    p_y_xi_v = jnp.sum(p_y_theta_xi_v * ptheta)
    eig_v = jnp.where(
        p_y_theta_xi_v > thr,
        jnp.log(p_y_theta_xi_v / p_y_xi_v) * ptheta * p_y_theta_xi_v,
        0,
    )
    return eig_v.sum()


mapped_k_eig_y_xi = jax.jit(jax.vmap(_eig_y_xi, in_axes=(0, None, None, None, None)))


@partial(jax.jit, static_argnums=(3,))
def eig(t: float, theta: ArrayLike, ptheta: ArrayLike, N: int):
    eig_v = mapped_k_eig_y_xi(jnp.arange(N + 1), t, theta, ptheta, N)
    return eig_v.sum()


mapped_t_eig = jax.vmap(eig, in_axes=(0, None, None, None))


@partial(jax.jit, static_argnums=(3,))
def optimal_t(ts: ArrayLike, theta: ArrayLike, ptheta: ArrayLike, N: int):
    eig_v = mapped_t_eig(ts, theta, ptheta, N) / jnp.log(ts / ts[0] * jnp.exp(1.0))
    return ts[jnp.argmax(eig_v)]
