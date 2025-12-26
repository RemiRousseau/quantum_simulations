import functools

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

s, ms, us = 1, 1e-3, 1e-6


class AdaptativeBayesian:
    key: jax.random.PRNGKey
    tcycle: float
    tmax: float
    N_gamma: int
    N_C0: int
    N_min: int
    N_max: int

    gammas: jax.Array
    C0s: jax.Array

    xis: jax.Array
    ys: jax.Array

    def __init__(
        self,
        tcycle: float,
        tmax: float,
        N_gamma: int,
        N_C0: int,
        N_min: int,
        N_max: int,
    ):
        self.key = jax.random.key(0)
        self.tcycle = tcycle
        self.tmax = tmax
        self.N_gamma = N_gamma
        self.N_C0 = N_C0
        self.N_min = N_min
        self.N_max = N_max

        gamma_max = 2 / tcycle
        gamma_min = 1 / tmax / 5

        self.key, subkey = jax.random.split(self.key)
        uni = jax.random.uniform(subkey, (N_gamma,))
        self.gammas = gamma_min * jnp.exp(uni * jnp.log(gamma_max / gamma_min))

        self.key, subkey = jax.random.split(self.key)
        self.C0s = jax.random.uniform(subkey, (N_C0,))

    @jax.jit
    def probability(self, t: ArrayLike, theta: ArrayLike):
        gamma, C0 = theta[..., 0], theta[..., 1]
        return (1 + C0 * jnp.exp(-gamma * t)) / 2

    @functools.cache
    @jax.jit
    def binomial_coefficient(self, k: int, N: int) -> jax.Array:
        return jnp.exp(
            jax.scipy.special.gammaln(N + 1)
            - jax.scipy.special.gammaln(k + 1)
            - jax.scipy.special.gammaln(N - k + 1)
        )

    @jax.jit
    def p_y_theta_xi(
        self, k: ArrayLike, t: ArrayLike, theta: ArrayLike, N: int
    ) -> jax.Array:
        t, theta = jnp.asarray(t), jnp.asarray(theta)
        p = self.probability(t, theta)
        return self.binomial_coefficient(k, N) * p**k * (1 - p) ** (N - k)
