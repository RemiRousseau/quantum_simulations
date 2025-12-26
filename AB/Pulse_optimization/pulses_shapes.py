import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


def random_normal_pulse(
    tpulse: ArrayLike, mean: float = 0, sigma: float = 1, clip: float = 1
) -> ArrayLike:
    random = jnp.array(np.random.normal(loc=mean, scale=sigma, size=len(tpulse) - 1))
    return jnp.clip(random, -clip, clip)


def square_pulse(tpulse: ArrayLike, *params: float) -> ArrayLike:
    period, amp = params
    y = 2 * ((tpulse % period) < period / 2) - 1
    return amp * y[:-1]


def triangle_pulse(tpulse: ArrayLike, *params: float) -> ArrayLike:
    period, amp = params
    y = abs((tpulse + period / 4) % (period) * 4 / period - 2) - 1
    return amp * y[:-1]


def random_fft_weights(n_weights: int, mean: float, sigma: float) -> ArrayLike:
    return jnp.array(np.random.normal(loc=mean, scale=sigma, size=2 * n_weights))


def pulse_from_fft(tpulse: ArrayLike, fft_weights: ArrayLike) -> ArrayLike:
    n_time = len(tpulse) - 1
    n_weights = len(fft_weights) // 2
    c_fft_weights = fft_weights[:n_weights] + 1j * fft_weights[n_weights:]
    fft_full = jnp.zeros(n_time, dtype=complex)
    fft_full = fft_full.at[1 : n_weights + 1].set(c_fft_weights)
    fft_full = fft_full.at[-n_weights:-1].set(jnp.conj(c_fft_weights[-1:0:-1]))
    return jnp.fft.ifft(fft_full).real


def pulse_from_fft_free_fundamental(tpulse: ArrayLike, params: ArrayLike) -> ArrayLike:
    fund_period, fft_weights = params[0], params[1:]

    n_weights = len(fft_weights) // 2
    c_fft_weights = fft_weights[:n_weights] + 1j * fft_weights[n_weights:]
    function_values = jnp.zeros_like(tpulse[1:], dtype=jnp.complex64)
    for n in range(n_weights):
        function_values += c_fft_weights[n] * jnp.exp(
            2j * jnp.pi * (n + 1) * tpulse[1:] / fund_period
        )
        function_values += jnp.conjugate(c_fft_weights[n]) * jnp.exp(
            -2j * jnp.pi * (n + 1) * tpulse[1:] / fund_period
        )
    return function_values.real / 2
