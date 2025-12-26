# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils import rotating_displaced_frame, set_free_gpu

set_free_gpu()

# %%
MHz = 2 * jnp.pi
kHz = 1e-3 * MHz
Hz = 1e-6 * MHz
us = 1


# %%
N = 10
wm = 4.4 * MHz
wq = jnp.linspace(-1, 1, 101) * 1 * MHz + wm
factor = 100
g = 1 * kHz / factor
kappa_1 = 1 / (30 * us)
kappa_2 = 1 / (10 * us)
kappa_mem = 24 * Hz
nth = 0
alpha0 = 2e2 * factor
t_max = 10 * us
Nt = 101
p0 = 1

result = jax.vmap(rotating_displaced_frame, in_axes=(None, None, 0) + (None,) * 9)(
    N, wm, wq, g, kappa_1, kappa_2, kappa_mem, nth, alpha0, t_max, Nt, p0
)

plt.imshow(
    result.expects[:, 2].real,
    aspect="auto",
    extent=[0, t_max / us, wq[0] / MHz, wq[-1] / MHz],
)
# %%
