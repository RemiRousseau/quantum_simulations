# %%
import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

# %%
N = 100
alpha = 2
alpha_moon = 2 * 0.96593
lam = 1.0

catp = dq.unit(dq.coherent(N, alpha) + dq.coherent(N, -alpha))
catm = dq.unit(dq.coherent(N, alpha) - dq.coherent(N, -alpha))

a = dq.destroy(N)
L2 = (
    a @ a
    - alpha_moon**2 * dq.eye(N)
    + lam * (dq.dag(a) @ a - alpha_moon**2 * dq.eye(N))
)
Hk = dq.dag(L2) @ L2
_, eigst = jnp.linalg.eigh(Hk)
if dq.expect(dq.parity(N), eigst[:, 0].reshape(-1, 1)) > 0:
    moon_catp, moon_catm = eigst[:, :2].T.reshape(2, -1, 1)
else:
    moon_catm, moon_catp = eigst[:, :2].T.reshape(2, -1, 1)


# %%
def plot_schematics(ax: Axes, catp, catm, xmax, ymax, p_wig, beta_disp, level):
    disp = dq.displace(N, beta_disp)
    dead = dq.unit(dq.todm(catp) + dq.todm(catm))
    cat_to_show = p_wig * dq.todm(catp) + (1 - p_wig) * dq.todm(catm)

    log0, log1 = catp + catm, catp - catm
    phase_disp = jnp.abs(beta_disp) * 4 * alpha
    cat_disp = disp @ dq.unit(log0 + jnp.exp(1j * phase_disp) * log1)

    cat_dead_disp = disp @ dead @ dq.dag(disp)

    kwargs_cmap = {
        "origin": "lower",
        "interpolation": "none",
        "extent": [-xmax, xmax, -ymax, ymax],
        "vmin": -1 / jnp.pi,
        "vmax": 1 / jnp.pi,
        "cmap": "bwr",
        "aspect": "equal",
    }

    ax.set_aspect("equal")
    xvec, yvec, w = dq.wigner(cat_to_show, xmax=xmax, ymax=ymax)
    ax.imshow(w, **kwargs_cmap)

    _, _, w_dead = dq.wigner(dead, xmax=xmax, ymax=ymax)
    ctr = ax.contour(xvec, yvec, w_dead, levels=[level], colors="k", linestyles="-")

    _, _, w_disp = dq.wigner(cat_dead_disp, xmax=xmax, ymax=ymax)
    ctr_disp = ax.contour(
        xvec, yvec, w_disp, levels=[level], colors="k", linestyles="--"
    )

    w_both = jnp.logical_and(w_dead > level, w_disp > level)
    ax.contourf(
        xvec, yvec, w_both, levels=[0.5, 1.5], colors="k", alpha=0.0, hatches=["///"]
    )

    p = ctr.collections[0].get_paths()[0].vertices
    p_extr = p[jnp.argmin(p[:, 1])]

    p = ctr_disp.collections[0].get_paths()[0].vertices
    p_extr_disp = p[jnp.argmin(p[:, 1])]

    head_length = 0.2
    ax.arrow(
        *p_extr,
        0,
        np.abs(beta_disp) - head_length,
        head_width=0.1,
        head_length=head_length,
        fc="g",
        ec="g",
    )
    ax.arrow(
        *(p_extr * jnp.array([-1, 1])),
        0,
        np.abs(beta_disp) - head_length,
        head_width=0.1,
        head_length=head_length,
        fc="g",
        ec="g",
    )
    ax.arrow(
        0,
        0,
        0,
        np.abs(beta_disp) - head_length,
        head_width=0.1,
        head_length=head_length,
        fc="g",
        ec="g",
    )

    xvec, yvec, wp = dq.wigner(catp, xmax=xmax, ymax=ymax)
    xvec, yvec, wd = dq.wigner(cat_disp, xmax=xmax, ymax=ymax)
    fidelity = jnp.sum(wp * wd) * (xvec[1] - xvec[0]) * (yvec[1] - yvec[0]) * jnp.pi
    print("Fidelity :", fidelity)
    print("Overlap :", jnp.sum(w_both) * (xvec[1] - xvec[0]) * (yvec[1] - yvec[0]))


fig, ax = plt.subplots(2, figsize=(6, 6))
xmax, ymax = 4, 2
level = 0.5 / jnp.pi
p_wig = 0.75
beta_dip = 2 * jnp.pi / 4 / alpha * 1j
plot_schematics(ax[0], catp, catm, xmax, ymax, p_wig, beta_dip, level)
plot_schematics(ax[1], moon_catp, moon_catm, xmax, ymax, p_wig, beta_dip, level)

# %%
