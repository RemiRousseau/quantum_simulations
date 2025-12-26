# %%
import os
from collections import OrderedDict

import dynamiqs as dq
import GPUtil
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils import MHz, batch_cnot_errors, kHz

dq.set_precision("simple")


def set_free_gpu():
    gpus = GPUtil.getGPUs()
    gpu = min(gpus[::-1], key=lambda x: x.load)
    device = jax.devices("gpu")[gpu.id]
    jax.config.update("jax_default_device", device)
    print(f"Default device: GPU {gpu.id} ({gpu.name}) load={gpu.load*100}%")


set_free_gpu()

# For all maps :
#   we vary lambda in [0, 0.5, 1.0]
#   the gate duration is fixed at 1/kappa_2, followed by 1/kappa_2/alpha^2 of restabilization
#   both stabilized and unstabilized gates are simulated
#   the cat size is fixed at nbar=4 photons

# %%
# Varying kappa_a, and fixing kappa_phi = K_4 = 0
max_batch_size = 100
params = {
    "N0": 100,
    "N1": 20,
    "alpha": 2,
    "kappa_2": 1 * MHz,
    "kappa_phi": 0 * kHz,
    "kerr": 0 * kHz,
    "T_gate": 1 / (1 * MHz),
    "Nt": 101,
}

lambdas = jnp.linspace(0, 1.0, 5)
kappa_a_o_kappa_2 = jnp.geomspace(1e-3, 1e-1, 11)
kappa_a = kappa_a_o_kappa_2 * params["kappa_2"]
stabilized = jnp.array([True, False])
sweep_params = OrderedDict(lam=lambdas, kappa_a=kappa_a, stabilized=stabilized)
errors, nbars, tgates = batch_cnot_errors(
    params, sweep_params, max_batch_size, cathesian_batching=True
)
# %%
fig = plt.figure(layout="constrained", figsize=(6, 5))
gs = fig.add_gridspec(2, 2, wspace=0.06, hspace=0.05)
ax = gs.subplots()
errors_name = [f"$\\epsilon_{op}^{cat}$" for op in ["Z", "X"] for cat in ["C", "T"]]
for a, data, title in zip(
    ax.flatten(), jnp.moveaxis(jnp.abs(errors), -1, 0), errors_name
):
    a.set_ylabel(title)
    a.set_xlabel(r"$\kappa_1/\kappa_2$")
    for ind_lam, color_lam in enumerate([f"C{i}" for i in range(len(lambdas))]):
        for ind_stab, maker in enumerate(["o", "v"]):
            a.plot(kappa_a_o_kappa_2, data[ind_lam, :, ind_stab], f"{color_lam}{maker}")
    if a == ax[1, 1]:
        for ind in range(len(lambdas)):
            a.plot([], [], f"C{ind}o", label=f"$\\lambda = {lambdas[ind]}$")
        a.plot([], [], "ko", label="Stab")
        a.plot([], [], "kv", label="No stab")
        a.legend()
    if a in ax[0, :]:
        a.set_yscale("log")
    if a in ax[1, :]:
        a.set_yscale("log")
    a.set_xscale("log")
    a.grid()
plt.savefig(os.getcwd() + "/figures/kappa_a_o_kappa_2.pdf", dpi=300)

# %%
# Varying K_4, and fixing kappa_phi = kappa_a = 0

max_batch_size = 100
params = {
    "N0": 100,
    "N1": 20,
    "alpha": 2,
    "kappa_2": 1 * MHz,
    "kappa_a": 0 * kHz,
    "kappa_phi": 0 * kHz,
    "T_gate": 1 / (1 * MHz),
    "Nt": 101,
}

lambdas = jnp.linspace(0, 1.0, 5)
kerr_o_kappa_2 = jnp.geomspace(1e-3, 1e-1, 11)
kerr = kerr_o_kappa_2 * params["kappa_2"]
stabilized = jnp.array([True, False])
sweep_params = OrderedDict(lam=lambdas, kerr=kerr, stabilized=stabilized)
errors, nbars, tgates = batch_cnot_errors(
    params, sweep_params, max_batch_size, cathesian_batching=True
)
# %%
fig = plt.figure(layout="constrained", figsize=(6, 5))
gs = fig.add_gridspec(2, 2, wspace=0.06, hspace=0.05)
ax = gs.subplots()
errors_name = [f"$\\epsilon_{op}^{cat}$" for op in ["Z", "X"] for cat in ["C", "T"]]
for a, data, title in zip(
    ax.flatten(), jnp.moveaxis(jnp.abs(errors), -1, 0), errors_name
):
    a.set_ylabel(title)
    a.set_xlabel(r"$K_4/\kappa_2$")
    for ind_lam, color_lam in enumerate([f"C{i}" for i in range(len(lambdas))]):
        for ind_stab, maker in enumerate(["o", "v"]):
            a.plot(kerr_o_kappa_2, data[ind_lam, :, ind_stab], f"{color_lam}{maker}")
    if a == ax[1, 1]:
        for ind in range(len(lambdas)):
            a.plot([], [], f"C{ind}o", label=f"$\\lambda = {lambdas[ind]}$")
        a.plot([], [], "ko", label="Stab")
        a.plot([], [], "kv", label="No stab")
        a.legend()
    if a in ax[0, :]:
        a.set_yscale("log")
    a.set_xscale("log")
    a.grid()
plt.savefig(os.getcwd() + "/figures/kerr_o_kappa_2.pdf", dpi=300)
# %%
# Varying kappa_phi, and fixing K_4 = kappa_a = 0

max_batch_size = 100
params = {
    "N0": 100,
    "N1": 20,
    "alpha": 2,
    "kappa_2": 1 * MHz,
    "kappa_a": 0 * kHz,
    "kerr": 0 * kHz,
    "T_gate": 1 / (1 * MHz),
    "Nt": 101,
}

lambdas = jnp.linspace(0, 1.0, 5)
kappa_phi_o_kappa_2 = jnp.geomspace(1e-3, 1e-1, 11)
kappa_phi = kappa_phi_o_kappa_2 * params["kappa_2"]
stabilized = jnp.array([True, False])
sweep_params = OrderedDict(lam=lambdas, kappa_phi=kappa_phi, stabilized=stabilized)
errors, nbars, tgates = batch_cnot_errors(
    params, sweep_params, max_batch_size, cathesian_batching=True
)
# %%
fig = plt.figure(layout="constrained", figsize=(6, 5))
gs = fig.add_gridspec(2, 2, wspace=0.06, hspace=0.05)
ax = gs.subplots()
errors_name = [f"$\\epsilon_{op}^{cat}$" for op in ["Z", "X"] for cat in ["C", "T"]]
for a, data, title in zip(
    ax.flatten(), jnp.moveaxis(jnp.abs(errors), -1, 0), errors_name
):
    a.set_ylabel(title)
    a.set_xlabel(r"$\kappa_\phi/\kappa_2$")
    for ind_lam, color_lam in enumerate([f"C{i}" for i in range(len(lambdas))]):
        for ind_stab, maker in enumerate(["o", "v"]):
            a.plot(
                kappa_phi_o_kappa_2, data[ind_lam, :, ind_stab], f"{color_lam}{maker}"
            )
    if a == ax[1, 1]:
        for ind in range(len(lambdas)):
            a.plot([], [], f"C{ind}o", label=f"$\\lambda = {lambdas[ind]}$")
        a.plot([], [], "ko", label="Stab")
        a.plot([], [], "kv", label="No stab")
        a.legend()
    if a in ax[0, :]:
        a.set_yscale("log")
    a.set_xscale("log")
    a.grid()
plt.savefig(os.getcwd() + "/figures/kappa_phi_o_kappa_2.pdf", dpi=300)
# %%
