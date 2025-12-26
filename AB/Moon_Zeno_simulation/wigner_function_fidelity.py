# %%
import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from tqdm import tqdm

key = random.key(0)


# %%
def random_dm(N0, key):
    rand = random.normal(key, (2, N0, N0))
    cpl = rand[0] + 1j * rand[1]
    density_matrix = cpl @ (cpl.T.conj())
    return dq.unit(density_matrix)


def random_pure_dm(N0, key):
    rand = random.normal(key, (2, N0, 1))
    cpl = rand[0] + 1j * rand[1]
    density_matrix = cpl @ (cpl.T.conj())
    return dq.unit(density_matrix)


# %%
def compute_fidelities(N0, keys):
    fidel, integ = jnp.zeros(2), jnp.zeros(2)
    for ind, state_func in enumerate([random_pure_dm, random_dm]):
        state1, state2 = (
            state_func(N0, keys[2 * ind]),
            state_func(N0, keys[2 * ind + 1]),
        )
        fidel = fidel.at[ind].set(dq.fidelity(state1, state2).real)
        xvec, yvec, w1 = dq.wigner(state1, xmax=4, ymax=4)
        _, _, w2 = dq.wigner(state2, xmax=4, ymax=4)
        integ = integ.at[ind].set(
            jnp.sum(w1 * w2) * (xvec[1] - xvec[0]) * (yvec[1] - yvec[0]) * jnp.pi
        )
    return fidel, integ


# %%
keys = random.split(key, 5)
key, subkeys = keys[0], keys[1:]
compute_fidelities(5, subkeys)

# %%
N_sample = 500
Ns = [2, 5, 10, 20, 50]
keys = random.split(key, N_sample * 4 + 1)
key, subkeys = keys[0], keys[1:].reshape(N_sample, 4)
fidels, integs = jnp.zeros((len(Ns), N_sample, 2)), jnp.zeros((len(Ns), N_sample, 2))

for indN, N in enumerate(tqdm(Ns)):
    for ind, sk in enumerate(subkeys):
        f, i = compute_fidelities(N, sk)
        fidels = fidels.at[indN, ind].set(f)
        integs = integs.at[indN, ind].set(i)
# %%
fig, ax = plt.subplots(2, figsize=(6, 6))
for ind, title in enumerate(["Pure state", "Full random"]):
    for subind, N in enumerate(Ns):
        hist, bins = jnp.histogram(fidels[subind, :, ind] - integs[subind, :, ind], 20)
        # ax[ind].bar((bins[1:] + bins[:-1])/2, hist/jnp.max(hist), width=bins[1] - bins[0], alpha=0.5, label=f"N={N}")
        ax[ind].hist(
            fidels[subind, :, ind] - integs[subind, :, ind],
            bins=20,
            alpha=0.5,
            label=f"N={N}",
            density=True,
        )
    ax[ind].set_title(title)
    ax[ind].legend()
    ax[ind].set_yscale("log")
fig.tight_layout()
plt.show()
