# %%
import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt

MHz = 2 * jnp.pi
kHz = MHz / 1e3

N = 50
w_mem = 5 * kHz
kappa_mem = 1 * kHz
times = jnp.linspace(0, 10 / kappa_mem, 1001)

a = dq.destroy(N)
H = w_mem * dq.dag(a) @ a
x = dq.position(N)
A = a + dq.dag(a)


def auto_correlation(nth):
    Lmu = jnp.sqrt(kappa_mem * (1 + nth)) * a
    Lmd = jnp.sqrt(kappa_mem * nth) * dq.dag(a)
    ns = jnp.arange(N)
    rho_mem = dq.unit(jnp.diag((1 + nth) ** (-1.0) * (nth / (1 + nth)) ** ns)).astype(
        complex
    )
    output = dq.mesolve(H, [Lmu, Lmd], A @ rho_mem, times)
    return dq.expect(A, output.states)


fig = plt.figure()
for nth in [0, 4]:
    auto = auto_correlation(nth)
    plt.plot(times * kappa_mem, auto, label=f"$n_{{th}}={nth}$")
    plt.plot(
        times * kappa_mem,
        (1 + 2 * nth) * jnp.exp(-kappa_mem / 2 * times) * jnp.cos(w_mem * times),
        "--",
        color="black",
    )
plt.legend()
plt.xlabel("Time [1/kappa_mem]")
plt.ylabel("$<(a+a^\\dag)(t)(a+a^\\dag)(0)>$")
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

MHz = 2 * np.pi
kHz = MHz / 1e3

N = 50
w_mem = 5 * kHz
kappa_mem = 1 * kHz
times = np.linspace(0, 10 / kappa_mem, 1001)

a = qt.destroy(N)
H = w_mem * a.dag() * a
x = qt.position(N)
A = a + a.dag()

qt.correlation_2op_1t


def auto_correlation(nth):
    Lmu = np.sqrt(kappa_mem * (1 + nth)) * a
    Lmd = np.sqrt(kappa_mem * nth) * a.dag()
    output = qt.mesolve(H, A * qt.thermal_dm(N, nth), times, [Lmu, Lmd])
    return qt.expect(A, output.states)


fig = plt.figure()
for nth in [0, 4]:
    auto = auto_correlation(nth)
    plt.plot(times * kappa_mem, auto, label=f"$n_{{th}}={nth}$")
    plt.plot(
        times * kappa_mem,
        (1 + 2 * nth) * np.exp(-kappa_mem / 2 * times) * np.cos(w_mem * times),
        "--",
        color="black",
    )
plt.legend()
plt.xlabel("Time [1/kappa_mem]")
plt.ylabel("$<(a+a^\\dag)(t)(a+a^\\dag)(0)>$")
plt.show()

# %%
