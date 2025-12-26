# %%
import dynamiqs as dq
import GPUtil
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.optimize as opt


def set_free_gpu():
    gpus = GPUtil.getGPUs()
    gpu = min(gpus[::-1], key=lambda x: x.load)
    device = jax.devices("gpu")[gpu.id]
    jax.config.update("jax_default_device", device)
    print(f"Default device: GPU {gpu.id} ({gpu.name}) load={gpu.load*100}%")


set_free_gpu()

# %%
MHz = 2 * jnp.pi
kHz = 1e-3 * MHz
Hz = 1e-6 * MHz
GHz = 1e3 * MHz
us = 1


# %%
def simulate_evolution(delta, alpha, nth, full_hamiltonian=True, displaced_frame=True):
    sp = dq.tensor(dq.sigmap(), dq.eye(Na))
    sm = dq.tensor(dq.sigmam(), dq.eye(Na))
    sx = dq.tensor(dq.sigmax(), dq.eye(Na))
    sy = dq.tensor(dq.sigmay(), dq.eye(Na))
    sz = dq.tensor(dq.sigmaz(), dq.eye(Na))
    a = dq.tensor(dq.eye(2), dq.destroy(Na))

    rho_qb = dq.todm(dq.unit(dq.fock(2, 0) + dq.fock(2, 1)))
    if nth == 0:
        rho_mem = dq.fock_dm(Na, 0)
    else:
        n = jnp.arange(Na)
        rho_mem = dq.unit(jnp.diag((1.0 + nth) ** (-1.0) * (nth / (1.0 + nth)) ** (n)))

    if displaced_frame:
        a = a + alpha * dq.eye(Na * 2)
    else:
        disp = dq.displace(Na, alpha)
        rho_mem = disp @ rho_mem @ dq.dag(disp)
    rho_0 = dq.tensor(rho_qb, rho_mem)

    if full_hamiltonian:
        H = g * dq.modulated(lambda t: jnp.exp(-1j * delta * t), sp @ a)
        H += g * dq.modulated(lambda t: jnp.exp(1j * delta * t), sm @ dq.dag(a))
        H += g * dq.modulated(
            lambda t: jnp.exp(1j * (2 * wm - delta) * t), sp @ dq.dag(a)
        )
        H += g * dq.modulated(lambda t: jnp.exp(-1j * (2 * wm - delta) * t), sm @ a)
    else:
        chi = jnp.sqrt(2) * g**2 / delta
        H = chi * dq.dag(a) @ a @ sz / 2

    c_ops = []
    if kappa_1 > 0:
        c_ops.append(jnp.sqrt(kappa_1 / 2) * sm)
        c_ops.append(jnp.sqrt(kappa_1 / 2) * sp)
    if kappa_phi > 0:
        c_ops.append(jnp.sqrt(kappa_phi / 2) * sz)
    if kappa_mem > 0:
        c_ops.append(jnp.sqrt(kappa_mem * (1 + nth)) * a)
    if kappa_mem * nth > 0:
        c_ops.append(jnp.sqrt(kappa_mem * nth) * dq.dag(a))

    error_a = a @ dq.dag(a) - dq.dag(a) @ a
    return dq.mesolve(
        H, c_ops, rho_0, jnp.linspace(0, t_max, Nt), exp_ops=[sx, sy, sz, error_a]
    ).expects


# %%
Na = 30
wm = 4.4 * MHz
delta = 2.0 * MHz
g = 2 * kHz
kappa_1 = 1 / (30 * us)
kappa_2 = 1 / (30 * us)
kappa_phi = max(kappa_2 - kappa_1 / 2, 0)
kappa_mem = 200 * Hz
t_max = 50 * us
Nt = 10_000
nth = 0
chi_alpha_2_max = 0.2 * MHz
full_hamiltonian = False
displaced_frame = True

chi = jnp.sqrt(2) * g**2 / delta
print("chi = ", chi / kHz, " kHz")
alpha_max = jnp.sqrt(chi_alpha_2_max / chi)
print("nbar_max = ", alpha_max**2)

print("chi/kappa_1 = ", chi / kappa_mem)
alphas = jnp.linspace(0, alpha_max, 51)


# %%
alphas = jnp.sqrt(jnp.geomspace(1e-5, 1, 51)) * alpha_max
expects = jax.vmap(simulate_evolution, in_axes=(None, 0, None, None, None))(
    delta, alphas, nth, full_hamiltonian, displaced_frame
)

# %%
plt.imshow(
    jnp.log10(jnp.abs(expects[:, 3] - 1)),
    interpolation="none",
    aspect="auto",
    origin="lower",
    extent=(0, t_max, 0, jnp.max(alphas) ** 2),
)
plt.colorbar()
# %%
tsave = jnp.linspace(0, t_max, Nt)


def complex_decay(t, w, gamma):
    return jnp.exp(-1j * w * t - gamma * t)


def complex_decay_gauss(t, w, gamma):
    return jnp.exp(-1j * w * t - gamma**2 * t**2)


Zs = expects[:, 0] + 1j * expects[:, 1]
popts, pcovs, fits = [], [], []
artificial = 4 * jnp.pi / jnp.max(tsave)
for ind in range(len(alphas)):
    Zs = expects[ind, 0] + 1j * expects[ind, 1]
    Zs *= jnp.exp(1j * artificial * tsave)
    guess = (
        -jnp.sqrt(2) * alphas[ind] ** 2 * g**2 / delta - artificial,
        kappa_1 / 2 + kappa_phi,
    )
    # try:
    popt_exp, pcov_exp = opt.curve_fit(complex_decay, tsave, Zs, p0=guess)
    # except RuntimeError:
    # popt_exp, pcov_exp = guess, jnp.diag(jnp.ones(2)*jnp.inf)
    # try:
    popt_gauss, pcov_gauss = opt.curve_fit(complex_decay_gauss, tsave, Zs, p0=guess)
    # except RuntimeError:
    # popt_gauss, pcov_gauss = guess, jnp.diag(jnp.ones(2)*jnp.inf)
    fit_exp = complex_decay(tsave, *popt_exp)
    fit_gauss = complex_decay_gauss(tsave, *popt_gauss)
    if (
        1
    ):  # jnp.sum(jnp.abs(fit_exp - Zs) ** 2) < jnp.sum(jnp.abs(fit_gauss - Zs) ** 2):
        popt, pcov, fit = popt_exp, pcov_exp, fit_exp
    else:
        popt, pcov, fit = popt_gauss, pcov_gauss, fit_gauss
        popt[1] = jnp.abs(popt[1])
    popts.append(popt)
    pcovs.append(pcov)
    fits.append(fit)
    # plt.plot(tsave, jnp.real(Zs))
    # plt.plot(tsave, jnp.real(fit), "k--")
    # plt.plot(tsave, jnp.imag(Zs))
    # plt.plot(tsave, jnp.imag(fit), "k--")
    # plt.show()
popts = jnp.array(popts)
popts = popts.at[:, 0].set(popts[:, 0] + artificial)
pcovs = jnp.array(pcovs)
fits = jnp.array(fits) * jnp.exp(-1j * artificial * tsave[None, :])

Zs = expects[:, 0] + 1j * expects[:, 1]
fig, ax = plt.subplots(3, 2, figsize=(8, 10))
for i_l, (vals, name) in enumerate(zip([Zs, fits], ["sim", "fit"])):
    for i_c, func in enumerate([jnp.real, jnp.imag]):
        ax[i_l, i_c].imshow(
            func(vals),
            aspect="auto",
            origin="lower",
            extent=(0, t_max, 0, jnp.max(alphas) ** 2),
            interpolation="none",
        )
        ax[i_l, i_c].set_xlabel("Time (us)")
        ax[i_l, i_c].set_ylabel("$\\bar{n}$ [photons]")
        ax[i_l, i_c].set_title(f"{func.__name__}({name})")
        # ax[i_l, i_c].set_yscale('log')

ax[2, 0].plot(alphas**2, popts[:, 0] / MHz, "o", label="fit")
ax[2, 0].plot(alphas**2, -chi / MHz * alphas**2, label="theory")
ax[2, 0].set_xlabel("$\\bar{n}$ [photons]")
ax[2, 0].set_ylabel("$\\Delta_{\\rm pull}$ [MHz]")
ax[2, 0].grid()
ax[2, 0].legend()
ax[2, 0].set_xscale("log")

ax[2, 1].plot(alphas**2, popts[:, 1] / kHz, "o", label="fit")
ax[2, 1].set_xscale("log")
# ax[2, 1].set_yscale("log")
ax[2, 1].axhline(kappa_2 / kHz, c="y", ls="--", label="$\\kappa_2$")
ax[2, 1].set_ylim(*ax[2, 1].get_ylim())
ax[2, 1].set_xlim(*ax[2, 1].get_xlim())
gamma_zaki = chi * jnp.sqrt(alphas**2 / 2)
gamma_manu = 2 * alphas**2 * chi**2 / kappa_mem
ax[2, 1].plot(alphas**2, (kappa_2 + gamma_zaki) / kHz, "k-", label="Gauss")
ax[2, 1].plot(alphas**2, (kappa_2 + gamma_manu) / kHz, "r-", label="Exp")
ax[2, 1].axvline(kappa_mem**2 / 8 / chi**2, c="g", ls="--", label="Crossover")
ax[2, 1].set_xlabel("$\\bar{n}$ [photons]")
ax[2, 1].set_ylabel("$\\Gamma$ [kHz]")
ax[2, 1].grid()
ax[2, 1].legend()

fig.suptitle(
    f"$g/2\\pi = ${g/kHz:_.0f} kHz, $\\kappa_m2\\pi = ${kappa_mem/kHz:_.0f} kHz, $\\Delta/2\\pi = ${delta/kHz:_.0f} kHz, $\\chi/\\kappa_1 = ${chi/kappa_mem:_.0e}"
)
fig.tight_layout()
plt.show()

# %%
