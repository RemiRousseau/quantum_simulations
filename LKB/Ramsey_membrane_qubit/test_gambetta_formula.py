# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# %%
MHz = 2 * np.pi
kHz = 2 * np.pi / 1e3
Hz = 2 * np.pi / 1e6
us = 1
ns = 1e-3


# %%
def gambetta_formula(gamma_2, nbar, chi, kappa, t):
    theta_0 = np.arctan(2 * chi / kappa)
    exp_val = -4 * nbar * theta_0**2 * (kappa * t / 2 - 1 + np.exp(-kappa * t / 2))
    return np.exp(-gamma_2 * t) * np.exp(exp_val)


def fit_exp(t, gamma):
    return np.exp(-gamma * t)


def fit_gaussian(t, gamma):
    return np.exp(-(gamma**2) * t**2)


# %%
t = np.geomspace(1e2 * ns, 1e7 * us, 1000)
gamma_2 = 0  # 1/(100*us)
nbars = np.geomspace(1, 300, 101)
chi = 1 * Hz
kappa = 24 * Hz

fig, ax = plt.subplots()
gamma_exp, gamma_gauss = [], []
error_exp, error_gauss = [], []
for nbar in nbars:
    trace = gambetta_formula(gamma_2, nbar, chi, kappa, t)
    popt_exp, pcov_exp = curve_fit(fit_exp, t, trace)
    popt_gaussian, pcov_gaussian = curve_fit(fit_gaussian, t, trace)
    gamma_exp.append(popt_exp[0])
    gamma_gauss.append(np.abs(popt_gaussian[0]))
    error_exp.append(np.sum((fit_exp(t, *popt_exp) - trace) ** 2))
    ax.plot(t, gambetta_formula(gamma_2, nbar, chi, kappa, t))
    ax.plot(t, fit_exp(t, *popt_exp), "k--")
    ax.plot(t, fit_gaussian(t, *popt_gaussian), "r--")
ax.set_xscale("log")
plt.show()

# %%
fig, ax = plt.subplots()
ax.plot(nbars, gamma_exp, "k", label="Exponential fit")
ax.plot(nbars, gamma_gauss, "r", label="Gaussian fit")
ax.plot(nbars, chi * np.sqrt(2 * nbars), "--", label="Zaki's formula")
ax.plot(nbars, 8 * nbars * chi**2 / kappa, "--", label="Manu's formula")
ax.axvline(kappa**2 / 32 / chi**2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\bar{n}$")
ax.set_ylabel(r"$\gamma_{fit}$")
ax.grid()
ax.legend()
ax.set_title(rf"$\chi = {chi/Hz:.2f}$ Hz, $\kappa = {kappa/kHz}$ kHz")
plt.show()
# %%
