# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# %%
kHz, Hz = 2 * np.pi, 2 * np.pi * 1e-3
ms = 1


# %%
def complex_spiral(t, omega, gamma):
    return np.exp(1j * omega * t - gamma * t)


def fit_function(t, omega, gamma):
    cd = complex_spiral(t, omega, gamma)
    return np.concatenate((cd.real, cd.imag))


# %%
omega, delta_omega, gamma = 2.005 * kHz, 2 * Hz, 1 / (16 * ms)
# t = np.linspace(0, 50*ms, 101)
# t = np.concatenate((np.linspace(0, 1, 11), np.linspace(20, 21, 11)))*ms
t = np.linspace(20, 21, 11) * ms
# t = np.linspace(0, 1, 11)*ms

spurious = 0.1 * complex_spiral(t, 0.1 * kHz, 0)
data = np.vstack(
    (
        complex_spiral(t, omega - delta_omega / 2, gamma) + spurious,
        complex_spiral(t, omega - delta_omega / 2, gamma) + spurious,
    )
)
amp_noise = 0.0
data += amp_noise * np.random.randn(*data.shape)
data += 1j * amp_noise * np.random.randn(*data.shape)
# data += 0.5*(np.random.random(*data.shape) - 0.5)

popt, pcov = curve_fit(
    fit_function, t, np.concatenate((data.real, data.imag)), p0=(omega, gamma)
)
plt.plot(t, data.real, "o")
plt.plot(t, data.imag, "o")

t_fit = np.linspace(t.min(), t.max(), t.size * 10)
fit_res = complex_spiral(t_fit, *popt)
plt.plot(t_fit, fit_res.real)
plt.plot(t_fit, fit_res.imag)

# %%
omegas = omega + np.linspace(-1, 1, 1001) * 0.2 * kHz
sq_err = np.zeros_like(omegas)
for i, om_g in enumerate(omegas):

    def fit_function_loc(t, gamma):
        cd = complex_spiral(t, om_g, gamma)
        return np.concatenate((cd.real, cd.imag))

    popt, pcov = curve_fit(
        fit_function_loc, t, np.concatenate((data.real, data.imag)), p0=(gamma,)
    )
    fit_res = complex_spiral(t, om_g, popt)
    sq_err[i] = np.sum(np.abs(data - fit_res) ** 2)


# %%
plt.plot(omegas, sq_err)
plt.axvline(omega, color="r")
# %%
