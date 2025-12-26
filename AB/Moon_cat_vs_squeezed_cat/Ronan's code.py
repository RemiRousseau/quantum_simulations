"""
bitflip_scaling.py

This script computes the scaling of bitflip errors in a single-mode cat qubit.
It uses QuTiP to create the system and SciPy to compute the bitflip rate for
different numbers of photons (nbar). Bitflip rates are computed from approximate
diagonalisation of the Liouvillian. The script then fits the computed bitflip
times to an exponential curve of the form
    \Gamma_Z = \Gamma_{Z,0} |\alpha|^2 \exp(-\gamma |\alpha|^2)
and plots the results.

The system parameters can be adjusted in the 'system' dictionary. The range of
nbar values to sweep over can be adjusted in the 'nbar_array'.

Author: Ronan Gautier
Date: 16/01/2024
"""

from math import pi, sqrt

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import scipy
from tqdm import tqdm

# units
Hz, kHz, MHz = 2 * pi * 1e-6, 2 * pi * 1e-3, 2 * pi * 1e0
ns, us, ms, s = 1e-3, 1e0, 1e3, 1e6

# ========================================================
# change your system parameters here
# ========================================================

# nbar array
nbar_array = np.linspace(1.0, 12.0, 12)

# system parameters
system = dict(
    kappa_2=500 * kHz,
    kappa_a=10 * kHz,
    kappa_phi_a=20 * kHz,
    K_a=50 * kHz,
    K6_a=1 * kHz,
    nth_a=0.1,
    nth_b=0.2,  # large impact on bitflip scaling
)
# ========================================================


# bitflip scaling functions
def sweep_bitflip(system, nbar_array, N):
    """Sweep over nbar values and compute bitflip rate."""
    bitflip_rate_array = np.zeros(len(nbar_array))
    for i, nbar in enumerate(tqdm(nbar_array)):
        bitflip_rate_array[i] = bitflip_rate(system, nbar, N)
    return bitflip_rate_array


def bitflip_rate(system, nbar, N):
    """Compute bitflip rate for a given nbar."""
    liouv = init_liouvillian(system, nbar, N)
    liouv_sp = scipy.sparse.csr_matrix(liouv)
    num_eigvals = 2 if system["kappa_a"] > 0 else 3
    eigvals, _ = scipy.sparse.linalg.eigs(liouv_sp, k=num_eigvals, sigma=0, which="LM")
    return -np.real(eigvals)[-1]


def init_liouvillian(system, nbar, N):
    """Instantiate the Liouvillian using QuTiP."""
    a = qt.destroy(N)
    H = 0 * a
    c_ops = []

    if system["kappa_a"] > 0:
        c_ops.append(sqrt(system["kappa_a"]) * a)
    if system["kappa_phi_a"] > 0:
        c_ops.append(sqrt(system["kappa_phi_a"]) * a.dag() * a)
    if system["kappa_a"] * system["nth_a"] > 0:
        c_ops.append(sqrt(system["kappa_a"] * system["nth_a"]) * a.dag())
    if system["nth_b"] > 0:
        c_ops.append(sqrt(system["kappa_2"] * system["nth_b"]) * (a.dag() ** 2 - nbar))
    if system["K_a"] > 0:
        H += -system["K_a"] * a.dag() ** 2 * a**2
    if system["K6_a"] > 0:
        H += system["K6_a"] * a.dag() ** 3 * a**3
    if system["kappa_2"] > 0:
        c_ops.append(sqrt(system["kappa_2"]) * (a**2 - nbar))
    else:
        raise ValueError(
            "The kappa_2 value must be positive, otherwise a qubit is not properly defined."
        )

    return qt.liouvillian(H, c_ops)


# compute bitflip
bitflip_rate_array = sweep_bitflip(system, nbar_array, N=60)
bitflip_time_array = 1 / bitflip_rate_array


# fit to exponential curve
def exponential(nbar, T0, gamma):
    return T0 * np.exp(gamma * nbar) / nbar


popt, pcov = scipy.optimize.curve_fit(
    exponential, nbar_array, bitflip_time_array, p0=(1, 1)
)

# plot bitflip
fig, ax = plt.subplots()
ax.semilogy(nbar_array, exponential(nbar_array, *popt) / s, ls="--", color="k")
ax.semilogy(nbar_array, bitflip_time_array / s, marker="o", ls="", color="C0")
ax.set_xlabel(r"$|\alpha|^2$")
ax.set_ylabel(r"$T_Z\,(\mathrm{s})$")
ax.legend(
    (
        rf"Fit with $\gamma = {popt[1]:.2f}$, $T_{{Z,\!0}} = {popt[0] / us:.2f}\,us$",
        r"Simulation",
    )
)
plt.show()
