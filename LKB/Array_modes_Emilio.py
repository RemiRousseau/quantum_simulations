# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig, eigvals, inv
from scipy.constants import e, h, pi


# %%
def freqs_normal_modes(C: np.ndarray, Ej_array: np.ndarray):
    """
    This function computes the normal mode frequencies of the linearized circuit.
    It takes as input the capacitance matrix C defined as follows:
    T = 1/2 Phi_dot.T @ C @ Phi_dot = sum_i=0^N Cg,i * Phi_dot[i]^2 + \sum_i=1^N CJ,i * (Phi_dot[i] - Phi_dot[i-1])^2 + CJb * (Phi_dot[0] - Phi_dot[N])^2

    and the vector of Josephson energies Ej_array along the array.

    The way it works is the following:
    It first transforms the capacitance matrix to the theta basis (phase difference between junctions) using the transformation matrix R.
    theta_i = phi_i - phi_{i+1} for i = 0, ..., N-1 # see https://arxiv.org/abs/1912.01018 app.A

    Then it finds normal mode frequency by simultaneous diagonalization of capacitance and inductance matrix. see https://arxiv.org/abs/2010.00620
    This is actually not needed as the inductance matrix is diagonal in the theta basis, but it makes the code easier to expand in the future.


    Args:
    -----------------------------------
        C (np.ndarray): Capacitance matrix
        Ej_array (np.ndarray): Vector of Josephson energies along the array

    Returns:
    -----------------------------------
        ωs_chain (np.ndarray): Normal mode frequencies of the chain. The first element (mode 0) is the fundamental (Fluxonium mode) and the rest are the chain modes.
    """
    assert C.shape[0] == C.shape[1], "C must be a square matrix"
    N = C.shape[0] - 1
    assert N == len(Ej_array), "Ej_array must have length N"

    # going to the theta basis (phase difference between junctions) - see https://arxiv.org/abs/1912.01018 app.A and http://arxiv.org/abs/1506.08599
    R = np.eye((N + 1))
    for i in range(0, N):
        R[i, i + 1] += -1
    R[-1, :] = 1
    invR = inv(R)
    Cθ = invR.T @ C @ invR
    Cθ = Cθ[:N, :N]  # neglecting zero frequncy mode

    # L^-1 matrix
    LJ = 163.46 / Ej_array
    inv_LJ = 1 / LJ
    inv_Lθ = np.diag(inv_LJ)

    # Diagonalize the linearized circuit - see https://arxiv.org/abs/2010.00620
    Λ, O = eig(inv_Lθ)
    Λ_inv_sqrt = np.diag(1 / np.sqrt(Λ))

    C_tilde = Λ_inv_sqrt @ O.T @ Cθ @ O @ Λ_inv_sqrt

    ωs_chain = np.sort(np.abs(1 / np.sqrt(eigvals(C_tilde))) / (2 * pi))

    return ωs_chain


# %%
# Computing the coupling matrix between fluxonium mode and chain modes
# see http://arxiv.org/abs/1506.08599


def coupling_matrix(C: np.ndarray):
    """
    This function computes the couplings between the fluxonium mode and the chain modes.

    The capacitance matrix `C` is defined as:
    T = 1/2 Phi_dot.T @ C @ Phi_dot = sum_i=0^N Cg,i * Phi_dot[i]^2
        + sum_i=1^N CJ,i * (Phi_dot[i] - Phi_dot[i-1])^2
        + CJb * (Phi_dot[0] - Phi_dot[N])^2

    The method works as follows:
    1. Transform the capacitance matrix `C` to the normal mode basis as defined in Viola & Catelani
       (see http://arxiv.org/abs/1506.08599).
       - The normal mode basis is defined as:
         xi_0 = sum_m=1^N theta_m
         xi_mu = sum_m W_m,mu * theta_m for mu = 1, ..., N-1
         (Refer to the paper for the definition of W).

    2. Build the Hamiltonian using a Legendre transformation:
       H = 1/2 sum_ij n_i * ECmatrix_ij * n_j - U
       where ECmatrix = (e^2 / h) * (C^-1).

    3. Extract the coupling matrix `g` from the off-diagonal terms of ECmatrix.
       - The diagonal terms represent the charging energy of the modes.
       - The off-diagonal terms g_0,mu represent the coupling between the fluxonium mode and the mu-th chain mode.

    Args:
    -----------------------------------
        C (np.ndarray): Capacitance matrix.

    Returns:
    -----------------------------------
        g_μ (np.array): Coupling matrix between the fluxonium mode and the chain modes.
    """
    N = C.shape[0] - 1
    assert C.shape[0] == C.shape[1], "C must be a square matrix"

    # going to the theta basis (phase difference between junctions) - see https://arxiv.org/abs/1912.01018 app.A and http://arxiv.org/abs/1506.08599
    R = np.eye((N + 1))
    for i in range(0, N):
        R[i, i + 1] += -1
    R[-1, :] = 1
    invR = inv(R)
    Cθ = invR.T @ C @ invR
    Cθ = Cθ[:N, :N]  # neglecting zero frequncy mode

    # Transformation from theta basis, to phi,xi basis of Viola,Catelani et al. see https://arxiv.org/abs/1506.08599
    def W(mu, m, N):
        return np.sqrt(2 / N) * np.cos(pi * mu * (m - 0.5) / N)

    Wm = np.zeros((N, N))
    Wm[:, 0] = 1 / N
    for m in range(1, N + 1):
        for mu in range(1, N):
            Wm[m - 1, mu] = W(mu, m, N)
    C_w = Wm.T @ Cθ @ Wm

    full_EC = (e**2 / (2 * h)) * inv(C_w)
    EC_s = np.diag(full_EC)
    # zpfs = [(8 * Ec / EJ_a) ** 0.25 for Ec in EC_s]
    # n_zpfs = np.array([1 / (np.sqrt(2) * zpf) for zpf in zpfs])
    g_μ = full_EC[0, 1:]  # np.multiply(full_EC[0, 1:], 4 * n_zpfs[1:])

    return g_μ


def coupling_RO(C: np.ndarray, N):
    """
    This function computes the couplings between the fluxonium mode and the RO resonator.
    Everything is the same as in the coupling_matrix function, except that we have an extra mode (the RO resonator) in the capacitance matrix.
    Such mode needs to be added in position 0 of the capacitance matrix.
    The capacitance matrix `C` is defined as:
    T = 1/2 Phi_dot.T @ C @ Phi_dot = sum_i=1^N+1 Cg,i * Phi_dot[i]^2
        + sum_i=1^N+1 CJ,i * (Phi_dot[i] - Phi_dot[i-1])^2
        + CJb * (Phi_dot[0] - Phi_dot[N])^2 + C_res * Phi_dot[0]^2 + C_c * (Phi_dot[0] - Phi_dot[1])^2

    Args
    -----------------------------------
        C (np.ndarray): Capacitance matrix.
    Returns
    -----------------------------------
        g_ro (np.array): Coupling matrix between the fluxonium mode and the RO resonator.
    """

    assert C.shape == (N + 2, N + 2), (
        "C must be a square matrix of size (N+2,N+2), resonator mode needs to be the first one"
    )

    # going to the theta basis (phase difference between junctions) - see https://arxiv.org/abs/1912.01018 app.A and http://arxiv.org/abs/1506.08599
    R = np.eye((N + 2))
    for i in range(1, N + 1):
        R[i, i + 1] += -1
    R[-1, :] = 1
    invR = inv(R)
    Cθ = invR.T @ C @ invR
    Cθ = Cθ[: N + 1, : N + 1]  # neglecting zero frequncy mode

    # Transformation from theta basis, to phi,xi basis of Viola,Catelani et al. see https://arxiv.org/abs/1506.08599
    def W(mu, m, N):
        return np.sqrt(2 / N) * np.cos(pi * mu * (m - 0.5) / N)

    Wm = np.zeros((N + 1, N + 1))
    Wm[0, 0] = 1
    Wm[:, 1] = 1 / N
    for m in range(1, N + 1):
        for mu in range(1, N):
            Wm[m, mu + 1] = W(mu, m, N)
    C_w = Wm.T @ Cθ @ Wm

    full_EC = (e**2 / (2 * h)) * inv(C_w)
    EC_s = np.diag(full_EC)
    g_ro = full_EC[0, 1:]  # np.multiply(full_EC[0, 1:], 4 * n_zpfs[1:])
    return g_ro


# %%
# Example usage


# Constants
ħ = h / (2 * pi)
ϕ0 = h / (2 * e)
Rq = h / (2 * e) ** 2

# Rescaling units
scaling_factor = 1e9
nH = 1e-9 * scaling_factor
GHz = 1e9 / scaling_factor
fF = 1e-15 * scaling_factor


# Circuit parameters
N = 360
El = 0.17 * GHz
EJ_a = El * N  # EJ array
Z = 5000  # Array impedance
ωp_j = 16 * GHz  # Plasma frequency of the array junctions
Ejb = 4.3 * GHz  # black sheep EJ
EC_b = 0.47 * GHz  # Shunt capacitance
C_c = 2 * fF  # Coupling capacitance to RO resonator

Lj = 163.46 / EJ_a
Cg = Lj / Z**2  # ground capacitance Z = sqrt(Lj / Cg)
EC_g = (e**2 / h) / (2 * Cg)  # ground capacitance
EC_a = (ωp_j) ** 2 / (8 * EJ_a)
CJ = (e**2 / h) / (2 * EC_a)  # Array capacitance
CJb = (e**2 / h) / (2 * EC_b)
Cg_0 = Cg
EC_c = (e**2 / h) / (2 * C_c)

# %%

# Capacitance matrix - see https://arxiv.org/abs/1912.01018 app.A
C = np.zeros((N + 1, N + 1))
# black sheep capacitance
C[0, 0] += CJb
C[-1, -1] += CJb
C[0, -1] -= CJb
C[-1, 0] -= CJb
for i in range(0, N + 1):
    C[i, i] += Cg

for i in range(1, N + 1):
    C[i, i] += CJ
    C[i - 1, i - 1] += CJ
    C[i, i - 1] -= CJ
    C[i - 1, i] -= CJ

C[0, 0] += C_c

# %%
# Computing the frequencies of the normal modes

freqs = freqs_normal_modes(C, np.array([EJ_a] * N))

N_max = 40
fig, ax = plt.subplots()
indexes = np.arange(0, N_max, 2)
ax.plot(indexes, freqs[indexes], "o", label="Full even")
indexes = np.arange(1, N_max, 2)
ax.plot(indexes, freqs[indexes], "o", label="Full odd")
ax.set_title("Normal mode frequencies")
ax.set_xlabel("Mode number")
ax.set_ylabel("Frequency (GHz)")

indexes = np.arange(0, N_max, 2)
ana_formula = (
    indexes / 2 * freqs[2] / np.sqrt(1 + (indexes / 2 * freqs[2] / freqs[-1]) ** 2)
)
ax.plot(indexes, ana_formula, "o", label="Analytical formula open chain")
ax.legend()

# %%
# computing the couplings of the chain modes with the fluxonium mode
g_μ = coupling_matrix(C)
fig, ax = plt.subplots()
ax.plot(np.abs(g_μ), "o")
ax.set_title("Coupling between fluxonium mode and chain modes")
ax.set_xlabel("Mode number")
ax.set_ylabel("Coupling (GHz)")
ax.set_yscale("log")

# %%
# Example coupling between chain modes to RO resonator
# RO resonator parameters
ωr = 6 * GHz
El_res = 100 * GHz
L_res = 163.46 / El_res
ω_res = 6 * GHz
Z_res = 80  # defined as √(L_res / C_res)
Ec_res = ω_res**2 / (8 * El_res)
C_res = (e**2 / (2 * h)) / Ec_res

C_matrix_ro = np.zeros((N + 2, N + 2))  # RO resonator is in position 0
# RO
C_matrix_ro[0, 0] += C_res
# coupling capacitor
C_matrix_ro[0, 0] += C_c
C_matrix_ro[1, 1] += C_c
C_matrix_ro[0, 1] -= C_c
C_matrix_ro[1, 0] -= C_c

# black sheep capacitance
C_matrix_ro[1, 1] += CJb
C_matrix_ro[-1, -1] += CJb
C_matrix_ro[1, -1] -= CJb
C_matrix_ro[-1, 1] -= CJb
for i in range(1, N + 2):
    C_matrix_ro[i, i] += Cg

for i in range(2, N + 2):
    C_matrix_ro[i, i] += CJ
    C_matrix_ro[i - 1, i - 1] += CJ
    C_matrix_ro[i, i - 1] -= CJ
    C_matrix_ro[i - 1, i] -= CJ

g_RO = coupling_RO(C_matrix_ro, N)
fig, ax = plt.subplots()
ax.plot(np.abs(g_RO), "o")
ax.set_title("Coupling between fluxonium mode and RO resonator")
ax.set_xlabel("Mode number")
ax.set_ylabel("Coupling (GHz)")
ax.set_yscale("log")
