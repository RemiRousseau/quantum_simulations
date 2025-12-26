# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import scipy
from tqdm import tqdm

matplotlib.rc_file("abplot.matplotlibrc")

# units
Hz, kHz, MHz = 2 * np.pi * 1e-6, 2 * np.pi * 1e-3, 2 * np.pi * 1e0
ns, us, ms, s = 1e-3, 1e0, 1e3, 1e6

colors = [(0, 0, 0), (0.8, 0.25, 0.3), (1, 0.85, 0.2)]
# colors = [(0, 0, 0), (0.4, 0.7, 0.3), (1, 0.85, 0.2)]
cs = scipy.interpolate.CubicSpline(np.linspace(0, 1, len(colors)), colors)
colors_cmap = np.clip(cs(np.linspace(0, 1, 101)), 0, 1)
ab_cmap = matplotlib.colors.ListedColormap(colors_cmap, "ab")


# %%
def init_L2(nbar: float, lam: float, squeezed: bool, N: int):
    a = qt.destroy(N)
    if squeezed:
        tan_r = lam / 2
        # tan_r = lam/(2+lam)
        r = np.arctanh(tan_r)
        alpha2 = np.exp(2 * r) * (nbar - np.sinh(r) ** 2)
        L_2 = (
            a**2
            + tan_r**2 * a.dag() ** 2
            + tan_r * (a * a.dag() + a.dag() * a)
            - alpha2
        )
    else:
        L_2 = a**2 - nbar + lam * (a.dag() * a - nbar)
    return L_2


def get_nbar_true(L2: qt.Qobj, N: int):
    a = qt.destroy(N)
    Hk = L2.dag() * L2
    _, eigvects = Hk.eigenstates()
    return qt.expect(a.dag() * a, eigvects[0])


def get_H_cops(system: dict, N: int):
    a = qt.destroy(N)
    H = 0 * a
    c_ops = []
    if system["kappa_a"] > 0:
        c_ops.append(np.sqrt(system["kappa_a"]) * a)
    if system["kappa_phi_a"] > 0:
        c_ops.append(np.sqrt(2 * system["kappa_phi_a"]) * a.dag() * a)
    if system["kappa_a"] * system["nth_a"] > 0:
        c_ops.append(np.sqrt(system["kappa_a"] * system["nth_a"]) * a.dag())
    if system["K_a"] > 0:
        H += -system["K_a"] / 2 * a.dag() ** 2 * a**2
    return H, c_ops


def error_rates(system: dict, nbars: np.ndarray, lambdas: np.ndarray, N: int):
    nbars_true = np.zeros((2, len(lambdas), len(nbars)))
    bit_flips = np.zeros((2, len(lambdas), len(nbars)))
    phase_flips = np.zeros((2, len(lambdas), len(nbars)))
    H, c_ops = get_H_cops(system, N)
    for squeezed in range(2):
        for i, lam in enumerate(tqdm(lambdas)):
            for j, nbar in enumerate(nbars):
                L2 = init_L2(nbar, lam, squeezed, N * 2)
                nbars_true[squeezed, i, j] = get_nbar_true(L2, N * 2)
                L2 = init_L2(nbar, lam, squeezed, N)
                liouv = qt.liouvillian(H, c_ops + [L2])
                liouv_sp = scipy.sparse.csr_matrix(liouv.full())
                eigvals, _ = scipy.sparse.linalg.eigs(
                    liouv_sp, k=3, sigma=0, which="LM"
                )
                bit_flips[squeezed, i, j] = -np.real(eigvals[1])
                phase_flips[squeezed, i, j] = -np.real(eigvals[2:]).mean()
    return nbars_true, bit_flips, phase_flips


# %%
system = dict(
    kappa_2=1 * MHz,
    kappa_a=1 * kHz,
    kappa_phi_a=20 * kHz,
    K_a=15 * kHz,
    nth_a=0.0,  # 93,
)
nbar_array = np.linspace(3.0, 8.0, 11)
lam_array = np.linspace(0.0, 0.8, 3)
N = 80
nbars_true, bit_flips, phase_flips = error_rates(system, nbar_array, lam_array, N)
# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, gridspec_kw={"wspace": 0.3})
colors = ab_cmap(np.linspace(0, 1, len(lam_array)))
markersize = 8
mew = 1.5
fillstyle = "none"
for sq, fmt in enumerate(["v", "o"]):
    for i, lam in enumerate(lam_array):
        nb = nbars_true[sq, i]
        ind_in = nb < np.max(nbar_array) + 1
        ax[0].plot(
            nbars_true[sq, i][ind_in],
            bit_flips[sq, i][ind_in] / system["kappa_a"],
            fmt,
            color=colors[i],
            markersize=markersize,
            fillstyle=fillstyle,
            mew=mew,
        )
        ax[1].plot(
            nbars_true[sq, i][ind_in],
            phase_flips[sq, i][ind_in] / system["kappa_a"],
            fmt,
            color=colors[i],
            markersize=markersize,
            fillstyle=fillstyle,
            mew=mew,
        )
ax[0].plot(
    [],
    [],
    "kv",
    label="$\\rm moon$",
    markersize=markersize,
    fillstyle=fillstyle,
    mew=mew,
)
ax[0].plot(
    [],
    [],
    "ko",
    label="$\\rm squeezed$",
    markersize=markersize,
    fillstyle=fillstyle,
    mew=mew,
)
for i, lam in enumerate(lam_array):
    ax[0].plot(
        [],
        [],
        ".",
        color=colors[i],
        label=f"$\\lambda={lam:.1f}$",
        markersize=markersize,
    )
ax[0].set_yscale("log")
ax[0].legend(fontsize=18)
ax[0].set_xlabel("$\\bar{n} \\rm \\; [photons]$", fontsize=18)
ax[1].set_xlabel("$\\bar{n} \\rm \\; [photons]$", fontsize=18)
ax[0].set_ylabel("$\\Gamma_{\\rm Z}/\\kappa_{\\rm a}$", fontsize=18)
ax[1].set_ylabel("$\\Gamma_{\\rm X}/\\kappa_{\\rm a}$", fontsize=18)
exps = np.arange(-7, 1, 2)
ax[0].set_yticks(10.0**exps, [f"$10^{{{exp}}}$" for exp in exps], fontsize=18)
ticks = np.arange(6, 19, 2)
ax[1].set_yticks(ticks, [f"${t:.0f}$" for t in ticks], fontsize=18)
ticks = np.arange(3, 10, 1)
ax[0].set_xticks(ticks, [f"${t:.0f}$" for t in ticks], fontsize=18)
ax[1].set_xticks(ticks, [f"${t:.0f}$" for t in ticks], fontsize=18)

trans = matplotlib.transforms.ScaledTranslation(-67 / 72, 0 / 72, fig.dpi_scale_trans)
ax[0].text(
    0, 1.0, "$\\rm (A)$", transform=ax[0].transAxes + trans, va="top", fontsize=18
)
trans = matplotlib.transforms.ScaledTranslation(-53 / 72, 0 / 72, fig.dpi_scale_trans)
ax[1].text(
    0, 1.0, "$\\rm (B)$", transform=ax[1].transAxes + trans, va="top", fontsize=18
)

# plt.show()
plt.savefig("moon_vs_squeezed_cats.pdf", bbox_inches="tight", dpi=300, format="pdf")
# %%
