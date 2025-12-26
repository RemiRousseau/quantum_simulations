# %%
import sympy as sp
from IPython.display import display

# %%
ma, mad, da, dad = sp.symbols("M[a] M[a^{\\dag}] D[a] D[a^{\\dag}]", commutative=False)
mat, madt, dat, dadt = sp.symbols(
    "M[a(t)] M[a^{\\dag}(t)] D[a(t)] D[a^{\\dag}(t)]", commutative=False
)
m1, m2, l1, l2 = sp.symbols("m_1 m_2 l_1 l_2")
kplus = (1 + m1 * ma + m2 * mad) / 2
kminus = (1 - m1 * ma - m2 * mad) / 2
kplust = (1 + m1 * mat + m2 * madt) / 2
kminust = (1 - m1 * mat - m2 * madt) / 2
(kplust * kplus + kminust * kminus - kplust * kminus - kminust * kplus).expand()

# %%
ma, mad, da, dad = sp.symbols("M[a] M[a^{\\dag}] D[a] D[a^{\\dag}]", commutative=False)
mat, madt, dat, dadt = sp.symbols(
    "M[a(t)] M[a^{\\dag}(t)] D[a(t)] D[a^{\\dag}(t)]", commutative=False
)
m1, m2, l1, l2 = sp.symbols("m_1 m_2 l_1 l_2")
kplus = (1 + m1 * ma + m2 * mad) / 2
kminus = (1 - m1 * ma - m2 * mad) / 2
kplust = (1 + m1 * mat + m2 * madt) / 2
kminust = (1 - m1 * mat - m2 * madt) / 2

p_pp, p_mm, p_pm, p_mp = sp.symbols(
    "p_{++} p_{--} p_{+-} p_{-+}", real=True, positive=True
)
eps = sp.symbols("\\epsilon", real=True, positive=True)
eps_g = sp.symbols("\\epsilon_g", real=True, positive=True)
eps_e = sp.symbols("\\epsilon_e", real=True, positive=True)

p_pp = 1 - eps_e
p_pm = eps_e
p_mp = eps_g
p_mm = 1 - eps_g

kplus_p = p_pp * kplus + p_pm * kminus
kminus_p = p_mp * kplus + p_mm * kminus
kplust_p = p_pp * kplust + p_pm * kminust
kminust_p = p_mp * kplust + p_mm * kminust
correlation = (
    kplust_p * kplus_p
    + kminust_p * kminus_p
    - kplust_p * kminus_p
    - kminust_p * kplus_p
).expand()
correlation
# %%
terms = [mat * ma, madt * mad, mat * mad, madt * ma, madt, mat, mad, ma]
prefactors = [0] * len(terms)
rest = 0
for term in correlation.args:
    found = False
    for ind, el in enumerate(terms):
        if term.has(el):
            prefactors[ind] += term.subs({madt: 1, mat: 1, mad: 1, ma: 1})
            found = True
            break
    if not found:
        rest += term
correls_syms = sp.symbols(r"Re(C_{{a^\dagger}a}) Re(C_{aa^\dagger})")
for term, (i0, i1) in zip(correls_syms, [(0, 3), (1, 2)]):
    display(term, (prefactors[i0] + prefactors[i1]).expand())
for term, of in zip([mad, ma], [4, 5]):
    v = prefactors[of] + prefactors[of + 2]
    display(term, v)
try:
    display(rest.simplify())
except AttributeError:
    display(1, rest)
# %%
nth = sp.symbols("n_{th}", real=True, positive=True)
amp = (2 * eps - 1) ** 2 * (m1 + m2) * (m1 * nth + m2 * (nth + 1))
print("Amplitude expression")
display(amp)
omega, tau_sigma, tau_2, p = sp.symbols(
    "\\Omega \\tau_{\\Sigma} \\tau_2 p", real=True, positive=True
)
m1_expr, m2_expr = (
    omega / 2 * ((2 * p - 1) * tau_sigma + tau_2),
    omega / 2 * ((2 * p - 1) * tau_sigma - tau_2),
)
print("m1 and m2 expression")
display(m1_expr, m2_expr)
amp = amp.subs({m1: m1_expr, m2: m2_expr})
eta_g, eta_e = sp.symbols("\\eta_g \\eta_e", real=True, positive=True)
# eta = sp.symbols("\\eta", real=True, positive=True)
# eta_g, eta_e = eta, eta
amp_g = amp.replace(p, (1 + eta_g) / 2).simplify()
amp_e = amp.replace(p, (1 - eta_e) / 2).simplify()
print("Amp_g and Amp_e expression")
display(amp_g, amp_e)

tau = sp.symbols("\\tau", real=True, positive=True)
ideal_replacement = {
    tau_sigma: tau,
    tau_2: tau,
    # eta_g: 1,
    # eta_e: 1,
}
amp_g_ideal = amp_g.subs(ideal_replacement).simplify()
amp_e_ideal = amp_e.subs(ideal_replacement).simplify()
print("Ideal Amp_g and Amp_e expression")
display(amp_g_ideal, amp_e_ideal)

amp_calib_g = (1 - 2 * eps) ** 2 * eta_g**2
amp_calib_e = (1 - 2 * eps) ** 2 * eta_e**2
amp_g_norm = (amp_g / amp_calib_g).simplify()
amp_e_norm = (amp_e / amp_calib_e).simplify()
print("Normalized Amp_g and Amp_e expression")
display(amp_g_norm, amp_e_norm)

amp_g_norm_ideal = amp_g_norm.subs(ideal_replacement).simplify()
amp_e_norm_ideal = amp_e_norm.subs(ideal_replacement).simplify()
print("Ideal normalized Amp_g and Amp_e expression")
display(amp_g_norm_ideal, amp_e_norm_ideal)
# %%


import matplotlib.pyplot as plt
import numpy as np

n_ave, n_sample = 10_000, 400
eps_a, eps_b = 0.1, 0.2
true_vals = (np.random.rand(n_ave, n_sample) > 0.5) * 2 - 1
random_2 = np.random.rand(n_ave, n_sample)
samples = np.zeros_like(true_vals)
mask_1 = true_vals == 1
samples[mask_1] = (random_2[mask_1] > 1 - eps_a) * 2 - 1
mask_2 = true_vals == -1
samples[mask_2] = (random_2[mask_2] < 1 - eps_b) * 2 - 1
print(np.mean(samples), eps_a - eps_b)
fft = np.fft.rfft(samples, axis=1)
plt.figure()
plt.plot(np.mean(np.abs(fft) ** 2, axis=0)[1:] / n_sample, label="FFT")
plt.axhline(1 - (eps_a - eps_b) ** 2)
# %%
