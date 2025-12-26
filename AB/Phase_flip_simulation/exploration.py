#%%

import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%%
MHz = 2*jnp.pi
kHz = MHz*1e-3
us = 1

# %%
def exp_decr(t, kappa, A, B):
    return A*jnp.exp(-kappa*t) + B

def fit_parity(t, parity):
    B = parity[-1]
    A = parity[0] - B
    kappa = -jnp.polyfit(t, jnp.log(parity - B), 1)[0]
    guess = [1, A, B]
    try:
        popt, pcov = curve_fit(exp_decr, t, parity, p0=guess, bounds=(0, jnp.inf))
        return popt, jnp.sqrt(jnp.diag(pcov))
    except:
        return guess, jnp.inf*jnp.ones(3)

# %%
def simulate_phase_flip(N: int, alpha: float, kappa_2: float, kappa_1: float, n_th: float, t_max: float)-> jax.Array:
    a = dq.destroy(N)

    H = 0*dq.eye(N)

    L2 = jnp.sqrt(kappa_2)*(a@a - alpha**2*dq.eye(N))
    L1_u = jnp.sqrt(kappa_1*n_th)*dq.dag(a)
    L1_d = jnp.sqrt(kappa_1*(n_th+1))*a
    c_ops = [L1_u, L1_d, L2]

    ket_0 = dq.unit(dq.coherent(N, alpha) + dq.coherent(N, -alpha))

    t_save = jnp.linspace(0, t_max, 101)

    output = dq.mesolve(H, c_ops, ket_0, t_save, exp_ops=[dq.parity(N)])
    return output.expects[0].real, output.tsave

# %%
N = 20
alpha = jnp.sqrt(0.5)
kappa_2 = 0.385*MHz
kappa_1 = 4*kHz
n_th = 0.9
t_max = 100*us

parity, tsave = simulate_phase_flip(N, alpha, kappa_2, kappa_1, n_th, t_max)

# %%
fig, ax = plt.subplots()
ax.plot(tsave, parity)
popt, pstd = fit_parity(tsave, parity)
print(popt)
ax.plot(tsave, exp_decr(tsave, *popt), '--')

#%%
N = 20
kappa_2 = 2*MHz
kappa_1 = 4*kHz
n_th = 0.9
t_max = 10*us

alphas = jnp.sqrt(jnp.linspace(0, 4, 101))
parity, tsave = jax.vmap(
    simulate_phase_flip, in_axes=(None, 0, None, None, None, None)
    )(N, alphas, kappa_2, kappa_1, n_th, t_max)

# %%
popts = []
pstds = []
for ind, (par, ts) in enumerate(zip(parity, tsave)):
    popt, pstd = fit_parity(ts, par)
    popts.append(popt)
    pstds.append(pstd)
popts = jnp.array(popts)
pstds = jnp.array(pstds)
# %%
fig, ax = plt.subplots()
nbars = alphas**2
ax.plot(nbars, popts[:, 0])
ax.plot(nbars, 2*kappa_1*((1+2*n_th)*nbars + n_th), '--', label="2$\\kappa_1((1+2n_{\\rm th})\\bar{n} + n_{\\rm th})$")
ax.axhline(kappa_1*(1+4*n_th), c="k", ls="--", label="$\\kappa_1(1+4n_{\\rm th})$")
ax.axhline(kappa_1*2*n_th, c="grey", ls="--", label="$2\\kappa_1 n_{\\rm th}$")
ax.grid()
ax.set_xlabel("$\\bar{n}\;\\mathrm{[photons]}$")
ax.set_ylabel("$\\Gamma_X$")
ax.legend()
plt.show()


#%%
N = 50
a = dq.destroy(N)
alpha = jnp.linspace(0, jnp.sqrt(4), 101)
cat_p = dq.unit(dq.coherent(N, alpha) + dq.coherent(N, -alpha))
cat_m = dq.unit(dq.coherent(N, alpha) - dq.coherent(N, -alpha))
cat_p_size = dq.expect(dq.dag(a)@a, cat_p)
cat_m_size = dq.expect(dq.dag(a)@a, cat_m)


# %%
plt.plot(alpha**2, cat_p_size, "C0")
plt.plot(alpha**2, alpha**2*jnp.tanh(alpha**2), "C1--")
plt.plot(alpha**2, cat_m_size, 'C2')
plt.plot(alpha**2, alpha**2/jnp.tanh(alpha**2), "C3--")
