# %%
import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm


# %%
def get_moon(N, alpha, lam):
    a = dq.destroy(N)
    L2 = a @ a - alpha**2 * dq.eye(N) + lam * (dq.dag(a) @ a - alpha**2 * dq.eye(N))
    Hk = dq.dag(L2) @ L2
    _, eigst = jnp.linalg.eigh(Hk)
    parity = dq.parity(N)
    if dq.expect(parity, eigst[:, 0].reshape(-1, 1)).real > 0:
        cp, cm = eigst[:, :2].T
    else:
        cm, cp = eigst[:, :2].T
    return cp.reshape(-1, 1), cm.reshape(-1, 1)


def get_moon_nb(N, nbar, lam):
    a = dq.destroy(N)

    def local_cost(alpha):
        moon = get_moon(N, alpha, lam)[0]
        return (dq.expect(dq.dag(a) @ a, moon).real - nbar) ** 2

    res = minimize(
        local_cost, jnp.sqrt(nbar), bounds=((0.0, 10.0),), jac=jax.jacobian(local_cost)
    )
    return get_moon(N, res.x[0], lam), res.x[0]


def get_log(cp, cm):
    N = cp.size
    l0 = dq.unit(cp + cm)
    l1 = dq.unit(cp - cm)
    if dq.expect(dq.destroy(N), l0).real < 0:
        l0, l1 = l1, l0
    return l0, l1


# %%
N = 70
nbar = 4
lams = jnp.linspace(0, 1.2, 201)

x = dq.position(N)
y = dq.momentum(N)
means_x, sigmas_x, means_y, sigmas_y = [], [], [], []
for lam in tqdm(lams):
    (cp, cm), res = get_moon_nb(N, nbar, lam)
    l0, l1 = get_log(cp, cm)
    mean_x = dq.expect(x, l0).real
    sigma_x = jnp.sqrt(dq.expect(x @ x, l0).real - mean_x**2)
    mean_y = dq.expect(y, l0).real
    sigma_y = jnp.sqrt(dq.expect(y @ y, l0).real - mean_y**2)
    means_x.append(mean_x)
    sigmas_x.append(sigma_x)
    means_y.append(mean_y)
    sigmas_y.append(sigma_y)
    if lam == lams[-1]:
        dq.plot.wigner(l0)
        plt.show()
sigmas_x, sigmas_y = jnp.array(sigmas_x), jnp.array(sigmas_y)

# %%
fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax[0, 0].plot(lams, means_x)
ax[0, 0].set_title("Mean x")
ax[0, 1].plot(lams, sigmas_x)
ax[0, 1].set_title("Sigma x")
ax[1, 0].plot(lams, means_y)
ax[1, 0].set_title("Mean y")
ax[1, 1].plot(lams, sigmas_y)
ax[1, 1].set_title("Sigma y")
plt.show()

fig, ax = plt.subplots()
dBs = -10 * jnp.log10(sigmas_x**2 / sigmas_x[0] ** 2)
ax.plot(lams, dBs)
max_order = 3
X = jnp.column_stack([lams**k for k in range(1, max_order + 1)])
a = jnp.linalg.inv(X.T @ X) @ X.T @ dBs
print(a)
ax.plot(lams, X @ a, "--")
ax.grid()
ax.set_xlabel("Lambda")
ax.set_ylabel("dB")
plt.show()

jnp.save("lms_to_db.npy", jnp.array([0, *a]))
# %%
X = jnp.column_stack([lams**k for k in range(max_order + 1)])
a = jnp.load("lms_to_db.npy")
plt.plot(lams, dBs)
plt.plot(lams, X @ a, "--")
# %%
X = jnp.array([0.8**k for k in range(max_order + 1)])
print(X @ a)

# %%
