# %%
import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
N = 30
alpha = 2
a = dq.destroy(N)
L = a @ a - alpha**2 * dq.eye(N)
D, P = jnp.linalg.eigh(dq.dag(a) + a)
signX = P @ jnp.diag(jnp.sign(D)) @ dq.dag(P)

kappa_1 = 1
c_ops = [jnp.sqrt(kappa_1) * a]


def cost(U: jax.Array, lam: float, show_trace: bool = False) -> float:
    Lp = U @ L @ dq.dag(U)
    _, eigst = jnp.linalg.eigh(dq.dag(Lp) @ Lp)
    cat_p = eigst[:, 0].reshape(-1, 1)
    cat_m = eigst[:, 1].reshape(-1, 1)
    log0 = dq.unit(cat_p + cat_m)
    output = dq.mesolve(
        dq.zero(N),
        c_ops + [Lp],
        log0,
        jnp.linspace(0, 20, 100),
        exp_ops=[signX],
        options=dq.Options() if show_trace else dq.Options(progress_meter=None),
    )
    if show_trace:
        plt.plot(output.expects[0])
        plt.show()
    cost = (output.expects[0][-1] / output.expects[0][1]).real
    nb = dq.expect(dq.dag(a) @ a, cat_p).real
    return jnp.log(1 - cost) + lam * (nb - alpha**2) ** 2


# %%
H_guess = dq.zero(N)
# H_guess = 1e-3j * (dq.dag(a) @ dq.dag(a) - a @ a)
U = dq.expm(1j * H_guess)
print(cost(U, 0.0, True))
# %%
lam = 0.1
learning_rate = 1e-3
n_iter = 100
costs = []

# %%
with tqdm(total=n_iter) as pbar:
    for _ in range(n_iter):
        v, grad = jax.value_and_grad(cost)(U, lam)
        costs.append(v)
        A = grad - U @ dq.dag(grad) @ U
        U = U @ dq.expm(-learning_rate * A)
        u, s, vh = jnp.linalg.svd(U)
        U = u @ vh
        pbar.set_postfix({"Loss": f"{v:.4f}"})
        pbar.update(1)
# %%
plt.plot(costs)
plt.show()
# %%
Lp = U @ L @ dq.dag(U)
_, eigst = jnp.linalg.eigh(dq.dag(Lp) @ Lp)
cat_p = eigst[:, 0].reshape(-1, 1)
cat_m = eigst[:, 1].reshape(-1, 1)
dq.plot.wigner(dq.unit(cat_p))
dq.plot.wigner(dq.unit(cat_m))
dq.plot.wigner(dq.unit(cat_p + cat_m))
