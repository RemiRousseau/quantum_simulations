# %%
from functools import partial

import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm


def parity_preserving_ops(max_order, N):
    assert max_order >= 2
    ops = [dq.eye(N)]
    a, a_dag = dq.destroy(N), dq.create(N)
    for order in range(1, max_order // 2 + 1):
        for k in range(order * 2 + 1):
            ops.append(
                jnp.linalg.matrix_power(a_dag, k)
                @ jnp.linalg.matrix_power(a, 2 * order - k)
            )
    return ops


def dissipator(coefs, ops):
    coefs_full = jnp.insert(coefs, 1, 1.0)
    return sum([coefs_full[i] * ops[i] for i in range(len(ops))])


# %%
N = 40
alpha = 2
max_order = 2

a = dq.destroy(N)
D, P = jnp.linalg.eigh(dq.dag(a) + a)
signX = P @ jnp.diag(jnp.sign(D)) @ dq.dag(P)

c_ops = [jnp.sqrt(0.1) * a]

ops = parity_preserving_ops(max_order, N)
coefs = jnp.zeros(len(ops) - 1, dtype=complex)
coefs = coefs.at[0].set(-(alpha**2))


@partial(jax.jit, static_argnums=(1, 2))
def cost(coefs: jax.Array, lam: float, show_trace: bool = False) -> float:
    Lp = dissipator(coefs, ops)
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
coefs = jnp.zeros(len(ops) - 1, dtype=complex)
coefs = coefs.at[0].set(-(alpha**2))
# %%
lam = 1
learning_rate = 1e-2
costs = []

# %%
n_iter = 100
with tqdm(total=n_iter) as pbar:
    for _ in range(n_iter):
        v, grad = jax.value_and_grad(cost)(coefs, lam)
        costs.append(v)
        coefs = coefs - grad * learning_rate
        pbar.set_postfix({"Loss": f"{v:.4f}"})
        pbar.update(1)
# %%
plt.plot(costs)
plt.show()
# %%
Lp = a @ a - alpha**2 * dq.eye(N)
_, eigst = jnp.linalg.eigh(dq.dag(Lp) @ Lp)
cat_p = eigst[:, 0].reshape(-1, 1)
cat_m = eigst[:, 1].reshape(-1, 1)
dq.plot.wigner(dq.unit(cat_p))
dq.plot.wigner(dq.unit(cat_m))
dq.plot.wigner(dq.unit(cat_p + cat_m))

# %%
# Second try with lindbladian diagonalization
dq.set_precision("double")
N0, N1 = 200, 30
alpha = 2
max_order = 2

a = dq.destroy(N0)
c_ops = [jnp.sqrt(0.01) * a]

ops = parity_preserving_ops(max_order, N0)
coefs = jnp.zeros(len(ops) - 1, dtype=complex)
coefs = coefs.at[0].set(-(alpha**2))


@partial(jax.jit, static_argnums=(1))
def cost(coefs: jax.Array, lam: float) -> float:
    Lp = dissipator(coefs, ops)
    _, U = jnp.linalg.eigh(dq.dag(Lp) @ Lp)
    U = U[:, :N1]
    cat_p = U[:, 0].reshape(-1, 1)
    nb = dq.expect(dq.dag(a) @ a, cat_p).real

    c_ops_basis = [dq.dag(U) @ c_op @ U for c_op in c_ops + [Lp]]
    Lind = dq.slindbladian(dq.zero(N1), c_ops_basis)
    eigvals = jnp.linalg.eigvals(Lind)
    indsort = jnp.argsort(eigvals.real)
    eigvals = eigvals[indsort][::-1]
    Tbf = 1 / jnp.abs(eigvals[1])
    cost = -jnp.log(Tbf)
    return cost + lam * (nb - alpha**2) ** 2


# %%
coefs = jnp.zeros(len(ops) - 1, dtype=complex)
coefs = coefs.at[0].set(-(alpha**2))
print(jnp.exp(-cost(coefs, 0)))
# %%
lam = 1
learning_rate = 1e-3
costs = []

# %%
n_iter = 100
with tqdm(total=n_iter) as pbar:
    for _ in range(n_iter):
        v, grad = jax.value_and_grad(cost)(coefs, lam)
        costs.append(v)
        coefs = coefs - grad * learning_rate
        pbar.set_postfix({"Loss": f"{v:.4f}"})
        pbar.update(1)
# %%
print(coefs)
plt.plot(costs)
plt.show()
# %%
Lp = dissipator(coefs, ops)
_, eigst = jnp.linalg.eigh(dq.dag(Lp) @ Lp)
cat_p = eigst[:, 0].reshape(-1, 1)
cat_m = eigst[:, 1].reshape(-1, 1)
dq.plot.wigner(dq.unit(cat_p))
dq.plot.wigner(dq.unit(cat_m))
dq.plot.wigner(dq.unit(cat_p + cat_m))

# %%
