# %%
import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt

jnp.set_printoptions(precision=3)
# %%

MHz = 2 * jnp.pi
us = 1

omega = 0.002 * MHz
kappa_1 = 1 / (5 * us)
kappa_phi = 1 / (5 * us)

sx = dq.sigmax()
sy = dq.sigmay()
sm = dq.sigmam()
sz = dq.sigmaz()
H = omega * sx
L1 = jnp.sqrt(kappa_1 / 2) * sm
L1u = jnp.sqrt(kappa_1 / 2) * dq.dag(sm)
Lphi = jnp.sqrt(kappa_phi / 2) * sz

times = jnp.linspace(0, 4 * us, 101)

A = kappa_1 / 2
B = kappa_phi + kappa_1 / 2
# C = 1j * omega * jnp.exp(-(2*A+B)*times)*(jnp.exp(B*times) - jnp.exp(2*A*times)) / (2 * A - B)
if 2 * A - B != 0:
    C = -1j * omega * (jnp.exp(-2 * A * times) - jnp.exp(-B * times)) / (2 * A - B)
else:
    print("in")
    C = 1j * omega * times * jnp.exp(-B * times)
U = jnp.array(
    [
        [(1 + jnp.exp(-2 * A * times)) / 2, C, -C, (1 - jnp.exp(-2 * A * times)) / 2],
        [C, jnp.exp(-B * times), 0 * times, -C],
        [-C, 0 * times, jnp.exp(-B * times), C],
        [(1 - jnp.exp(-2 * A * times)) / 2, -C, C, (1 + jnp.exp(-2 * A * times)) / 2],
    ]
)
U = jnp.moveaxis(U, 2, 0)

pp = 1.0
fig, ax = plt.subplots(2)
for p in [pp, 1 - pp]:
    rho_0 = p * dq.fock_dm(2, 1) + (1 - p) * dq.fock_dm(2, 0)
    output = dq.mesolve(H, [L1, L1u, Lphi], rho_0, times)
    print(rho_0.flatten().reshape(-1, 1))
    print(U[-1])

    states_thr = (U @ rho_0.flatten().reshape(-1, 1)).reshape(-1, 2, 2)

    for i, s in enumerate([sy, sz]):
        ax[i].plot(times, dq.expect(s, output.states), label="dynamiqs")
        ax[i].plot(times, dq.expect(s, states_thr), "--", label="analytical")
plt.show()

# fig, ax = plt.subplots(2)
# ax[0].plot(times, output.states[:, 0, 0].real)
# ax[0].plot(times, states_thr[:, 0, 0].real, "--")
# ax[0].plot(times, output.states[:, 1, 1].real)
# ax[0].plot(times, states_thr[:, 1, 1].real, "--")
# ax[1].plot(times, output.states[:, 0, 1].imag)
# ax[1].plot(times, states_thr[:, 0, 1].imag, "--")
# ax[1].plot(times, output.states[:, 1, 0].imag)
# ax[1].plot(times, states_thr[:, 1, 0].imag, "--")
# plt.show()
# %%
nth = 1
omega = 0.5 * MHz
kappa_1 = 1 / (1 * us)
kappa_phi = 1 / (10 * us)

N = 50
IN, I2 = dq.eye(N), dq.eye(2)
sx = dq.tensor(dq.sigmax(), IN)
sy = dq.tensor(dq.sigmay(), IN)
sm = dq.tensor(dq.sigmam(), IN)
sz = dq.tensor(dq.sigmaz(), IN)
a = dq.tensor(I2, dq.destroy(N))
H = omega * (dq.dag(a) @ sm + a @ dq.dag(sm))
L1 = jnp.sqrt(kappa_1 / 2) * sm
L1u = jnp.sqrt(kappa_1 / 2) * dq.dag(sm)
Lphi = jnp.sqrt(kappa_phi / 2) * sz

times = jnp.linspace(0, 4 * us, 1001)
print(omega * times[-1] * jnp.sqrt(nth))
states = []
for p in [0, 1]:
    rho_qb = p * dq.fock_dm(2, 1) + (1 - p) * dq.fock_dm(2, 0)
    # ns = jnp.arange(N)
    # rho_mem = dq.unit(jnp.diag((1 + nth) ** (-1.0) * (nth / (1 + nth)) ** ns)).astype(
    #     complex
    # )
    rho_mem = dq.fock_dm(N, nth)
    # rho_mem = dq.coherent_dm(N, jnp.sqrt(nth))
    rho_0 = dq.tensor(rho_qb, rho_mem)
    output = dq.mesolve(H, [L1, L1u, Lphi], rho_0, times)
    states.append(output.states)

fig, ax = plt.subplots(3)
for i, s in enumerate([sx, sy, sz]):
    ax[i].grid()
    for state in states:
        ax[i].plot(times, dq.expect(s, state), label="dynamiqs")
        # ax[i].plot(times, dq.expect(s, states_thr), "--", label="analytical")
plt.show()
# %%
plt.plot(times, dq.ptrace(states[0], 0, (2, N))[:, 1, 1])
plt.plot(times, 1 / 2 + (1 - 2 * 1) / 2 * jnp.exp(-kappa_1 * times), "--")
# %%
ind_st = 1
plt.plot(times, dq.ptrace(dq.dag(a) @ states[ind_st], 0, (2, N))[:, 0, 1].imag)
plt.plot(times, dq.ptrace(states[ind_st] @ dq.dag(a), 0, (2, N))[:, 0, 1].imag, "--")
plt.plot(times, dq.ptrace(a @ states[ind_st], 0, (2, N))[:, 1, 0].imag)
plt.plot(times, dq.ptrace(states[ind_st] @ a, 0, (2, N))[:, 1, 0].imag, "--")
# plt.plot(times, dq.ptrace(dq.dag(a)@states[1], 0, (2, N))[:, 0, 1].imag)
# plt.plot(times, -dq.ptrace(a@states[1], 0, (2, N))[:, 1, 0].imag, "--")
p = 0
A = kappa_1 / 2
B = kappa_phi + kappa_1 / 2
thr = -(1 - jnp.exp(-B * times)) / B
thr += (jnp.exp(-B * times) - jnp.exp(-2 * A * times)) / (2 * A - B) * (1 - 2 * p)
thr *= omega / 2 * (nth + 1) * 2
# plt.plot(times, thr, '--')
# %%
sp = dq.sigmap()
sm = dq.sigmam()
I2 = dq.eye(2)
dq.tensor(I2, sm)
# %%
