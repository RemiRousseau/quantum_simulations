# %%
import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt
import qutip as qt

kHz = 2 * jnp.pi
ms = 1
us = 1e-3

# %% ================= No interaction =================
sm = dq.sigmam()
sp = dq.sigmap()
sx = dq.sigmax()
sy = dq.sigmay()
sz = dq.sigmaz()

g = 1 * kHz
t_max = 1 * ms
ket_0 = dq.fock(2, 1)

times = jnp.linspace(0, t_max, 100)
H = 1j * g * (sp - sm)
output = dq.sesolve(H, ket_0, times)

# %%
plt.plot(times, dq.expect(sx, output.states).real, label="x")
plt.plot(times, dq.expect(sy, output.states).real, label="y")
plt.plot(times, dq.expect(sz, output.states).real, label="z")
# %%
dq.tensor(dq.eye(2), sp - sm)
# %% ================= Interaction =================
N = 100
IN = dq.eye(N)
sm = dq.tensor(IN, dq.sigmam())
sp = dq.tensor(IN, dq.sigmap())
sx = dq.tensor(IN, dq.sigmax())
sy = dq.tensor(IN, dq.sigmay())
sz = dq.tensor(IN, dq.sigmaz())
a = dq.tensor(dq.destroy(N), dq.eye(2))

g = 1 * kHz
alpha_0 = 5
qb_state = dq.fock(2, 1)
tmax = 0.1 * ms

times = jnp.linspace(0, tmax, 100)
H = 1j * g * (a @ sp - a.dag() @ sm)
ket_0 = dq.tensor(dq.coherent(N, alpha_0), qb_state)
output = dq.sesolve(H, ket_0, times)

theta = g * times
ana_state = []
a = dq.destroy(N)
for t in times:
    theta = g * t
    U = dq.tensor(IN - theta**2 / 2 * a @ a.dag(), jnp.array([[1, 0], [0, 0]]))
    U += dq.tensor(IN - theta**2 / 2 * a.dag() @ a, jnp.array([[0, 0], [0, 1]]))
    U += dq.tensor(theta * a, jnp.array([[0, 1], [0, 0]]))
    U += dq.tensor(-theta * a.dag(), jnp.array([[0, 0], [1, 0]]))
    ana_state.append(U @ ket_0)
# %%
plt.plot(times, dq.expect(sx, output.states).real, label="x")
plt.plot(times, dq.expect(sy, output.states).real, label="y")
plt.plot(times, dq.expect(sz, output.states).real, label="z")
plt.plot(times, dq.expect(sx, ana_state).real, "k--", label="x")
plt.plot(times, dq.expect(sy, ana_state).real, "k--", label="y")
plt.plot(times, dq.expect(sz, ana_state).real, "k--", label="z")
plt.ylim(-1, 1)
plt.legend()
plt.show()

# %% ============== PDF of sigma x ==============
N = 100
IN = dq.eye(N)
sm = dq.tensor(IN, dq.sigmam())
sp = dq.tensor(IN, dq.sigmap())
sx = dq.tensor(IN, dq.sigmax())
sy = dq.tensor(IN, dq.sigmay())
sz = dq.tensor(IN, dq.sigmaz())
a = dq.tensor(dq.destroy(N), dq.eye(2))

sxss, syss, szss = [], [], []
nbars = jnp.linspace(0, 5, 101)
for iqb, qb_state in enumerate([dq.fock(2, 1), dq.fock(2, 0)]):
    loc_sxs, loc_sys, loc_szs = [], [], []
    for ialpha, alpha in enumerate(jnp.sqrt(nbars)):
        g = 0.5 * kHz
        nth = 10
        rho_mem = dq.coherent_dm(N, alpha)
        tmax = 20 * us

        times = jnp.linspace(0, tmax, 100)
        H = 1j * g * (a @ sp - a.dag() @ sm)
        ket_0 = dq.tensor(rho_mem, qb_state @ qb_state.dag())
        output = dq.mesolve(H, [], ket_0, times)
        qb_rho = dq.ptrace(output.states[-1], 1, (N, 2))
        sx, sy, sz = dq.sigmax(), dq.sigmay(), dq.sigmaz()
        sxv, syv, szv = (
            dq.expect(sx, qb_rho),
            dq.expect(sy, qb_rho),
            dq.expect(sz, qb_rho),
        )
        loc_sxs.append(sxv)
        loc_sys.append(syv)
        loc_szs.append(szv)
    sxss.append(loc_sxs)
    syss.append(loc_sys)
    szss.append(loc_szs)
# %%
plt.plot(nbars, sxss[0])
plt.plot(nbars, -jnp.array(sxss[1]))
plt.show()
plt.plot(nbars, szss[0])
plt.plot(nbars, -jnp.array(szss[1]))

# %%
# rho_qb = qt.Qobj(qb_ket.asdense().data)
rho_qb = (qt.fock_dm(2, 1) * 0.0 + qt.fock_dm(2, 0)).unit()
# rho_qb = qt.ket2dm((qt.fock(2, 1) + qt.fock(2, 0)).unit())
theta = jnp.linspace(0, jnp.pi, 101)
phi = jnp.linspace(0, 2 * jnp.pi, 101)
spn_wig, thetas, phis = qt.spin_q_function(rho_qb, theta, phi)
# %%
plt.imshow(spn_wig, extent=[theta[0], theta[-1], phi[0], phi[-1]])
plt.colorbar()
# %%

fig, ax = plt.subplots()
for p in jnp.linspace(0, 1, 11):
    rho_qb = qt.fock_dm(2, 1) * p + qt.fock_dm(2, 0) * (1 - p)
    spn_wig, thetas, phis = qt.spin_q_function(rho_qb, theta, [0])
    spn_wig, thetas = spn_wig[0, :], thetas[0]
    intg = jnp.sum(spn_wig) * (theta[1] - theta[0])
    ax.plot(thetas, spn_wig, label=f"{intg}")
ax.legend()
# %%
