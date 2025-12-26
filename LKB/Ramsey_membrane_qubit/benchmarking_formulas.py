# %%
import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt
from utils import lab_frame, rotating_displaced_frame, rotating_frame, set_free_gpu

set_free_gpu()

MHz = 2 * jnp.pi
kHz = 1e-3 * MHz
Hz = 1e-6 * MHz
us = 1

# %%
N = 100
wm = 4.4 * MHz
wq = 2.4 * MHz
g = 0.5 * kHz
kappa_1 = 1 / (30 * us)
kappa_2 = 1 / (15 * us)
kappa_mem = 24 * Hz
nth = 0
alpha0 = jnp.sqrt(50)
t_max = 20 * us
Nt = 10_000

output = lab_frame(
    N, wm, wq, g, kappa_1, kappa_2, kappa_mem, nth, alpha0, t_max, Nt, save_states=True
)
output_rot = rotating_frame(
    N, wm, wq, g, kappa_1, kappa_2, kappa_mem, nth, alpha0, t_max, Nt, save_states=True
)
output_rot_disp = rotating_displaced_frame(
    N, wm, wq, g, kappa_1, kappa_2, kappa_mem, nth, alpha0, t_max, Nt, save_states=True
)
# %%
expects = output.expects
expects_rot = output_rot.expects
expects_rot_disp = output_rot_disp.expects
fig, ax = plt.subplots(5)
for i, ax in enumerate(ax):
    ax.plot(expects[i])
    ax.plot(expects_rot[i], "--")
    ax.plot(expects_rot_disp[i], ":")
# %%
Cv = expects[0] + 1j * expects[1]
times = jnp.linspace(0, t_max, Nt)
derot = Cv * jnp.exp(-1j * wq * times)
plt.plot(times, jnp.real(derot), label="Re")
plt.plot(times, jnp.imag(derot), label="Im")
plt.plot(times, expects_rot[0], "--", label="Re")
plt.plot(times, expects_rot[1], "--", label="Im")
plt.plot(times, expects_rot_disp[0], ":", label="Re")
plt.plot(times, expects_rot_disp[1], ":", label="Im")
plt.show()
plt.plot(times, jnp.real(expects[3] * jnp.exp(1j * wm * times)), label="Re")
plt.plot(times, jnp.imag(expects[3] * jnp.exp(1j * wm * times)), label="Re")
plt.plot(times, jnp.real(expects_rot[3]), "--", label="Re")
plt.plot(times, jnp.imag(expects_rot[3]), "--", label="Re")
plt.plot(times, jnp.real(expects_rot_disp[3]), ":", label="Re")
plt.plot(times, jnp.imag(expects_rot_disp[3]), ":", label="Re")
plt.show()
# %%
dq.plot.wigner_gif(dq.ptrace(output.states, 1, (2, N)))
dq.plot.wigner_gif(dq.ptrace(output_rot.states, 1, (2, N)))
dq.plot.wigner_gif(dq.ptrace(output_rot_disp.states, 1, (2, N)))
# %%
