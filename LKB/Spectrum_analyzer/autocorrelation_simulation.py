import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

N = 100
delta = 1
kappa = 0.05
nth = 4
tmax = 40

a = qt.destroy(N)
x = (a.dag() + a) / np.sqrt(2)
p = 1j * (a.dag() - a) / np.sqrt(2)

A = a.dag()
B = a
H = delta * a.dag() * a
L = (kappa * (nth + 1)) ** 0.5 * a
Lu = (kappa * nth) ** 0.5 * a.dag()
time = np.linspace(0, tmax, 1000)

rho = qt.thermal_dm(N, nth)
output = qt.mesolve(H, A * rho, time, [L, Lu])

plt.plot(time, qt.expect(B, output.states).real)
plt.plot(time, (nth + 1) * np.exp(-kappa / 2 * time) * np.cos(delta * time), "--")
plt.plot(time, qt.expect(B, output.states).imag)
plt.plot(time, -(nth + 1) * np.exp(-kappa / 2 * time) * np.sin(delta * time), "--")
plt.show()

A = a
B = a.dag()
H = delta * a.dag() * a
L = (kappa * (nth + 1)) ** 0.5 * a
Lu = (kappa * nth) ** 0.5 * a.dag()
time = np.linspace(0, tmax, 1000)

rho = qt.thermal_dm(N, nth)
output = qt.mesolve(H, A * rho, time, [L, Lu])

plt.plot(time, qt.expect(B, output.states).real)
plt.plot(time, nth * np.exp(-kappa / 2 * time) * np.cos(delta * time), "--")
plt.plot(time, qt.expect(B, output.states).imag)
plt.plot(time, nth * np.exp(-kappa / 2 * time) * np.sin(delta * time), "--")
plt.show()
