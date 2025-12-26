# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
g2 = 1
xi_b = 4

kappa_a = 5e-3
kappa_b = 20

deltas_a = np.linspace(-1, 1, 101) * kappa_a * 20
deltas_b = np.linspace(-1, 1, 101) * kappa_b * 20

da, db = np.meshgrid(deltas_a, deltas_b, indexing="ij")
z = (1j * kappa_a / 2 + da) * (1j * kappa_b / 2 + db) / 2 / g2**2

alpha_2 = np.real(z) + np.sqrt(xi_b**2 / g2**2 - np.imag(z) ** 2)
alpha_2[xi_b**2 / g2**2 - np.imag(z) ** 2 < 0] = 0
alpha_2[alpha_2 < 0] = 0
plt.imshow(
    alpha_2,
    origin="lower",
    extent=[deltas_a[0], deltas_a[-1], deltas_b[0], deltas_b[-1]],
    interpolation="none",
    aspect="auto",
)
plt.colorbar()
plt.show()

plt.plot(np.diag(alpha_2))
plt.plot(np.diag(alpha_2[::-1]))
