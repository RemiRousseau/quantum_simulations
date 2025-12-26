import numpy as np
import scipy.constants as const


def get_thermal_pop(att, Ts, omega):
    att = np.asarray(att)
    Ts = np.asarray(Ts)
    Gs = 10 ** (-att / 10)
    ns = 1 / (np.exp(const.hbar * omega / (const.k * Ts)) - 1)
    nth = 0
    for k in range(len(att)):
        n_loc = (1 - Gs[k]) * np.prod(Gs[k + 1 :]) * ns[k]
        nth += n_loc
    return nth


print(1 / (np.exp(const.hbar * 2 * np.pi * 1e9 / (const.k * 0.02)) - 1))

temps_K = [300, 50, 4, 1, 0.2, 0.02]

# atts_dB = [np.inf, 0, 20, 0, 20, 6]
atts_dB = [np.inf, 0, 20, 6, 10, 10]
omega_Hz = 1e9 * 2 * np.pi
nth = get_thermal_pop(atts_dB, temps_K, omega_Hz)
print("Pump line memory frequency:", nth)

omega_Hz = 7.5e9 * 2 * np.pi
nth = get_thermal_pop(atts_dB, temps_K, omega_Hz)
print("Pump line buffer frequency:", nth)

atts_dB = [np.inf, 0, 20, 0, 20, 30]
omega_Hz = 7.5e9 * 2 * np.pi
nth = get_thermal_pop(atts_dB, temps_K, omega_Hz)
print("Drive line buffer frequency:", nth)
