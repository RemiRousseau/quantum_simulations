import string
from functools import partial

import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import optax
from IPython.display import clear_output
from jax.typing import ArrayLike
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm


class Simulation:
    cut_offs: tuple[int]
    hamiltonian_terms: dict[str:callable]
    jump_operators: dict[str:callable]
    pulse_functions: dict[str:callable]

    rho_zeros: ArrayLike

    tpulse: ArrayLike
    tsave: ArrayLike

    to_optimize: list[str]
    pulse_params: dict[str, ArrayLike]
    cost_function: callable

    last_states: ArrayLike = None

    params_history: list[dict[str, ArrayLike]]
    costs: list[float] = []

    ind_train: int = 0
    N_optimize: int

    exp_plot: list[ArrayLike]
    _expects: dict[str, ArrayLike] = None

    optimizer: optax.GradientTransformation

    def __init__(
        self,
        cut_offs: tuple[int] | int,
        hamiltonian_terms: dict[str:callable],
        jump_operators: dict[str:callable],
        pulse_functions: dict[str:callable],
        rho_zeros: ArrayLike,
        tpulse: ArrayLike,
        tsave: ArrayLike,
        to_optimize: list[str],
        pulse_params: dict[str, ArrayLike],
        cost_function: callable,
        N_optimize: int,
        exp_plot: list[ArrayLike] = [],
        optimizer: optax.GradientTransformation = optax.adam(1e-2),
    ) -> None:
        all_labels = list(hamiltonian_terms.keys()) + list(jump_operators.keys())
        assert all([label in pulse_functions for label in all_labels])
        assert all([label in pulse_params for label in all_labels])
        assert all([label in all_labels for label in to_optimize])

        self.cut_offs = (cut_offs,) if isinstance(cut_offs, int) else cut_offs
        self.hamiltonian_terms = hamiltonian_terms
        self.jump_operators = jump_operators
        self.pulse_functions = pulse_functions
        self.rho_zeros = rho_zeros
        self.tpulse = tpulse
        self.tsave = tsave
        self.to_optimize = to_optimize
        self.pulse_params = pulse_params
        self.params_history = [self.pulse_params.copy()]
        self.cost_function = cost_function
        self.N_optimize = N_optimize
        self.exp_plot = exp_plot
        self.optimizer = optimizer

        self._to_optimize = {key: self.pulse_params[key] for key in self.to_optimize}
        self._optimizer_state = self.optimizer.init(self._to_optimize)

    @property
    def pulses(self) -> dict[str, ArrayLike]:
        return {
            key: self.pulse_functions[key](self.tpulse, self.pulse_params[key])
            for key in self.pulse_params
        }

    @property
    def hamiltonian(self) -> ArrayLike:
        H = dq.eye(*self.cut_offs) * 0
        for key, pulse in self.pulses.items():
            if key in self.hamiltonian_terms:
                H += self.hamiltonian_terms[key](self.tpulse, pulse)
        return H

    @property
    def dissipator(self) -> ArrayLike:
        L = []
        for key, pulse in self.pulses.items():
            if key in self.jump_operators:
                L.append(self.jump_operators[key](self.tpulse, pulse))
        return L

    @property
    def expects(self) -> dict[str, ArrayLike]:
        if self.last_states is None:
            self.simulate()
        if self._expects is None:
            self._expects = {}
            for key, op in self.exp_plot.items():
                self._expects[key] = dq.expect(op, self.last_states)
        return self._expects

    def simulate(self, progress_bar: bool = True) -> ArrayLike:
        self.last_states = dq.mesolve(
            self.hamiltonian,
            self.dissipator,
            self.rho_zeros,
            self.tsave,
            options=(
                dq.Options(progress_bar=dq.NoProgressMeter())
                if not progress_bar
                else dq.Options()
            ),
        ).states

    def _cost_function(self, _to_optimize: dict[str, ArrayLike]) -> float:
        self.pulse_params.update(_to_optimize)
        self.simulate(progress_bar=False)
        return self.cost_function(self.tsave, self.last_states, self.pulses)

    def optimize(self, gradient_clip: float = None) -> None:
        for loc_ind_train in range(self.N_optimize):
            cost, cost_grad = jax.value_and_grad(self._cost_function)(self._to_optimize)
            if gradient_clip is not None:
                cost_grad = {
                    key: jnp.clip(grad, -gradient_clip, gradient_clip)
                    for key, grad in cost_grad.items()
                }
            cost_grad = {key: -grad for key, grad in cost_grad.items()}
            updates, self._optimizer_state = self.optimizer.update(
                cost_grad, self._optimizer_state
            )
            self._to_optimize = optax.apply_updates(self._to_optimize, updates)
            self.pulse_params.update(self._to_optimize.copy())
            self.costs.append(cost)
            self.params_history.append(self.pulse_params.copy())
            self.ind_train += 1

            clear_output(wait=True)
            fig, _ = self.plot_evolution(plot_expect=False, plot_errors=False)
            fig.suptitle(
                f"Epoch: {loc_ind_train + 1}/{self.N_optimize} | Cost: {cost:.2f}"
            )
            plt.show()
        self.simulate(progress_bar=False)

    def _fig_plot(
        self,
        plot_pulses: bool = True,
        plot_expect: bool = True,
        plot_errors: bool = True,
        plot_cost: bool = True,
    ) -> tuple[Figure, Axes]:
        n_plot = len(self.to_optimize) * plot_pulses
        n_plot += len(self.exp_plot) * plot_expect
        n_plot += len(self.cut_offs) * plot_errors
        n_plot += plot_cost
        fig, axes = plt.subplots(n_plot, 1, figsize=(5, 2 * n_plot))
        if len(self.to_optimize) == 1:
            axes = np.array([axes])
        return fig, axes

    def _sub_plot_pulses(self, axes: list[Axes]):
        for key, ax in zip(self.to_optimize, axes.flatten()):
            ax.bar(
                self.tpulse[:-1],
                self.pulses[key],
                self.tpulse[1],
                align='edge',
                edgecolor="#507fe3",
                facecolor="#9db5e9",
            )
            ax.set_ylabel('Amplitude')
            ax.set_title(key)
            ax.set_xlabel('Time')
            ax.grid()

    def _sub_plot_expects(self, axes: list[Axes]):
        for key, ax in zip(self.expects, axes.flatten()):
            for ind in range(len(self.rho_zeros)):
                ax.plot(
                    self.tsave,
                    np.real(self.expects[key][ind]),
                    f"C{ind}",
                    label=f"Rho_{ind}",
                )
                imag = np.imag(self.expects[key][ind])
                if imag.any():
                    ax.plot(self.tsave, imag, f"C{ind}--")
            ax.legend()
            ax.set_xlabel('Time')
            ax.grid()

    def _sub_plot_errors(self, axes: list[Axes]):
        operators = dq.destroy(*self.cut_offs)
        for op, name, ax in zip(operators, string.ascii_lowercase, axes.flatten()):
            error_op = op @ dq.dag(op) - dq.dag(op) @ op
            error = np.abs(dq.expect(error_op, self.last_states) - 1)
            for ind in range(len(self.rho_zeros)):
                ax.plot(self.tsave, error[ind], f"C{ind}", label=f"Rho_{ind}")
            ax.legend()
            ax.grid()
            ax.set_xlabel('Time')
            ax.set_ylabel('Error')
            ax.set_title(f"Error {name}")

    def _sub_plot_cost(self, ax: Axes) -> None:
        ax.plot(self.costs)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.grid()

    def plot_evolution(
        self,
        plot_pulses: bool = True,
        plot_expect: bool = True,
        plot_errors: bool = True,
        plot_cost: bool = True,
    ) -> tuple[Figure, list[Axes]]:
        if len(self.costs) == 0:
            plot_cost = False
        fig, axes = self._fig_plot(plot_pulses, plot_expect, plot_errors, plot_cost)
        ind_0 = 0
        if plot_pulses:
            self._sub_plot_pulses(axes.flatten()[: len(self.to_optimize)])
            ind_0 += len(self.to_optimize)
        if plot_expect:
            self._sub_plot_expects(axes.flatten()[ind_0 : ind_0 + len(self.exp_plot)])
            ind_0 += len(self.exp_plot)
        if plot_errors:
            self._sub_plot_errors(axes.flatten()[ind_0 : ind_0 + len(self.cut_offs)])
            ind_0 += len(self.cut_offs)
        if plot_cost:
            self._sub_plot_cost(axes.flatten()[-1])
        suptitle = f"Epoch: {self.ind_train}"
        suptitle += f" | Cost: {self.costs[-1]:.2f}" if len(self.costs) != 0 else ""
        fig.suptitle(suptitle)
        fig.tight_layout()
        return fig, axes

    def plot_wigner_gif(
        self, ind_el: tuple[int] | int = None, sub_system: int = None, xmax: float = 5
    ):
        if ind_el is None:
            states = self.last_states
        else:
            states = self.last_states[ind_el]
        sub_states = dq.ptrace(states, sub_system, self.cut_offs)
        dq.plot_wigner_gif(sub_states, xmax=xmax)
