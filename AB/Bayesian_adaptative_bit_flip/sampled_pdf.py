import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.typing import ArrayLike
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class SampledPDF:
    def __init__(
        self, prior: ArrayLike, a_param: float = 0.98, resample_threshold: float = 0.5
    ) -> None:
        self.samples = jnp.asarray(prior)
        self.n_samples = self.samples.shape[1]
        self.n_dims = self.samples.shape[0]
        self._sample_indices = jnp.arange(self.n_samples)
        self.sample_weights = jnp.ones(self.n_samples) / self.n_samples
        self.resample_threshold = resample_threshold
        self.key = jax.random.PRNGKey(0)
        self.a_param = a_param

    def mean(self) -> jax.Array:
        return jnp.sum(self.samples * self.sample_weights, axis=1)

    def covariance(self) -> jax.Array:
        cov = jnp.cov(self.samples, aweights=self.sample_weights)
        if self.n_dims == 1:
            return cov.reshape((1, 1))
        else:
            return cov

    def std(self) -> jax.Array:
        mean = self.mean()
        msq = jnp.sum(self.samples**2 * self.sample_weights, axis=1)
        return jnp.sqrt(msq - mean**2)

    def bayesian_update(self, likelihood: ArrayLike) -> jax.Array:
        tmp = jnp.nan_to_num(self.sample_weights * likelihood)
        self.sample_weights = jnp.nan_to_num(tmp / jnp.sum(tmp))
        self.resample_test()

    def resample_test(self) -> None:
        ess = 1 / jnp.sum(jnp.nan_to_num(self.sample_weights**2))
        if ess < self.n_samples * self.resample_threshold:
            print("Resampling")
            self.resample()

    def resample(self) -> None:
        self.key, subkey = jax.random.split(self.key)
        choices = jax.random.choice(
            subkey,
            self._sample_indices,
            shape=(self.n_samples,),
            replace=False,
            p=self.sample_weights,
        )
        covar = self.covariance()
        old_center = self.mean().reshape((self.n_dims, 1))
        newcovar = (1 - self.a_param**2) * covar

        self.key, subkey = jax.random.split(self.key)
        shift = jax.random.multivariate_normal(
            subkey, jnp.zeros(self.n_dims), newcovar, self.n_particles
        ).T
        self.samples = self.samples[:, choices] + shift
        self.samples = self.samples * self.a_param + old_center * (1 - self.a_param)

    def plot(self, fig: Figure = None, ax: Axes = None, nbin: int = 51) -> None:
        n_plots = self.n_dims if self.n_dims != 2 else 1
        if fig is None and ax is None:
            fig = plt.figure(figsize=(5 * n_plots, 3))
        if ax is None:
            ax = fig.subplots(1, n_plots)
        elif np.asarray([ax]).flatten() != n_plots:
            raise ValueError("Number of axes must match number of dimensions")
        if self.n_dims != 2:
            for i, ax in enumerate(np.asarray([ax]).flatten()):
                ax.hist(self.samples[i], bins=nbin, weights=self.sample_weights)
                ax.set_title(f"Dimension {i}")
        else:
            _, _, _, im = ax.hist2d(
                self.samples[0],
                self.samples[1],
                bins=nbin,
                weights=self.sample_weights,
                density=True,
            )
            fig.colorbar(im, ax=ax, label='Probability density')
            ax.set_title("2D histogram")
