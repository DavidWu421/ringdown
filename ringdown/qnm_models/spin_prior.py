"""Configurable spin (chi) prior for qnm models.

Provides a mixin that lets any qnm model class sample `chi` from either
the default uniform prior, or a named/custom 1D prior specified by a
tabulated density/KDE (e.g., an NR-surrogate-derived posterior on the
remnant spin under uniform progenitor priors).
"""

__all__ = [
    "SpinPriorMixin",
    "load_chi_prior_from_file",
    "REGISTERED_CHI_PRIORS",
]

from pathlib import Path
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import Transform
from numpyro.distributions import constraints

# -----------------------------------------------------------------------
# Built-in priors bundled with the package: name -> path, resolved relative
# to this file so they work regardless of install location.
# -----------------------------------------------------------------------
_PRIORS_DIR = Path(__file__).parent / "priors"

REGISTERED_CHI_PRIORS = {
    "nrsur": _PRIORS_DIR / "remnant_spin_kde.txt",
}


class _MonotonicInverseCDFTransform(Transform):
    """Maps u ~ Uniform(0,1) -> x via a monotonic inverse-CDF given as a
    tabulated grid (u_grid, x_grid), both increasing."""

    domain = constraints.unit_interval
    codomain = constraints.real

    def __init__(self, u_grid, x_grid):
        self.u_grid = jnp.asarray(u_grid)
        self.x_grid = jnp.asarray(x_grid)

    def __call__(self, u):
        return jnp.interp(u, self.u_grid, self.x_grid)

    def _inverse(self, x):
        return jnp.interp(x, self.x_grid, self.u_grid)

    def log_abs_det_jacobian(self, u, x, intermediates=None):
        eps = 1e-5
        u_hi = jnp.clip(u + eps, 0.0, 1.0)
        u_lo = jnp.clip(u - eps, 0.0, 1.0)
        x_hi = jnp.interp(u_hi, self.u_grid, self.x_grid)
        x_lo = jnp.interp(u_lo, self.u_grid, self.x_grid)
        slope = (x_hi - x_lo) / (u_hi - u_lo)
        return jnp.log(jnp.abs(slope) + 1e-300)

    def tree_flatten(self):
        return (self.u_grid, self.x_grid), (("u_grid", "x_grid"), dict())


def load_chi_prior_from_file(
    path_or_name: str,
    n_grid: int = 1000,
    x_min: float = 0.0,
    x_max: float = 1.0,
):
    """Build a numpyro distribution for chi from a tabulated KDE file.

    `path_or_name` can be:
      - the name of a built-in prior bundled with the package, e.g. 'nrsur'
        (see `REGISTERED_CHI_PRIORS`); or
      - a path to a text file with two columns (chi, pdf); or
      - a path to a file with a single column of chi samples, which will
        be KDE'd with `scipy.stats.gaussian_kde` and evaluated on a grid
        over [x_min, x_max].

    Returns
    -------
    prior : numpyro.distributions.TransformedDistribution
    """
    if path_or_name in REGISTERED_CHI_PRIORS:
        path = REGISTERED_CHI_PRIORS[path_or_name]
    else:
        path = Path(path_or_name)

    if not Path(path).exists():
        raise FileNotFoundError(
            f"chi prior file not found: {path} (from '{path_or_name}'); "
            f"known built-in priors: {list(REGISTERED_CHI_PRIORS)}"
        )

    data = np.atleast_2d(np.loadtxt(path))
    if data.shape[0] == 1 and data.shape[1] > 2:
        data = data.T

    if data.shape[1] >= 2:
        # (chi, pdf) grid provided directly
        x, p = data[:, 0], data[:, 1]
        order = np.argsort(x)
        x, p = x[order], np.clip(p[order], 0, None)
    else:
        from scipy.stats import gaussian_kde

        samples = data[:, 0]
        kde = gaussian_kde(samples)
        x = np.linspace(x_min, x_max, n_grid)
        p = kde(x)

    cdf = np.concatenate(
        [[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))]
    )
    if cdf[-1] <= 0:
        raise ValueError(f"degenerate density loaded from {path}")
    cdf /= cdf[-1]

    cdf_u, idx = np.unique(cdf, return_index=True)
    x_u = x[idx]

    transform = _MonotonicInverseCDFTransform(cdf_u, x_u)
    return dist.TransformedDistribution(dist.Uniform(0.0, 1.0), [transform])


class SpinPriorMixin:
    """Mixin adding a configurable prior for `chi` to a qnm model class.

    Classes using this mixin should include, in `self.prior_kwargs`:
        chi_prior : str
            'uniform' (default), a name in `REGISTERED_CHI_PRIORS`
            (e.g. 'nrsur'), or a path to a custom KDE/samples file.
        chi_prior_path : str | None
            only needed to override a registered name with a custom file;
            normally left as None.

    Call `self.sample_chi()` inside `prior_sample()` instead of sampling
    `chi` directly from `dist.Uniform(chi_min, chi_max)`.
    """

    _chi_prior_dist = None

    def set_chi_prior(self, kind: str = "uniform", path: str | None = None,
                       **kws):
        self.prior_kwargs["chi_prior"] = kind
        if kind == "uniform":
            self._chi_prior_dist = None
            return
        # path override takes precedence; otherwise resolve `kind` itself
        # (a registered name or a path) inside load_chi_prior_from_file
        source = path or self.prior_kwargs.get("chi_prior_path") or kind
        self._chi_prior_dist = load_chi_prior_from_file(
            source,
            x_min=self.prior_kwargs.get("chi_min", 0.0),
            x_max=self.prior_kwargs.get("chi_max", 1.0),
            **kws,
        )

    def sample_chi(self):
        kind = self.prior_kwargs.get("chi_prior", "uniform")
        if kind == "uniform":
            d = dist.Uniform(
                self.prior_kwargs["chi_min"], self.prior_kwargs["chi_max"]
            )
            return numpyro.sample("chi", d)
        if self._chi_prior_dist is None:
            self.set_chi_prior(kind, self.prior_kwargs.get("chi_prior_path"))
        return numpyro.sample("chi", self._chi_prior_dist)