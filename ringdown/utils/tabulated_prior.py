"""Tabulated 1D priors (e.g., on remnant spin) read from two-column text files."""

__all__ = ["make_tabulated_log_prior"]

from pathlib import Path
import numpy as np
import jax.numpy as jnp

PRIOR_DIR = Path(__file__).parent.parent / "priors"
_LOG_FLOOR = 1e-300


def _resolve_path(spec) -> Path:
    """Accept a path, or the name of a file in ringdown/priors/
    (with or without the .txt extension)."""
    p = Path(spec)
    if p.is_file():
        return p
    for candidate in (PRIOR_DIR / str(spec), PRIOR_DIR / f"{spec}.txt"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"could not find prior '{spec}' as a path or in {PRIOR_DIR}"
    )


def make_tabulated_log_prior(spec):
    """Return a JAX-compatible function ``log_prior(x)`` from a two-column
    text file (grid, pdf); lines starting with '#' are ignored.

    The pdf is normalized and linearly interpolated. Outside the tabulated
    range it is clamped to the edge values, so the sampling bounds
    (chi_min, chi_max) should sit inside the tabulated range.
    Returns None if ``spec`` is None or 'uniform'.
    """
    if spec is None or (isinstance(spec, str) and spec.lower() == "uniform"):
        return None

    path = _resolve_path(spec)
    x, p = np.loadtxt(path, comments="#", unpack=True)
    order = np.argsort(x)
    x, p = x[order], p[order]
    if np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError(f"invalid pdf values in {path}")
    norm = np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(x))  # trapezoid rule
    if norm <= 0:
        raise ValueError(f"pdf in {path} has non-positive integral")
    x_j, p_j = jnp.asarray(x), jnp.asarray(p / norm)

    def log_prior(chi):
        return jnp.log(jnp.maximum(jnp.interp(chi, x_j, p_j), _LOG_FLOOR))

    log_prior.path = str(path)
    return log_prior