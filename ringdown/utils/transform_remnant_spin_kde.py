"""Shift and rescale a prior on the remnant spin in logit space.

This is the machinery behind the ``chi_prior_shift`` and ``chi_prior_scale``
options of :func:`ringdown.model.make_model`, used to test how sensitive a
result is to the remnant-spin prior (``ringdown/priors/remnant_spin_kde.txt``).

The density ``p(x)`` on (0, 1) is mapped to logit space, ``y = log(x/(1-x))``,
where it has density ``q(y) = p(x) x (1-x)``.  With ``m`` and ``sd`` the mean
and standard deviation of ``q``, the perturbed density is

    q'(y) = (1/scale) q(m + (y - m - shift_frac*sd)/scale),

i.e., ``q`` is shifted by ``shift_frac`` standard deviations (negative =
towards lower spin) and broadened by ``scale`` about its mean (``scale > 1``
broadens, ``< 1`` narrows); it is then mapped back to ``x`` and renormalized.
``shift_frac=0, scale=1`` returns the original density.

Only numpy is needed, so the functions here can be used on their own.  The
module can also be run from the command line to write a transformed file::

    python -m ringdown.utils.transform_remnant_spin_kde \\
        --shift -0.05 --scale 1.05 --out my_prior.txt --plot

and the result used with ``chi_prior='my_prior.txt'``.
"""

__all__ = ["transform_kde", "load_kde", "write_transformed_kde"]

import argparse

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


def transform_kde(x, p, shift_frac=0.0, scale=1.0):
    """Shift and rescale a density on (0, 1) in logit space.

    Arguments
    ---------
    x : array
        grid on [0, 1]; need not include the end points and must be sorted.
    p : array
        density on `x` (non-negative).
    shift_frac : float
        shift in units of the standard deviation of the *original* density in
        logit space (negative = towards lower values of `x`).
    scale : float
        factor by which the logit-space density is broadened about its mean
        (``> 1`` broadens, ``< 1`` narrows); must be positive.

    Returns
    -------
    p_new : array
        perturbed density on the same grid `x`, normalized to unit integral
        over `x`.  Grid points where the logit is undefined (``x`` of 0 or 1)
        or the original density is zero take the value of the nearest
        well-defined grid point (these are far in the tails and tiny).
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    if scale <= 0:
        raise ValueError("scale must be positive")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    if np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError("p must be finite and non-negative")

    p0 = p / _trapz(p, x)
    if shift_frac == 0 and scale == 1:
        return p0

    ok = (x > 0) & (x < 1) & (p > 0)  # logit undefined at 0 and 1
    if ok.sum() < 3:
        raise ValueError("need at least 3 grid points inside (0, 1)")
    xs, ps = x[ok], p[ok]

    y = np.log(xs / (1 - xs))  # logit transform
    q = ps * xs * (1 - xs)  # density in logit space (Jacobian)
    q = q / _trapz(q, y)

    m = _trapz(y * q, y)  # pivot: mean in logit space
    sd = np.sqrt(_trapz((y - m) ** 2 * q, y))

    d = shift_frac * sd
    yy = m + (y - m - d) / scale
    with np.errstate(divide="ignore"):
        logq = np.interp(yy, y, np.log(q), left=-np.inf, right=-np.inf)
    pn = np.exp(logq) / scale / (xs * (1 - xs))

    # put back on the original grid; excluded points copy their nearest
    # included neighbour
    p_new = np.interp(x, xs, pn)
    p_new = p_new / _trapz(p_new, x)
    return p_new


def load_kde(spec):
    """Load a two-column (x, density) file given as a path or as the name of a
    file in ``ringdown/priors/``.  Returns sorted ``(x, p)`` arrays."""
    from .tabulated_prior import _resolve_path  # lazy: avoids a circular import

    path = _resolve_path(spec)
    x, p = np.loadtxt(path, comments="#", unpack=True)
    order = np.argsort(x)
    return x[order], p[order]


def write_transformed_kde(spec, out, shift_frac=0.0, scale=1.0):
    """Transform the KDE in `spec` and write it as a two-column text file."""
    x, p = load_kde(spec)
    p_new = transform_kde(x, p, shift_frac=shift_frac, scale=scale)
    header = (
        f"chi_f pdf (logit-space transform of {spec}: "
        f"shift_frac={shift_frac}, scale={scale})"
    )
    np.savetxt(out, np.column_stack([x, p_new]), header=header)
    return x, p_new


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--kde", default="remnant_spin_kde", help="input KDE (name or path)"
    )
    ap.add_argument("--shift", type=float, default=0.0, help="shift_frac")
    ap.add_argument("--scale", type=float, default=1.0, help="scale")
    ap.add_argument("--out", required=True, help="output text file")
    ap.add_argument("--plot", action="store_true", help="plot old vs. new")
    args = ap.parse_args(argv)

    x, p_new = write_transformed_kde(args.kde, args.out, args.shift, args.scale)
    x0, p0 = load_kde(args.kde)
    mean = lambda pp: _trapz(x * pp, x)
    print(
        f"wrote {args.out}; mean chi: original {mean(p0 / _trapz(p0, x)):.4f}, "
        f"transformed {mean(p_new):.4f}"
    )
    if args.plot:
        import matplotlib.pyplot as plt

        plt.plot(x0, p0 / _trapz(p0, x0), "k", lw=2, label="original")
        plt.plot(
            x, p_new, "r", lw=2, label=f"shift={args.shift}, scale={args.scale}"
        )
        plt.xlabel(r"$\chi_f$")
        plt.ylabel("density")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()