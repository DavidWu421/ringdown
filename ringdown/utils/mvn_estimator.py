import functools
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp

# ------------------------------------------------------------------
# Helper: build a custom-VJP function whose backward pass implements
#         the analytic gradients we derived above.
# ------------------------------------------------------------------
def _make_Z_single(f, u):
    """
    Returns a `Z_single(mu, L)` function (scalar output)
    with a custom-defined VJP so that JAX autodiff w.r.t. `mu`
    *and* the Cholesky factor of the **precision** (`L`) works.
    """
    @jax.custom_vjp
    def Z_single(mu, L):
        # forward pass – plain Monte-Carlo indicator
        x  = mu + jsp.solve_triangular(L, u.T, lower=True, trans="T").T  # (n,d)
        b  = (f(x) > 0).astype(x.dtype)                                  # (n,)
        return jnp.mean(b)                                               # scalar

    # ---------- forward & backward rules ----------
    def fwd(mu, L):
        x  = mu + jsp.solve_triangular(L, u.T, lower=True, trans="T").T
        b  = (f(x) > 0).astype(x.dtype)
        diff = x - mu                       # (n,d)
        return jnp.mean(b), (diff, b, L)    # save residuals

    def bwd(res, g):
        diff, b, L = res                    # (n,d) (n,) (d,d)
        n          = diff.shape[0]
        d          = diff.shape[1]

        # -------- grad w.r.t. mu --------
        mean_diff  = jnp.mean(b[:, None] * diff, axis=0)                 # (d,)
        P          = L @ L.T                                             # precision Σ⁻¹
        grad_mu    = P @ mean_diff                                       # (d,)

        # -------- grad w.r.t. L --------
        inv_L_T    = jnp.linalg.inv(L).T                                # L^{-T}
        diff_outer = diff[:, :, None] * diff[:, None, :]                # (n,d,d)
        score_L    = inv_L_T - jnp.matmul(diff_outer, L)                # (n,d,d)
        grad_L     = jnp.mean(b[:, None, None] * score_L, axis=0)        # (d,d)

        return (g * grad_mu, g * grad_L)  # (∂Z/∂mu, ∂Z/∂L)

    Z_single.defvjp(fwd, bwd)
    return Z_single
# ------------------------------------------------------------------


class MVNConstraintMC:
    """
    Monte-Carlo estimator for             Z = P[f(x)>0] ,
    where   x ~ 𝒩(μ, Σ)  and the user supplies L = chol(Σ⁻¹).

    Parameters
    ----------
    f   : callable      — constraint function; returns scalar or broadcast array
    n   : int           — number of stored standard-normal samples
    d   : int           — dimensionality
    key : jax.random.PRNGKey
    """

    def __init__(self, f, n, d, key):
        self.f   = f
        self.n   = n
        self.d   = d
        key_u, _ = jax.random.split(key)
        self.u   = jax.random.normal(key_u, (n, d))     # fixed N(0,I) draws

        # Build the custom-VJP scalar estimator for one (μ, L) pair
        self._Z_single = _make_Z_single(self.f, self.u)

    # ------------------------------------------------------------------
    # Public API --------------------------------------------------------
    # ------------------------------------------------------------------
    def Z(self, mu, L):
        """
        Monte-Carlo estimate of  Z(μ,Σ)  with gradients.

        Accepts broadcast shapes:
            mu : (d,)   or (B,d)
            L  : (d,d)  or (B,d,d)
        """
        if mu.ndim == 1:
            return self._Z_single(mu, L)               # scalar
        else:
            return jax.vmap(self._Z_single)(mu, L)     # (B,)