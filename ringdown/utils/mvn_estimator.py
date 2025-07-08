import functools
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp

# ------------------------------------------------------------------
# Custom-VJP builder for a single (mu, L) pair
# ------------------------------------------------------------------
def _make_Z_single(weight_fn, u):
    """
    Build Z_single(mu, L) that returns  E[w(x)]
    and carries an analytic backward pass so that gradients wrt
    mu and the precision-Cholesky L are correct.

    * weight_fn(x) must return an array broadcastable to shape (n,)
      when x has shape (n, d).
    * u : (n, d)  – fixed standard-normal draws reused for every call.
    """
    @jax.custom_vjp
    def Z_single(mu, L):
        x = mu + jsp.solve_triangular(L, u.T, lower=True, trans="T").T  # (n,d)
        w = weight_fn(x)                                                # (n,)
        return jnp.mean(w)                                              # scalar

    # ---------------- forward & backward ---------------------------
    def fwd(mu, L):
        x = mu + jsp.solve_triangular(L, u.T, lower=True, trans="T").T
        w = weight_fn(x)#.reshape(-1)            # ensure (n,)
        diff = x - mu                           # (n,d)
        return jnp.mean(w), (diff, w, L)

    def bwd(res, g):
        diff, w, L = res
        P          = L @ L.T                                   # Σ⁻¹
        n          = diff.shape[0]

        # ---- ∂/∂mu -------------------------------------------------
        grad_mu = P @ jnp.mean(w[:, None] * diff, axis=0)      # (d,)

        # ---- ∂/∂L  (precision-Cholesky) ---------------------------
        inv_L_T    = jnp.linalg.inv(L).T                       # L^{-T}
        diff_outer = diff[:, :, None] * diff[:, None, :]       # (n,d,d)
        score_L    = inv_L_T - jnp.matmul(diff_outer, L)       # (n,d,d)
        grad_L     = jnp.mean(w[:, None, None] * score_L, axis=0)

        return (g * grad_mu, g * grad_L)

    Z_single.defvjp(fwd, bwd)
    return Z_single
# ------------------------------------------------------------------


class MVNMonteCarlo:
    """
    Monte-Carlo estimator of the weighted expectation

        Z(mu, Σ) = E_{x~N(mu,Σ)}[ w(x) ],

    with gradients wrt  mu  and  L = chol(Σ⁻¹).

    Parameters
    ----------
    weight_fn : callable
        Function w(x) returning weights with shape broadcastable to
        the leading sample dimension.  Use   w(x)=(f(x)>0).astype(...)
        for a constraint indicator.
    n : int
        Number of stored standard-normal samples.
    d : int
        Dimensionality of x.
    key : jax.random.PRNGKey
        PRNG key used once to generate the fixed u~N(0,I).
    """
    def __init__(self, weight_fn, n, d, key):
        self.weight_fn = weight_fn
        self.n         = n
        self.d         = d
        self.u         = jax.random.normal(key, (n, d))
        self._Z_single = _make_Z_single(self.weight_fn, self.u)

    # ----------------------------------------------------------------
    def Z(self, mu, L):
        """
        Estimate Z for a single (mu,L) or a batch:

            mu : (d,)      or (B,d)
            L  : (d,d)     or (B,d,d)

        Returns scalar or length-B array accordingly.
        """
        return (self._Z_single(mu, L)
                if mu.ndim == 1
                else jax.vmap(self._Z_single)(mu, L))