import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


class MaxAmplitudeAtIndexSampler:
    """
    Sample standard-normal quadratures (Apx, Apy, Acx, Acy) for `nmodes`
    *and then* deterministically permute them so that the mode at `index`
    has the largest amplitude A.

    The returned tensors keep shape `(nmodes,)`, only their ordering changes,
    so downstream code that expects full-length arrays continues to work.

    Example
    -------
    sampler = MaxAmplitudeAtIndexSampler(nmodes=5, index=2)
    apx, apy, acx, acy = sampler.sample()
    # guaranteed: argmax(A(apx,apy,acx,acy)) == 2
    """

    def __init__(self, nmodes: int, index: int):
        if not (0 <= index < nmodes):
            raise ValueError("index must be in [0, nmodes)")
        self.nmodes = nmodes
        self.index = index

    # ------------------------------------------------------------------
    # public API: call inside a NumPyro model
    # ------------------------------------------------------------------
    def sample(self):
        n = self.nmodes
        i = self.index

        # 1. draw *all* quadratures
        apx = numpyro.sample("apx_unit_0", dist.Normal(0, 1), sample_shape=(n,))
        apy = numpyro.sample("apy_unit_0", dist.Normal(0, 1), sample_shape=(n,))
        acx = numpyro.sample("acx_unit_0", dist.Normal(0, 1), sample_shape=(n,))
        acy = numpyro.sample("acy_unit_0", dist.Normal(0, 1), sample_shape=(n,))

        # 2. compute amplitude A for every mode
        term1 = jnp.sqrt(jnp.square(acy + apx) + jnp.square(acx - apy))
        term2 = jnp.sqrt(jnp.square(acy - apx) + jnp.square(acx + apy))
        A     = 0.5 * (term1 + term2)                           # shape (n,)

        # 3. identify the winning mode and build a permutation that
        #    swaps it into position `i`
        j_max = jnp.argmax(A)                                   # scalar

        # record for diagnostics / debugging
        numpyro.deterministic("argmax_A_before_swap", j_max)
        numpyro.deterministic("A_values_before_swap", A)

        # if j_max == i, permutation is identity
        perm = jnp.arange(n)
        perm = perm.at[i].set(j_max)
        perm = perm.at[j_max].set(i)

        # 4. apply the same permutation to every quadrature vector
        apx_ = apx[perm]
        apy_ = apy[perm]
        acx_ = acx[perm]
        acy_ = acy[perm]

        apx = numpyro.deterministic("apx_unit", apx_)
        apy = numpyro.deterministic("apy_unit", apy_)
        acx = numpyro.deterministic("acx_unit", acx_)
        acy = numpyro.deterministic("acy_unit", acy_)

        # check (optional, useful in testing)
        numpyro.deterministic("A_values_after_swap",
                              0.5 * (jnp.sqrt(jnp.square(acy + apx)
                                              + jnp.square(acx - apy))
                                     + jnp.sqrt(jnp.square(acy - apx)
                                                + jnp.square(acx + apy))))
        return apx, apy, acx, acy
    
    def sample_single_pol(self):
        """
        Return (ax_unit, ay_unit), each of shape (nmodes,),
        with the guarantee that mode `index` has the largest
        amplitude A = sqrt(ax^2 + ay^2).

        Intended to replace the old block

            ax_unit = numpyro.sample("ax_unit", ...)
            ay_unit = numpyro.sample("ay_unit", ...)
            quads   = jnp.concatenate((ax_unit, ay_unit))
        """
        n, i = self.n, self.i

        # 1. draw both quadratures for all modes
        ax = numpyro.sample("ax_unit", dist.Normal(0, 1), sample_shape=(n,))
        ay = numpyro.sample("ay_unit", dist.Normal(0, 1), sample_shape=(n,))

        # 2. compute per-mode amplitude  A_m = sqrt(ax^2 + ay^2)
        A = jnp.sqrt(jnp.square(ax) + jnp.square(ay))      # shape (n,)

        # 3. build a permutation that brings the max-A mode to slot `i`
        j_max = jnp.argmax(A)                               # scalar
        perm  = jnp.arange(n)
        perm  = perm.at[i].set(j_max)
        perm  = perm.at[j_max].set(i)

        # 4. apply the same permutation to both quadrature vectors
        ax, ay = ax[perm], ay[perm]

        # 5. (optional diagnostics)
        numpyro.deterministic("argmax_A_values_before_swap", j_max)
        numpyro.deterministic("A_values_after_swap", A[perm])

        return ax, ay