import jax
import jax.numpy as jnp
from .mode_coeffs import a_omega, a_gamma

jax.config.update("jax_enable_x64", True)
"""
Method of calculating mdoes taken from Cano et.al (2024)
https://arxiv.org/abs/2409.04517 and then fitted to a polynomial fit. The
included modes are listed (in order) in mode_list. This is currrently for even 
parity cubic gravity. Other higher order theories from this paper are to be
implemented
"""
mode_list = jnp.array([[2, 0, 0], [2, 0, 1], [2, 1, 0], [2, 1, 1], [2, 2, 0],
                       [2, 2, 1], [2, 2, 2], [3, 2, 0], [3, 2, 1], [3, 3, 0],
                       [3, 3, 1], [4, 2, 0], [4, 4, 0],
                       [-2, 0, 0], [-2, 0, 1], [-2, 1, 0], [-2, 1, 1], [-2, 2, 0],
                       [-2, 2, 1], [-2, 2, 2], [-3, 2, 0], [-3, 2, 1], [-3, 3, 0],
                       [-3, 3, 1], [-4, 2, 0], [-4, 4, 0]])

# The first set of coefficients corresponds to the real coeffs while the
# second set corresponds to the imag coeffs. They then go in order of the
# mode_list.

# Fit coefficients for the KN shifts
aij_omega = jnp.stack(a_omega, dtype=jnp.float64)
aij_gamma = jnp.stack(a_gamma, dtype=jnp.float64)


# # Main function for computing omega/gamma values
# def jnp_chi_cubic_exact_factors(chi, ai):
#     chi_powers = chi**(jnp.arange(16)[:, None] + 1)
#     num = jnp.einsum('i,i->', ai, chi_powers).flatten()
#     return num.flatten()


def get_poly_chi_basis(chi):
    x = chi
    return x**(jnp.arange(16)[:, None])


# Main function for computing omega/gamma values
def compute_shifts(chi, aij):
    poly_term = jnp.einsum('ij,jk->ik', aij, get_poly_chi_basis(chi))
    return poly_term


def get_odd_cubic_omega_shifts(chi, alpha):
    omega = compute_shifts(chi, aij_omega).astype(jnp.complex64)
    gamma = compute_shifts(chi, aij_gamma).astype(jnp.complex64)
    return (alpha * (omega + 1j * gamma)).flatten()