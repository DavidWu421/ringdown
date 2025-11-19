import jax
import jax.numpy as jnp
from .mode_coeffs import a_omega, a_gamma, b_omega, b_gamma

jax.config.update("jax_enable_x64", True)
"""
Method of calculating mdoes taken from  Mark et.al (2014)
https://arxiv.org/abs/1409.5800 and then fitted to a polynomial+log fit. The
included modes are listed (in order) in mode_list
"""
mode_list = jnp.array([[2, 0, 0], [2, 0, 1], [2, 1, 0], [2, 1, 1], [2, 2, 0],
                       [2, 2, 1], [2, 2, 2], [3, 2, 0], [3, 2, 1], [3, 3, 0],
                       [3, 3, 1], [4, 2, 0], [4, 4, 0]])

# The first set of 10 coefficients corresponds to the real coeffs while the
# second set corresponds to the imag coeffs. The first 5 are the polynomail
# coeffs, and the second 5 are the log coeffs. They then go in order of the
# mode_list. The fit is very sensitive to the fit parameters so float64 is used.

# Fit coefficients for the KN shifts
aij_omega = jnp.stack(a_omega, dtype=jnp.float64)
aij_gamma = jnp.stack(a_gamma, dtype=jnp.float64)
bij_omega = jnp.stack(b_omega, dtype=jnp.float64)
bij_gamma = jnp.stack(b_gamma, dtype=jnp.float64)


# # Main function for computing omega/gamma values
# def jnp_chiq_exact_factors(chi, Y0, aij, bij):
#     chi_powers = chi**(jnp.arange(3)[:, None] + 1)
#     num = jnp.einsum('ij,jk->ik', aij, chi_powers).flatten()
#     den = jnp.einsum('ij,jk->ik', bij, chi_powers).flatten()
#     return (Y0 * (1 + num) / (1 + den)).flatten()


def get_poly_chi_basis(chi):
    x = jnp.sqrt(1 - chi**2)
    return x**jnp.arange(5)[:, None]


def get_log_chi_basis(chi):
    x = jnp.sqrt(1 - chi**2)
    return jnp.log(x)**(jnp.arange(5)[:, None] + 1)


# Main function for computing omega/gamma values
def compute_shifts(chi, aij, bij):
    poly_term = jnp.einsum('ij,jk->ik', aij, get_poly_chi_basis(chi))
    log_term = jnp.einsum('ij,jk->ik', bij, get_log_chi_basis(chi))
    return poly_term + log_term


def get_charged_omega_shifts(chi, Q):
    omega = compute_shifts(chi, aij_omega, bij_omega).astype(jnp.complex64)
    gamma = compute_shifts(chi, aij_gamma, bij_gamma).astype(jnp.complex64)
    return (Q**2 * (omega + 1j * gamma)).flatten()
